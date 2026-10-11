"""EDA helpers + readmission prediction model. Run: python src/analysis.py"""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hospital_patients.csv"
FIGS = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
TARGET = "readmitted_30d"
THRESHOLD = 0.20  # flag patients with >=20% predicted risk (overall rate is ~14%)
NUM = ["age", "bmi", "prior_admissions", "length_of_stay", "num_medications",
       "num_lab_procedures", "hba1c", "systolic_bp", "diabetes", "hypertension"]
CAT = ["gender", "smoker", "insurance", "discharge_to", "followup_scheduled"]


def load(path=DATA):
    # keep_default_na=False so text categories are never mistaken for missing values
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def quality(df):
    m = df.isna().sum()
    return pd.DataFrame({"missing": m, "missing_%": (m / len(df) * 100).round(2)}), int(df.duplicated().sum())


def rate_by(df, col):
    return df.groupby(col, observed=True)[TARGET].agg(rate="mean", n="size").assign(rate=lambda d: (d.rate * 100).round(1))


def add_bands(df):
    d = df.copy()
    d["age_band"] = pd.cut(d.age, [17, 40, 55, 70, 95], labels=["18-40", "41-55", "56-70", "71+"])
    d["prior_band"] = d.prior_admissions.clip(upper=4).astype(str).replace("4", "4+")
    d["los_band"] = pd.cut(d.length_of_stay, [0, 3, 6, 10, 30], labels=["1-3", "4-6", "7-10", "11+"])
    return d


def tests(df):
    out = {}
    out["chi2_followup_p"] = stats.chi2_contingency(pd.crosstab(df.followup_scheduled, df[TARGET]))[1]
    out["chi2_diabetes_p"] = stats.chi2_contingency(pd.crosstab(df.diabetes, df[TARGET]))[1]
    a, b = df[df[TARGET] == 1], df[df[TARGET] == 0]
    out["ttest_age_p"] = stats.ttest_ind(a.age, b.age).pvalue
    out["ttest_prior_p"] = stats.ttest_ind(a.prior_admissions, b.prior_admissions).pvalue
    return out


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("model", model)])


def train(df):
    X, y = df[NUM + CAT], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    models = {
        "Logistic regression": build_pipeline(LogisticRegression(max_iter=1000)),
        "Random forest": build_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                               random_state=42, n_jobs=-1))}
    metrics, fitted = {}, {}
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        metrics[name] = dict(roc_auc=roc_auc_score(yte, proba), accuracy=accuracy_score(yte, pred),
                             precision=precision_score(yte, pred), recall=recall_score(yte, pred), f1=f1_score(yte, pred))
        fitted[name] = pipe
    return fitted, metrics, (Xte, yte)


def make_figures(df, fitted, test, best):
    FIGS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    avg, d = df[TARGET].mean() * 100, add_bands(df)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), ["age_band", "prior_band", "los_band", "discharge_to", "insurance", "followup_scheduled"]):
        r = rate_by(d, col).reset_index()
        sns.barplot(data=r, x=col, y="rate", ax=ax, color="#0f766e")
        ax.axhline(avg, color="k", ls="--", lw=1); ax.set(ylabel="Readmission rate (%)", xlabel=col)
    fig.suptitle("30-day readmission rate by group (dashed = overall)"); fig.tight_layout()
    fig.savefig(FIGS / "readmission_by_group.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[NUM + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout(); fig.savefig(FIGS / "correlation_heatmap.png", dpi=120); plt.close(fig)

    Xte, yte = test
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, Xte, yte, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("ROC curves (test set)")
    fig.tight_layout(); fig.savefig(FIGS / "roc_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    pred = (fitted[best].predict_proba(Xte)[:, 1] >= THRESHOLD).astype(int)
    ConfusionMatrixDisplay.from_predictions(yte, pred, ax=ax, cmap="Blues", colorbar=False,
                                            display_labels=["Not readmitted", "Readmitted"])
    ax.set_title(f"{best}: confusion matrix (threshold {THRESHOLD:.0%})"); fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrix.png", dpi=120); plt.close(fig)

    imp = importance(fitted[best], Xte, yte)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.sort_values().plot.barh(ax=ax, color="#0f766e"); ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title(f"Permutation importance ({best})"); fig.tight_layout()
    fig.savefig(FIGS / "feature_importance.png", dpi=120); plt.close(fig)


def importance(pipe, Xte, yte):
    r = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=42)
    return pd.Series(r.importances_mean, index=Xte.columns).sort_values(ascending=False)


if __name__ == "__main__":
    df = load()
    q, dups = quality(df)
    print("Shape:", df.shape, "| duplicates:", dups); print(q[q.missing > 0], "\n")
    print("Overall 30-day readmission: %.1f%%\n" % (df[TARGET].mean() * 100))
    d = add_bands(df)
    for c in ["age_band", "prior_band", "los_band", "discharge_to", "insurance", "followup_scheduled", "diabetes", "smoker"]:
        print(rate_by(d, c), "\n")
    print(tests(df), "\n")
    fitted, metrics, test = train(df)
    print(pd.DataFrame(metrics).T.round(3), "\n")
    best = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    imp = importance(fitted[best], *test)
    print("Best model by ROC-AUC:", best); print(imp.round(4), "\n")
    make_figures(df, fitted, test, best)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODELS / "best_model.joblib")
    json.dump({"best": best, "threshold": THRESHOLD, "metrics": metrics, "importance": imp.to_dict()}, open(MODELS / "metrics.json", "w"), indent=2)
    print("Saved figures to", FIGS, "and model to", MODELS)

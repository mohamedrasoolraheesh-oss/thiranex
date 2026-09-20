#!/usr/bin/env python3
"""
Predictive Modeling Using Machine Learning
==========================================
Trains, tests and compares three supervised-learning algorithms:
Linear (or Logistic) Regression, Decision Tree and Random Forest.

  * Classification demo : Breast Cancer Wisconsin  -> confusion matrices + ROC curves
  * Regression demo     : Diabetes progression     -> predicted-vs-actual plots

Usage
-----
    python predictive_modeling.py                        # run both built-in demos
    python predictive_modeling.py --csv data.csv --target price
    python predictive_modeling.py --csv data.csv --target churn --task classification

Every figure and a results table are written to the --outdir folder (default: results/).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to disk; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, auc, accuracy_score, classification_report, f1_score,
    mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score,
    roc_curve, root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler, label_binarize,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
COLORS = ["#2a6f97", "#e07a5f", "#3d9970"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def demo_classification():
    data = load_breast_cancer(as_frame=True)
    # Label malignant tumours as the positive class (alphabetical: benign=0, malignant=1)
    y = pd.Series(np.where(data.target == 0, "malignant", "benign"), name="diagnosis")
    return data.data, y, "Breast Cancer Wisconsin"


def demo_regression():
    data = load_diabetes(as_frame=True)
    return data.data, data.target, "Diabetes disease progression"


def load_csv(path, target, max_categories=50):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found. Available: {list(df.columns)}")
    df = df.dropna(subset=[target])
    y, X = df[target], df.drop(columns=[target])
    # Drop ID-like / free-text columns: too many distinct values to one-hot encode
    text_cols = X.select_dtypes(exclude="number").columns
    drop = [c for c in text_cols if X[c].nunique() > max_categories]
    if drop:
        print(f"Dropping high-cardinality columns: {drop}")
        X = X.drop(columns=drop)
    return X, y, Path(path).stem


def infer_task(y):
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    if y.nunique() <= 10 and np.allclose(y, np.round(y)):
        return "classification"
    return "regression"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _as_object(df):
    # pandas 3 reads text as its new "str" dtype, which scikit-learn's imputer misreads as numeric
    return df.astype(object)


def make_preprocessor(X):
    """Impute missing values, scale numbers, one-hot encode categories.
    Lives inside the Pipeline, so it is fit on training folds only (no data leakage)."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("to_object", FunctionTransformer(_as_object, feature_names_out="one-to-one")),
                          ("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def classifiers(seed):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),  # linear model for classes
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                random_state=seed, n_jobs=-1),
    }


def regressors(seed):
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                               random_state=seed, n_jobs=-1),
    }


def build(X, model):
    return Pipeline([("prep", make_preprocessor(X)), ("model", model)])


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_feature_importance(pipe, path, title, top=10):
    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    imp = imp.nlargest(top).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color=COLORS[0])
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title(title)
    save(fig, path)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def run_classification(X, y, name, out, seed, test_size):
    print(f"\n{'=' * 72}\nCLASSIFICATION | {name}\n{'=' * 72}")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    labels = [str(c) for c in le.classes_]
    k = len(labels)
    counts = {label: int(n) for label, n in zip(labels, np.bincount(y_enc))}
    print(f"{len(X)} samples, {X.shape[1]} features, classes: {counts}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=test_size, stratify=y_enc, random_state=seed)
    print(f"Train/test split: {len(X_tr)} / {len(X_te)}")
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    avg = "binary" if k == 2 else "weighted"

    rows, fitted = [], {}
    for mname, model in classifiers(seed).items():
        pipe = build(X, model)
        cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)
        roc_auc = (roc_auc_score(y_te, proba[:, 1]) if k == 2
                   else roc_auc_score(y_te, proba, multi_class="ovr", average="macro"))
        rows.append({
            "Model": mname,
            "CV accuracy": cv_acc.mean(), "CV std": cv_acc.std(),
            "Test accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, average=avg, zero_division=0),
            "Recall": recall_score(y_te, pred, average=avg, zero_division=0),
            "F1": f1_score(y_te, pred, average=avg, zero_division=0),
            "ROC AUC": roc_auc,
        })
        fitted[mname] = (pipe, pred, proba)

    results = pd.DataFrame(rows)
    results.to_csv(out / "classification_results.csv", index=False)
    print("\n" + results.round(3).to_string(index=False))

    # Model selection uses training-set cross-validation only; the test set is for reporting.
    best = results.loc[results["CV accuracy"].idxmax(), "Model"]
    print(f"\nBest model by cross-validation: {best}. Test-set report:")
    print(classification_report(y_te, fitted[best][1], target_names=labels, digits=3))

    # Confusion matrices
    fig, axes = plt.subplots(1, len(fitted), figsize=(4.8 * len(fitted), 4.6))
    for ax, (mname, (_, pred, _)) in zip(np.atleast_1d(axes), fitted.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_te, pred, labels=range(k), display_labels=labels,
            cmap="Blues", colorbar=False, ax=ax)
        ax.grid(False)
        ax.set_title(f"{mname}\naccuracy = {accuracy_score(y_te, pred):.3f}")
    save(fig, out / "classification_confusion_matrices.png")

    # ROC curves (binary: standard; multiclass: micro-average)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for color, (mname, (_, _, proba)) in zip(COLORS, fitted.items()):
        if k == 2:
            fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
        else:
            fpr, tpr, _ = roc_curve(label_binarize(y_te, classes=range(k)).ravel(), proba.ravel())
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{mname} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves" + ("" if k == 2 else " (micro-average)"))
    ax.legend(loc="lower right")
    save(fig, out / "classification_roc_curves.png")

    plot_feature_importance(fitted["Random Forest"][0],
                            out / "classification_feature_importance.png",
                            "Random Forest: top 10 features")


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def run_regression(X, y, name, out, seed, test_size):
    print(f"\n{'=' * 72}\nREGRESSION | {name}\n{'=' * 72}")
    print(f"{len(X)} samples, {X.shape[1]} features, target range "
          f"[{y.min():.1f}, {y.max():.1f}]")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    print(f"Train/test split: {len(X_tr)} / {len(X_te)}")
    cv = KFold(5, shuffle=True, random_state=seed)

    rows, fitted = [], {}
    for mname, model in regressors(seed).items():
        pipe = build(X, model)
        cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": mname,
            "CV R2": cv_r2.mean(), "CV std": cv_r2.std(),
            "Test R2": r2_score(y_te, pred),
            "RMSE": root_mean_squared_error(y_te, pred),
            "MAE": mean_absolute_error(y_te, pred),
        })
        fitted[mname] = (pipe, pred)

    results = pd.DataFrame(rows)
    results.to_csv(out / "regression_results.csv", index=False)
    print("\n" + results.round(3).to_string(index=False))
    best = results.loc[results["CV R2"].idxmax(), "Model"]
    print(f"\nBest model by cross-validation: {best}")

    # Predicted vs actual
    fig, axes = plt.subplots(1, len(fitted), figsize=(4.8 * len(fitted), 4.6), sharex=True, sharey=True)
    all_vals = np.concatenate([np.asarray(y_te)] + [p for _, p in fitted.values()])
    lo, hi = all_vals.min(), all_vals.max()
    for ax, color, (mname, (_, pred)) in zip(np.atleast_1d(axes), COLORS, fitted.items()):
        ax.scatter(y_te, pred, s=22, alpha=0.7, color=color, edgecolor="white", linewidth=0.4)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xlabel="Actual", title=f"{mname}\nR² = {r2_score(y_te, pred):.3f}")
    np.atleast_1d(axes)[0].set_ylabel("Predicted")
    save(fig, out / "regression_predicted_vs_actual.png")

    plot_feature_importance(fitted["Random Forest"][0],
                            out / "regression_feature_importance.png",
                            "Random Forest: top 10 features")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", help="path to your own CSV (omit to run the built-in demos)")
    p.add_argument("--target", help="column to predict (required with --csv)")
    p.add_argument("--task", choices=["classification", "regression"],
                   help="problem type (default: auto-detect from the target column)")
    p.add_argument("--test-size", type=float, default=0.2, help="fraction held out for testing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    if args.csv and not args.target:
        p.error("--target is required when using --csv")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    runners = {"classification": run_classification, "regression": run_regression}

    if args.csv:
        X, y, name = load_csv(args.csv, args.target)
        task = args.task or infer_task(y)
        print(f"Task: {task}" + ("" if args.task else " (auto-detected)"))
        runners[task](X, y, name, out, args.seed, args.test_size)
    else:
        runners["classification"](*demo_classification(), out, args.seed, args.test_size)
        runners["regression"](*demo_regression(), out, args.seed, args.test_size)
    print(f"\nDone. Outputs are in: {out.resolve()}")


if __name__ == "__main__":
    main()

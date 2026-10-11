"""Reusable EDA functions + a CLI that prints results and saves figures."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telecom_churn_dataset.csv"
FIGS = ROOT / "reports" / "figures"
CATS = ["contract", "internet_service", "autopay", "region"]
NUMS = ["age", "tenure_months", "monthly_charges", "support_tickets", "total_charges"]


def load(path=DATA) -> pd.DataFrame:
    # keep_default_na=False so the category "None" (no internet) is not read as missing
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def quality(df):
    miss = df.isna().sum()
    return pd.DataFrame({"missing": miss, "missing_%": (miss / len(df) * 100).round(2)}), int(df.duplicated().sum())


def clean(df):
    out = df.copy()
    for c in ["age", "total_charges"]:
        out[c] = out[c].fillna(out[c].median())  # <5% missing, so median imputation is safe
    return out


def summary(df):
    s = df[NUMS].describe().T[["mean", "50%", "std", "min", "max"]]
    s["skew"] = df[NUMS].skew()
    return s.round(2)


def churn_by(df, col):
    return df.groupby(col, observed=True).churn.agg(rate="mean", n="size").assign(rate=lambda d: (d.rate * 100).round(1))


def tenure_bands(df):
    d = df.assign(tenure_band=pd.cut(df.tenure_months, [0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]))
    return churn_by(d, "tenure_band")


def ticket_bands(df):
    return churn_by(df.assign(tickets=df.support_tickets.clip(upper=4)), "tickets")


def two_way(df):
    return (df.pivot_table(index="contract", columns="internet_service", values="churn", aggfunc="mean") * 100).round(1)


def correlations(df):
    return df[NUMS + ["churn"]].corr(method="pearson").round(3)


def tests(df):
    chi = stats.chi2_contingency(pd.crosstab(df.contract, df.churn))
    t = stats.ttest_ind(df[df.churn == 1].tenure_months, df[df.churn == 0].tenure_months)
    return {"chi2_contract_vs_churn_p": chi[1], "ttest_tenure_p": t.pvalue, "ttest_tenure_stat": t.statistic}


def make_figures(df, outdir=FIGS):
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    avg = df.churn.mean() * 100
    for col in CATS:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        d = churn_by(df, col).reset_index()
        sns.barplot(data=d, x=col, y="rate", ax=ax, color="#3b3bd6")
        ax.axhline(avg, color="k", ls="--", lw=1)
        ax.set(ylabel="Churn rate (%)", title=f"Churn by {col}")
        fig.tight_layout(); fig.savefig(outdir / f"churn_by_{col}.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(correlations(df), annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix"); fig.tight_layout(); fig.savefig(outdir / "correlation_heatmap.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, c in zip(axes, ["tenure_months", "monthly_charges", "total_charges"]):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, color="#3b3bd6"); ax.set_title(c)
    fig.tight_layout(); fig.savefig(outdir / "distributions.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(data=df, x="churn", y="tenure_months", ax=ax, hue="churn", legend=False)
    ax.set_title("Tenure by churn"); fig.tight_layout(); fig.savefig(outdir / "tenure_boxplot.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raw = load()
    q, dups = quality(raw)
    print("Shape:", raw.shape, "| duplicates:", dups); print(q[q.missing > 0], "\n")
    df = clean(raw)
    print(summary(df), "\n")
    print("Overall churn: %.1f%%\n" % (df.churn.mean() * 100))
    for c in CATS: print(churn_by(df, c), "\n")
    print(tenure_bands(df), "\n", ticket_bands(df), "\n", two_way(df), "\n")
    print(correlations(df)["churn"], "\n", tests(df))
    make_figures(df)
    print(f"\nFigures saved to {FIGS}")

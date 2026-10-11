"""Generate a synthetic telecom customer churn dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "telecom_churn_dataset.csv"


def generate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[.45, .35, .20])
    tenure = np.clip(rng.gamma(2.2, 14, n), 1, 72).round().astype(int)
    tenure = np.where(contract == "Two year", np.clip(tenure + 10, 1, 72), tenure)
    base = {"Fiber": 75, "DSL": 50, "None": 22}
    monthly = np.clip(np.array([base[i] for i in internet]) + rng.normal(0, 8, n), 15, 130).round(2)
    tickets = rng.poisson(np.where(internet == "Fiber", 1.6, 1.0))
    age = np.clip(rng.normal(44, 14, n), 18, 85).round().astype(int)
    autopay = rng.choice(["Yes", "No"], n, p=[.45, .55])
    region = rng.choice(["North", "South", "East", "West"], n)

    # Churn probability: the hidden "ground truth" the EDA should rediscover
    z = (-0.6
         + np.select([contract == "Month-to-month", contract == "One year"], [1.3, -0.2], -1.4)
         - 0.035 * tenure + 0.012 * (monthly - 60) + 0.35 * tickets
         - 0.5 * (autopay == "Yes")
         + np.where((internet == "Fiber") & (contract == "Month-to-month"), 0.4, 0))
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame(dict(
        customer_id=[f"C{i:04d}" for i in range(n)], age=age, region=region,
        tenure_months=tenure, contract=contract, internet_service=internet,
        autopay=autopay, monthly_charges=monthly, support_tickets=tickets, churn=churn))
    df["total_charges"] = (df.tenure_months * df.monthly_charges * rng.normal(1, .03, n)).round(2)
    # Inject missing values so the cleaning step has something to find
    df.loc[rng.choice(n, 70, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 45, replace=False), "total_charges"] = np.nan
    return df


if __name__ == "__main__":
    df = generate()
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} cols to {OUT}")

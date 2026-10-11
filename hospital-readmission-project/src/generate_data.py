"""Generate a synthetic hospital patient dataset for 30-day readmission analysis."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "hospital_patients.csv"


def generate(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(62, 16, n), 18, 95).round().astype(int)
    gender = rng.choice(["Female", "Male"], n)
    bmi = np.clip(rng.normal(28.5, 5.5, n) + (age - 60) * 0.02, 16, 50).round(1)
    smoker = rng.choice(["Yes", "No"], n, p=[.22, .78])
    diabetes = (rng.random(n) < 0.12 + 0.004 * np.clip(age - 40, 0, None) + 0.004 * np.clip(bmi - 25, 0, None)).astype(int)
    hypertension = (rng.random(n) < 0.15 + 0.006 * np.clip(age - 35, 0, None)).astype(int)
    prior = rng.poisson(0.6 + 0.02 * np.clip(age - 50, 0, None) + 0.8 * diabetes)
    los = np.clip(rng.gamma(2.0, 2.2, n) + 1.2 * diabetes + 0.04 * np.clip(age - 60, 0, None), 1, 30).round().astype(int)
    meds = np.clip(rng.poisson(8 + 2.5 * diabetes + 1.5 * hypertension + 0.2 * los), 1, 40)
    labs = np.clip(rng.normal(40 + 1.5 * los, 12, n), 1, 100).round().astype(int)
    hba1c = np.clip(rng.normal(5.5 + 2.0 * diabetes, 0.8, n), 4, 14).round(1)
    sbp = np.clip(rng.normal(122 + 16 * hypertension + 0.2 * (age - 60), 14, n), 85, 220).round().astype(int)
    insurance = rng.choice(["Private", "Medicare", "Medicaid", "Uninsured"], n, p=[.35, .40, .17, .08])
    discharge = rng.choice(["Home", "Rehab", "Nursing facility"], n, p=[.65, .15, .20])
    followup = rng.choice(["Yes", "No"], n, p=[.62, .38])

    z = (-3.1 + 0.018 * (age - 60) + 0.45 * prior + 0.07 * los + 0.03 * (meds - 10)
         + 0.55 * diabetes + 0.25 * hypertension + 0.30 * (smoker == "Yes")
         + 0.30 * np.clip(hba1c - 6.5, 0, None) * diabetes
         + np.select([discharge == "Nursing facility", discharge == "Rehab"], [0.45, 0.2], 0.0)
         + np.select([insurance == "Uninsured", insurance == "Medicaid"], [0.5, 0.25], 0.0)
         - 0.85 * (followup == "Yes"))
    readmit = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame(dict(
        patient_id=[f"P{i:05d}" for i in range(n)], age=age, gender=gender, bmi=bmi, smoker=smoker,
        diabetes=diabetes, hypertension=hypertension, prior_admissions=prior, length_of_stay=los,
        num_medications=meds, num_lab_procedures=labs, hba1c=hba1c, systolic_bp=sbp,
        insurance=insurance, discharge_to=discharge, followup_scheduled=followup, readmitted_30d=readmit))
    # realistic missingness: HbA1c is often not measured, BMI sometimes not recorded
    df.loc[rng.choice(n, int(n * 0.10), replace=False), "hba1c"] = np.nan
    df.loc[rng.choice(n, int(n * 0.04), replace=False), "bmi"] = np.nan
    return df


if __name__ == "__main__":
    df = generate()
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} cols to {OUT}")

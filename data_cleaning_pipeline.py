"""
Data Cleaning & Visualization Project
======================================
Step-by-step pipeline: inspect -> clean -> engineer features -> analyze -> visualize.
Run this file top to bottom; each section prints what it found and what it did.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
BASE_DIR = Path(__file__).resolve().parent          # folder this script lives in
RAW_PATH = BASE_DIR / "raw_ecommerce_sales.csv"      # put the raw CSV next to this script

# =====================================================================
# STEP 1: LOAD & INSPECT
# =====================================================================
print("=" * 70)
print("STEP 1: LOAD & INSPECT")
print("=" * 70)

df = pd.read_csv(RAW_PATH)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
n_dupes_exact = df.duplicated().sum()
print(f"\nFully duplicated rows: {n_dupes_exact}")

report = {"raw_rows": int(df.shape[0])}

# =====================================================================
# STEP 2: REMOVE DUPLICATES
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: REMOVE DUPLICATES")
print("=" * 70)

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - len(df)} exact duplicate rows -> {len(df)} rows remain")
report["duplicates_removed"] = int(before - len(df))

# =====================================================================
# STEP 3: STANDARDIZE TEXT FIELDS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: STANDARDIZE TEXT FORMATTING")
print("=" * 70)

text_cols = ["product_category", "city", "payment_method", "order_status", "customer_name"]
for col in text_cols:
    before_unique = df[col].nunique(dropna=True)
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].where(df[col] != "nan", np.nan)
    df[col] = df[col].apply(lambda v: v.title() if isinstance(v, str) else v)
    after_unique = df[col].nunique(dropna=True)
    print(f"  {col}: {before_unique} unique values -> {after_unique} after trim/casefix")

# fix a known title-case quirk: "Home & Kitchen" -> "Home & Kitchen" stays fine with .title()
df["product_category"] = df["product_category"].replace({"Home & Kitchen".title(): "Home & Kitchen"})

# clean emails: lowercase, flag malformed (no "@")
df["email"] = df["email"].astype(str).str.strip().str.lower().replace("nan", np.nan)
malformed_email = df["email"].notna() & ~df["email"].str.contains("@", na=False)
print(f"  Malformed emails found (no '@'): {malformed_email.sum()} -> set to missing")
df.loc[malformed_email, "email"] = np.nan

# =====================================================================
# STEP 4: PARSE MIXED DATE FORMATS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: PARSE MIXED DATE FORMATS")
print("=" * 70)

def parse_mixed_date(s):
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%b %d, %Y", "%m-%d-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

df["order_date"] = df["order_date"].apply(parse_mixed_date)
unparsed = df["order_date"].isna().sum()
print(f"Parsed order_date into datetime. Unparseable dates: {unparsed}")

# =====================================================================
# STEP 5: HANDLE MISSING VALUES
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: HANDLE MISSING VALUES")
print("=" * 70)

miss_before = df.isna().sum()
print("Missing before:\n", miss_before[miss_before > 0])

df["customer_name"] = df["customer_name"].fillna("Unknown Customer")
df["city"] = df["city"].fillna("Unknown")
df["region"] = df["region"].fillna("Unknown")
df["discount_percent"] = df["discount_percent"].fillna(df["discount_percent"].median())
# impute missing unit_price using the median price for that product's category
df["unit_price"] = df.groupby("product_category")["unit_price"].transform(
    lambda s: s.fillna(s.median())
)
# email left as missing (NaN) -- not something we should fabricate

miss_after = df.isna().sum()
print("\nMissing after (email intentionally left as-is):\n", miss_after[miss_after > 0])
report["missing_before"] = {k: int(v) for k, v in miss_before.items() if v > 0}
report["missing_after"] = {k: int(v) for k, v in miss_after.items() if v > 0}

# =====================================================================
# STEP 6: HANDLE OUTLIERS (IQR method)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: HANDLE OUTLIERS")
print("=" * 70)

# invalid quantities (<= 0) are data-entry errors -> drop
bad_qty = df["quantity"] <= 0
print(f"Rows with non-positive quantity (data-entry errors): {bad_qty.sum()} -> dropped")
df = df[~bad_qty].reset_index(drop=True)
report["invalid_qty_dropped"] = int(bad_qty.sum())

def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

qty_lo, qty_hi = iqr_bounds(df["quantity"])
price_lo, price_hi = iqr_bounds(df["unit_price"])
print(f"Quantity IQR bounds: [{qty_lo:.2f}, {qty_hi:.2f}]")
print(f"Unit price IQR bounds: [{price_lo:.2f}, {price_hi:.2f}]")

qty_outliers = ((df["quantity"] < qty_lo) | (df["quantity"] > qty_hi)).sum()
price_outliers = ((df["unit_price"] < price_lo) | (df["unit_price"] > price_hi)).sum()
print(f"Quantity outliers capped: {qty_outliers}")
print(f"Unit price outliers capped: {price_outliers}")

df["quantity"] = df["quantity"].clip(lower=max(qty_lo, 1), upper=qty_hi)
df["unit_price"] = df["unit_price"].clip(lower=max(price_lo, 1), upper=price_hi)
report["outliers_capped"] = {"quantity": int(qty_outliers), "unit_price": int(price_outliers)}

# =====================================================================
# STEP 7: FEATURE ENGINEERING
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: FEATURE ENGINEERING")
print("=" * 70)

df["gross_amount"] = df["quantity"] * df["unit_price"]
df["net_amount"] = (df["gross_amount"] * (1 - df["discount_percent"] / 100)).round(2)
df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
print("Added columns: gross_amount, net_amount, order_month")

# =====================================================================
# STEP 8: SAVE CLEANED DATA
# =====================================================================
out_path = BASE_DIR / "cleaned_ecommerce_sales.csv"
df.to_csv(out_path, index=False)
print(f"\nCleaned dataset saved -> {out_path}")
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
report["final_rows"] = int(df.shape[0])
report["final_cols"] = int(df.shape[1])

# =====================================================================
# STEP 9: AGGREGATE INSIGHTS FOR DASHBOARD
# =====================================================================
print("\n" + "=" * 70)
print("STEP 9: AGGREGATE KEY INSIGHTS")
print("=" * 70)

revenue_by_category = df.groupby("product_category")["net_amount"].sum().sort_values(ascending=False).round(2)
monthly_revenue = df.groupby("order_month")["net_amount"].sum().sort_index().round(2)
top_cities = df.groupby("city")["net_amount"].sum().sort_values(ascending=False).head(8).round(2)
status_counts = df["order_status"].value_counts()
payment_counts = df["payment_method"].value_counts()
region_revenue = df.groupby("region")["net_amount"].sum().sort_values(ascending=False).round(2)

kpis = {
    "total_orders": int(len(df)),
    "total_revenue": round(float(df["net_amount"].sum()), 2),
    "avg_order_value": round(float(df["net_amount"].mean()), 2),
    "delivered_rate": round(float((df["order_status"] == "Delivered").mean() * 100), 1),
    "cancelled_returned_rate": round(float(df["order_status"].isin(["Cancelled", "Returned"]).mean() * 100), 1),
}

dashboard_data = {
    "kpis": kpis,
    "cleaning_report": report,
    "revenue_by_category": {"labels": revenue_by_category.index.tolist(), "values": revenue_by_category.values.tolist()},
    "monthly_revenue": {"labels": monthly_revenue.index.tolist(), "values": monthly_revenue.values.tolist()},
    "top_cities": {"labels": top_cities.index.tolist(), "values": top_cities.values.tolist()},
    "status_counts": {"labels": status_counts.index.tolist(), "values": status_counts.values.tolist()},
    "payment_counts": {"labels": payment_counts.index.tolist(), "values": payment_counts.values.tolist()},
    "region_revenue": {"labels": region_revenue.index.tolist(), "values": region_revenue.values.tolist()},
}

with open(BASE_DIR / "dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)

print("KPIs:", json.dumps(kpis, indent=2))
print("\nSaved dashboard_data.json for visualization step.")

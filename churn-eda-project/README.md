# Customer Churn: Exploratory Data Analysis

An end-to-end EDA project on a synthetic telecom dataset (2,000 customers). It generates the data, audits and cleans it, computes statistics, finds the drivers of churn, and presents everything in an interactive dashboard and a written report.

## Objective
Find which customer attributes are linked to churn so retention effort can be aimed at the right customers.

## Key findings
| Factor | Finding |
|---|---|
| Contract | Month-to-month churn is 48.0%, two-year is 4.4% |
| Tenure | Churn falls from 51% in year one to 13% after year four |
| Support tickets | 21% with none, about 51% with four or more |
| Segment | Fiber + month-to-month is the riskiest group (59%) |
| Region, age | Negligible effect |

## Project structure
```
churn-eda-project/
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── data/telecom_churn_dataset.csv
├── src/
│   ├── generate_data.py        # builds the synthetic dataset (seed 42)
│   └── eda.py                  # cleaning, statistics, tests, figures
└── reports/
    ├── churn_eda_report.html   # full written report
    └── figures/                # saved charts
```

## Run locally
Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py      # optional: regenerates the CSV
python src/eda.py                # prints stats, saves figures
streamlit run app.py             # opens http://localhost:8501
```

To view the written report, open `reports/churn_eda_report.html` in a browser.

## Methods
- Data quality: missing values, duplicates, outliers. Missing age and total_charges (<5%) are filled with the median.
- Summary statistics: mean, median, std, skewness.
- Visuals: histograms, box plots, bar charts, correlation heatmap.
- Statistical tests: chi-square (contract vs churn) and t-test (tenure of churners vs stayers).

## Limitations
The data is synthetic, so findings reflect how it was generated. Correlation does not imply causation.

## Tech stack
Python, pandas, NumPy, SciPy, Matplotlib, Seaborn, Streamlit.

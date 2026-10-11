# Hospital Readmission Analysis and Prediction (Healthcare)

An end-to-end data science project on 4,000 synthetic patient records: clean the data, explore what drives 30-day hospital readmission, train prediction models, and present the results in an interactive dashboard.

> The data is synthetic (random seed 7), so the findings reflect how it was generated. It is for learning only and not for clinical use.

## Problem
Hospitals are penalised and patients are harmed when people return within 30 days of discharge. Which patients are at highest risk, and can we flag them before they leave?

## Key findings
| Factor | Readmission rate |
|---|---|
| Overall | 13.6% |
| 0 prior admissions vs 4 or more | 7.0% vs 51.4% |
| Age 18-40 vs 71+ | 6.2% vs 21.6% |
| Diabetes: no vs yes | 9.1% vs 28.9% |
| Follow-up scheduled: yes vs no | 10.0% vs 19.1% |
| Uninsured vs insured (all other types) | 21.8% vs about 13% |
| Length of stay 1-3 days vs 11+ | 9.4% vs 21.8% |

**Models** (25% held-out test set, patients flagged at 20% or higher predicted risk):

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | 0.730 | 0.318 | 0.507 | 0.391 |
| Random forest | 0.711 | 0.299 | 0.471 | 0.366 |

Logistic regression performs best. The strongest predictors are prior admissions, whether a follow-up was scheduled, and diabetes.

**Takeaways:** schedule follow-up appointments for every patient before discharge, give extra support to patients with several prior admissions or diabetes, and review uninsured patients' discharge plans.

## Project structure
```
hospital-readmission-project/
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── data/hospital_patients.csv
├── models/                   # trained model and metrics
├── reports/figures/          # saved charts
└── src/
    ├── generate_data.py      # builds the synthetic dataset
    └── analysis.py           # EDA, tests, model training, figures
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python src/generate_data.py        # optional: regenerate the data
python src/analysis.py             # EDA, training, figures (about 20 seconds)
python -m streamlit run app.py     # opens http://localhost:8501
```

## Methods
1. **Data quality:** missing HbA1c (10%) and BMI (4%), duplicate check. Missing values are filled with the median inside the model pipeline, so test data never leaks into training.
2. **EDA:** readmission rate by age, prior admissions, stay length, diabetes, follow-up, insurance and discharge destination. Chi-square and t-tests confirm the differences are not chance (all p < 0.001).
3. **Modelling:** stratified 75/25 split, logistic regression and random forest in scikit-learn pipelines.
4. **Evaluation:** ROC-AUC, precision, recall, F1, confusion matrix and permutation feature importance.

## Limitations
- Synthetic data, so real hospital data would be noisier and the model would likely perform differently.
- Moderate accuracy (ROC-AUC 0.73): the model helps rank patients by risk but should not make decisions alone.
- Correlation is not causation. A scheduled follow-up goes with fewer readmissions, but this data cannot prove it causes them.
- Age and prior admissions overlap, so age shows low importance even though older patients are readmitted more often.

## Tech stack
Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Streamlit.

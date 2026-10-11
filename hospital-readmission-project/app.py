"""Streamlit dashboard. Run with: python -m streamlit run app.py"""
import json
import joblib
import pandas as pd
import streamlit as st
from src import analysis as an

st.set_page_config(page_title="Hospital Readmission Analysis", layout="wide")
st.title("30-day hospital readmission: analysis and risk prediction")
st.caption("Synthetic patient data for learning purposes. Not for clinical use.")


@st.cache_data
def get_data():
    return an.load()


@st.cache_resource
def get_model():
    if not (an.MODELS / "best_model.joblib").exists():
        return None, None
    return joblib.load(an.MODELS / "best_model.joblib"), json.load(open(an.MODELS / "metrics.json"))


df = get_data()
d = an.add_bands(df)
model, meta = get_model()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{len(df):,}")
c2.metric("30-day readmission rate", f"{df[an.TARGET].mean() * 100:.1f}%")
c3.metric("Average age", f"{df.age.mean():.1f}")
c4.metric("Average stay", f"{df.length_of_stay.mean():.1f} days")

t1, t2, t3, t4 = st.tabs(["Data & quality", "Explore drivers", "Model performance", "Risk predictor"])

with t1:
    st.dataframe(df.head(50))
    q, dups = an.quality(df)
    st.subheader("Missing values")
    st.dataframe(q[q.missing > 0])
    st.caption(f"Duplicate rows: {dups}. Missing HbA1c and BMI are filled with the median inside the model pipeline.")
    st.subheader("Summary statistics")
    st.dataframe(df[an.NUM].describe().T.round(2))

with t2:
    groups = ["age_band", "prior_band", "los_band", "diabetes", "followup_scheduled", "discharge_to", "insurance", "smoker"]
    labels = {"age_band": "Age band", "prior_band": "Prior admissions", "los_band": "Length of stay (days)", "diabetes": "Diabetes (1 = yes)",
              "followup_scheduled": "Follow-up scheduled", "discharge_to": "Discharged to", "insurance": "Insurance", "smoker": "Smoker"}
    col = st.selectbox("Group patients by", groups, format_func=lambda g: labels[g])
    r = an.rate_by(d, col)
    st.bar_chart(r["rate"])
    st.dataframe(r.rename(columns={"rate": "readmission rate (%)", "n": "patients"}))
    st.caption(f"Overall rate: {df[an.TARGET].mean() * 100:.1f}%")
    st.image(str(an.FIGS / "correlation_heatmap.png"), caption="Correlation matrix of numeric variables") if (an.FIGS / "correlation_heatmap.png").exists() else None

with t3:
    if meta is None:
        st.info("Run `python src/analysis.py` first to train the model.")
    else:
        st.write(f"Best model by ROC-AUC: **{meta['best']}**. Patients with predicted risk of {meta['threshold']:.0%} or more are flagged.")
        st.dataframe(pd.DataFrame(meta["metrics"]).T.round(3))
        a, b = st.columns(2)
        a.image(str(an.FIGS / "roc_curves.png")); b.image(str(an.FIGS / "confusion_matrix.png"))
        st.image(str(an.FIGS / "feature_importance.png"))

with t4:
    if model is None:
        st.info("Run `python src/analysis.py` first to train the model.")
    else:
        st.subheader("Estimate a patient's readmission risk")
        a, b, c = st.columns(3)
        age = a.slider("Age", 18, 95, 65)
        prior = a.slider("Prior admissions", 0, 8, 1)
        los = a.slider("Length of stay (days)", 1, 30, 5)
        meds = b.slider("Number of medications", 1, 40, 10)
        diabetes = b.selectbox("Diabetes", ["No", "Yes"]) == "Yes"
        hyper = b.selectbox("Hypertension", ["No", "Yes"]) == "Yes"
        followup = c.selectbox("Follow-up scheduled", ["Yes", "No"])
        insurance = c.selectbox("Insurance", ["Private", "Medicare", "Medicaid", "Uninsured"])
        discharge = c.selectbox("Discharged to", ["Home", "Rehab", "Nursing facility"])
        smoker = c.selectbox("Smoker", ["No", "Yes"])
        row = pd.DataFrame([dict(age=age, bmi=28.5, prior_admissions=prior, length_of_stay=los, num_medications=meds,
                                 num_lab_procedures=45, hba1c=7.5 if diabetes else 5.5, systolic_bp=140 if hyper else 120,
                                 diabetes=int(diabetes), hypertension=int(hyper), gender="Female", smoker=smoker,
                                 insurance=insurance, discharge_to=discharge, followup_scheduled=followup)])
        p = float(model.predict_proba(row[an.NUM + an.CAT])[0, 1])
        tier = "High" if p >= 0.25 else "Moderate" if p >= 0.10 else "Low"
        st.metric("Predicted 30-day readmission risk", f"{p * 100:.1f}%")
        (st.error if tier == "High" else st.warning if tier == "Moderate" else st.success)(f"{tier} risk")
        st.caption("BMI, lab procedures and gender are set to typical values here. Educational demo on synthetic data.")

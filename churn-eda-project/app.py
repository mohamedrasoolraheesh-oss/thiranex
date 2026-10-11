"""Streamlit dashboard. Run with: streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from src import eda

st.set_page_config(page_title="Customer Churn EDA", layout="wide")
st.title("Customer churn: exploratory data analysis")


@st.cache_data
def get_data():
    return eda.load()


raw = get_data()
df = eda.clean(raw)

with st.sidebar:
    st.header("Filters")
    contract = st.multiselect("Contract", sorted(df.contract.unique()), default=sorted(df.contract.unique()))
    internet = st.multiselect("Internet service", sorted(df.internet_service.unique()), default=sorted(df.internet_service.unique()))
    lo, hi = st.slider("Tenure (months)", 1, 72, (1, 72))
view = df[df.contract.isin(contract) & df.internet_service.isin(internet) & df.tenure_months.between(lo, hi)]
if view.empty:
    st.warning("No customers match these filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(view):,}")
c2.metric("Churn rate", f"{view.churn.mean() * 100:.1f}%")
c3.metric("Avg tenure", f"{view.tenure_months.mean():.1f} mo")
c4.metric("Avg monthly bill", f"${view.monthly_charges.mean():.2f}")

t1, t2, t3, t4, t5 = st.tabs(["Data & quality", "Distributions", "Churn drivers", "Correlations", "Insights"])

with t1:
    st.dataframe(view.head(50))
    q, dups = eda.quality(raw)
    st.subheader("Missing values (raw data)")
    st.dataframe(q[q.missing > 0])
    st.caption(f"Duplicate rows: {dups}. Missing age and total_charges were filled with the median.")
    st.subheader("Summary statistics")
    st.dataframe(eda.summary(view))

with t2:
    col = st.selectbox("Numeric column", eda.NUMS, index=1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.histplot(data=view, x=col, hue="churn", kde=True, ax=ax, multiple="stack")
    st.pyplot(fig)

with t3:
    avg = view.churn.mean() * 100
    cols = st.columns(2)
    for i, c in enumerate(eda.CATS):
        with cols[i % 2]:
            st.subheader(f"By {c}")
            st.bar_chart(eda.churn_by(view, c)["rate"])
    cols = st.columns(2)
    cols[0].subheader("By tenure band"); cols[0].bar_chart(eda.tenure_bands(view)["rate"])
    cols[1].subheader("By support tickets"); cols[1].bar_chart(eda.ticket_bands(view)["rate"])
    st.subheader("Contract × internet service (churn %)")
    st.dataframe(eda.two_way(view).style.background_gradient(cmap="Purples"))
    st.caption(f"Average churn in current selection: {avg:.1f}%")

with t4:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(eda.correlations(view), annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    t = eda.tests(df)
    st.write(f"Chi-square, contract vs churn: p = {t['chi2_contract_vs_churn_p']:.2e}")
    st.write(f"t-test, tenure of churners vs stayers: p = {t['ttest_tenure_p']:.2e}")

with t5:
    st.markdown("""
- **Contract type** is the strongest driver: month-to-month customers churn about 11x more than two-year customers.
- **Tenure**: churn is highest in the first 12 months and falls steadily after that.
- **Support tickets**: more tickets go with higher churn.
- **Fiber + month-to-month** is the riskiest segment (about 59%).
- **Region and age** have almost no effect.
- **Caveat**: the data is synthetic and correlation is not causation.
""")

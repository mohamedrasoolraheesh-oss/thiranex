import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_ecommerce_sales.csv"
JSON_PATH = BASE_DIR / "dashboard_data.json"

st.set_page_config(
    page_title="E-Commerce Sales Analytics",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    return df


@st.cache_data
def load_dashboard_data():
    if JSON_PATH.exists():
        with open(JSON_PATH, "r") as file:
            return json.load(file)

    return {}


try:
    df = load_data()
    dashboard_data = load_dashboard_data()

except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()


# ------------------------------------------------------------
# CALCULATED COLUMNS
# ------------------------------------------------------------

if "net_amount" not in df.columns:
    df["gross_amount"] = df["quantity"] * df["unit_price"]

    df["net_amount"] = (
        df["gross_amount"]
        * (1 - df["discount_percent"].fillna(0) / 100)
    )

df["order_month"] = df["order_date"].dt.to_period("M").astype(str)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("📊 E-Commerce Sales Analytics Dashboard")

st.markdown(
    "Interactive analytics dashboard based on the cleaned e-commerce sales dataset."
)

st.divider()


# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------

st.sidebar.header("🔎 Filters")

regions = sorted(df["region"].dropna().unique().tolist())

selected_regions = st.sidebar.multiselect(
    "Region",
    regions,
    default=regions,
)

categories = sorted(
    df["product_category"].dropna().unique().tolist()
)

selected_categories = st.sidebar.multiselect(
    "Product Category",
    categories,
    default=categories,
)

statuses = sorted(
    df["order_status"].dropna().unique().tolist()
)

selected_statuses = st.sidebar.multiselect(
    "Order Status",
    statuses,
    default=statuses,
)

payment_methods = sorted(
    df["payment_method"].dropna().unique().tolist()
)

selected_payment_methods = st.sidebar.multiselect(
    "Payment Method",
    payment_methods,
    default=payment_methods,
)


# ------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------

filtered_df = df[
    df["region"].isin(selected_regions)
    & df["product_category"].isin(selected_categories)
    & df["order_status"].isin(selected_statuses)
    & df["payment_method"].isin(selected_payment_methods)
].copy()


# ------------------------------------------------------------
# KPI CALCULATIONS
# ------------------------------------------------------------

total_orders = len(filtered_df)

total_revenue = filtered_df["net_amount"].sum()

avg_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)

delivered_count = (
    filtered_df["order_status"]
    .astype(str)
    .str.lower()
    .eq("delivered")
    .sum()
)

delivered_rate = (
    delivered_count / total_orders * 100
    if total_orders > 0
    else 0
)

cancelled_returned_count = (
    filtered_df["order_status"]
    .astype(str)
    .str.lower()
    .isin(["cancelled", "returned"])
    .sum()
)

cancelled_returned_rate = (
    cancelled_returned_count / total_orders * 100
    if total_orders > 0
    else 0
)


# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Orders",
    f"{total_orders:,}"
)

col2.metric(
    "Total Revenue",
    f"₹{total_revenue:,.2f}"
)

col3.metric(
    "Average Order Value",
    f"₹{avg_order_value:,.2f}"
)

col4.metric(
    "Delivered Rate",
    f"{delivered_rate:.1f}%"
)

col5.metric(
    "Cancelled / Returned",
    f"{cancelled_returned_rate:.1f}%"
)


st.divider()


# ------------------------------------------------------------
# MONTHLY REVENUE
# ------------------------------------------------------------

st.subheader("📈 Monthly Revenue Trend")

monthly = (
    filtered_df
    .groupby("order_month", as_index=False)["net_amount"]
    .sum()
    .sort_values("order_month")
)

if not monthly.empty:
    fig_monthly = px.line(
        monthly,
        x="order_month",
        y="net_amount",
        markers=True,
        title="Monthly Net Revenue",
        labels={
            "order_month": "Month",
            "net_amount": "Revenue",
        },
    )

    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True,
    )
else:
    st.info("No data available for the selected filters.")


# ------------------------------------------------------------
# CATEGORY + REGION
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("🛍️ Revenue by Product Category")

    category_data = (
        filtered_df
        .groupby("product_category", as_index=False)["net_amount"]
        .sum()
        .sort_values("net_amount", ascending=False)
    )

    if not category_data.empty:
        fig_category = px.bar(
            category_data,
            x="product_category",
            y="net_amount",
            title="Revenue by Category",
            labels={
                "product_category": "Category",
                "net_amount": "Revenue",
            },
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True,
        )

with col2:
    st.subheader("🌍 Revenue by Region")

    region_data = (
        filtered_df
        .groupby("region", as_index=False)["net_amount"]
        .sum()
        .sort_values("net_amount", ascending=False)
    )

    if not region_data.empty:
        fig_region = px.bar(
            region_data,
            x="region",
            y="net_amount",
            title="Revenue by Region",
            labels={
                "region": "Region",
                "net_amount": "Revenue",
            },
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True,
        )


# ------------------------------------------------------------
# ORDER STATUS + PAYMENT METHOD
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Order Status Distribution")

    status_data = (
        filtered_df["order_status"]
        .value_counts()
        .reset_index()
    )

    status_data.columns = [
        "order_status",
        "count",
    ]

    if not status_data.empty:
        fig_status = px.pie(
            status_data,
            names="order_status",
            values="count",
            title="Orders by Status",
            hole=0.4,
        )

        st.plotly_chart(
            fig_status,
            use_container_width=True,
        )

with col2:
    st.subheader("💳 Payment Methods")

    payment_data = (
        filtered_df["payment_method"]
        .value_counts()
        .reset_index()
    )

    payment_data.columns = [
        "payment_method",
        "count",
    ]

    if not payment_data.empty:
        fig_payment = px.pie(
            payment_data,
            names="payment_method",
            values="count",
            title="Orders by Payment Method",
            hole=0.4,
        )

        st.plotly_chart(
            fig_payment,
            use_container_width=True,
        )


# ------------------------------------------------------------
# TOP PRODUCTS
# ------------------------------------------------------------

st.subheader("🏆 Top Products by Revenue")

product_data = (
    filtered_df
    .groupby("product_name", as_index=False)
    .agg(
        Revenue=("net_amount", "sum"),
        Orders=("order_id", "count"),
        Quantity=("quantity", "sum"),
    )
    .sort_values("Revenue", ascending=False)
    .head(10)
)

if not product_data.empty:
    fig_products = px.bar(
        product_data.sort_values("Revenue"),
        x="Revenue",
        y="product_name",
        orientation="h",
        title="Top 10 Products",
        labels={
            "Revenue": "Revenue",
            "product_name": "Product",
        },
    )

    st.plotly_chart(
        fig_products,
        use_container_width=True,
    )


# ------------------------------------------------------------
# CITY ANALYSIS
# ------------------------------------------------------------

st.subheader("🏙️ Revenue by City")

city_data = (
    filtered_df
    .groupby("city", as_index=False)["net_amount"]
    .sum()
    .sort_values("net_amount", ascending=False)
    .head(10)
)

if not city_data.empty:
    fig_city = px.bar(
        city_data.sort_values("net_amount"),
        x="net_amount",
        y="city",
        orientation="h",
        title="Top 10 Cities by Revenue",
        labels={
            "net_amount": "Revenue",
            "city": "City",
        },
    )

    st.plotly_chart(
        fig_city,
        use_container_width=True,
    )


# ------------------------------------------------------------
# DATA TABLE
# ------------------------------------------------------------

st.subheader("📋 Filtered Sales Data")

display_columns = [
    "order_id",
    "order_date",
    "customer_name",
    "product_category",
    "product_name",
    "quantity",
    "unit_price",
    "discount_percent",
    "payment_method",
    "city",
    "region",
    "order_status",
    "gross_amount",
    "net_amount",
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns],
    use_container_width=True,
    hide_index=True,
)


# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------

st.subheader("⬇️ Download Cleaned Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered CSV",
    data=csv_data,
    file_name="filtered_ecommerce_sales.csv",
    mime="text/csv",
)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    f"Dashboard showing {len(filtered_df):,} records from the cleaned dataset."
)

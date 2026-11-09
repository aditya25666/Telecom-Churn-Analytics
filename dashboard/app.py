
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Telecom Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "processed"

# =====================================================
# 2. DARK THEME
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b1220;
    color: #e5edf8;
}

[data-testid="stHeader"] {
    background: rgba(11,18,32,.94);
}

[data-testid="stSidebar"] {
    background: #101a2d;
    border-right: 1px solid #24334d;
}

[data-testid="stSidebar"] * {
    color: #e8eef8 !important;
}

[data-testid="stSidebar"] p {
    color: #aebbd0 !important;
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #f1f5fb !important;
}

h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 1.35rem !important;
}

p, label, .stMarkdown {
    color: #c3cede;
}

[data-testid="stMetric"] {
    background: #141f33;
    border: 1px solid #263751;
    padding: 17px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"] {
    color: #a8b6ca !important;
}

[data-testid="stMetricValue"] {
    color: #f4f7fc !important;
    font-weight: 800 !important;
}

.hero {
    background: linear-gradient(
        115deg, #111c32 0%, #17365c 58%, #1e5790 100%
    );
    border: 1px solid #29486e;
    border-radius: 18px;
    padding: 25px 28px;
    margin-bottom: 20px;
}

.hero h1 {
    color: white !important;
    margin-bottom: 5px;
}

.hero p {
    color: #d0def0;
}

.eyebrow {
    color: #9fc4ed;
    font-size: .72rem;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.section-note {
    color: #9aaac0;
    font-size: .88rem;
    margin-bottom: 14px;
}

.insight {
    background: #14243a;
    border: 1px solid #29415f;
    border-left: 4px solid #60a5fa;
    border-radius: 10px;
    padding: 13px 16px;
    color: #d5e2f2;
    margin: 12px 0;
}

.small-muted {
    color: #91a1b8;
    font-size: .8rem;
}

hr {
    border-color: #263751 !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATASETS
# =====================================================

@st.cache_data(show_spinner="Loading telecom datasets...")
def load_data():

    names = [
        "customers",
        "subscriptions",
        "usage",
        "recharge",
        "complaints",
        "churn"
    ]

    datasets = {}

    for name in names:

        path = DATA_DIR / name / "data.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed dataset: {path}"
            )

        datasets[name] = pd.read_parquet(
            path,
            engine="pyarrow"
        )

    return datasets


try:
    data = load_data()

except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.info("Run Phase 4 data cleaning first.")
    st.stop()


customers = data["customers"].copy()
subscriptions = data["subscriptions"].copy()
usage = data["usage"].copy()
recharge = data["recharge"].copy()
complaints = data["complaints"].copy()
churn = data["churn"].copy()


# =====================================================
# 4. DATA PREPARATION
# =====================================================

def normalize_status(value):

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    if value in ["churned", "yes", "1", "true"]:
        return "Churned"

    if value in ["retained", "no", "0", "false", "active"]:
        return "Retained"

    return "Unknown"


churn["churn_status"] = (
    churn["churn_status"].apply(normalize_status)
)


for frame, column in [
    (customers, "joining_date"),
    (usage, "usage_date"),
    (recharge, "recharge_date"),
    (complaints, "complaint_date"),
    (churn, "churn_date")
]:

    if column in frame.columns:

        frame[column] = pd.to_datetime(
            frame[column],
            errors="coerce"
        )


# =====================================================
# 5. CUSTOMER AGGREGATIONS
# =====================================================

usage_metrics = usage.groupby("customer_id").agg(
    total_data_usage_gb=("data_usage_gb", "sum"),
    avg_data_usage_gb=("data_usage_gb", "mean"),
    total_call_minutes=("call_minutes", "sum"),
    total_sms=("sms_count", "sum")
).reset_index()


recharge_metrics = recharge.groupby("customer_id").agg(
    recharge_count=("customer_id", "size"),
    total_recharge_amount=("recharge_amount", "sum"),
    avg_recharge_amount=("recharge_amount", "mean")
).reset_index()


complaint_metrics = complaints.groupby("customer_id").agg(
    complaint_count=("customer_id", "size")
).reset_index()


customer = (
    customers
    .merge(subscriptions, on="customer_id", how="left")
    .merge(
        churn[["customer_id", "churn_status", "churn_date"]],
        on="customer_id",
        how="left"
    )
    .merge(usage_metrics, on="customer_id", how="left")
    .merge(recharge_metrics, on="customer_id", how="left")
    .merge(complaint_metrics, on="customer_id", how="left")
)


customer["churn_status"] = (
    customer["churn_status"].fillna("Unknown")
)


numeric_columns = [
    "total_data_usage_gb",
    "avg_data_usage_gb",
    "total_call_minutes",
    "total_sms",
    "recharge_count",
    "total_recharge_amount",
    "avg_recharge_amount",
    "complaint_count"
]

for column in numeric_columns:
    customer[column] = pd.to_numeric(
        customer[column], errors="coerce"
    ).fillna(0)


# =====================================================
# 6. CUSTOMER SEGMENTATION
# =====================================================

def classify_customer(row):

    if row["churn_status"] == "Churned":
        return "Historically Churned"

    if row["complaint_count"] >= 2:
        return "Frequent Complaints"

    if row["recharge_count"] == 0:
        return "No Recorded Recharge"

    if row["recharge_count"] >= 8:
        return "Frequent Recharge"

    return "Regular Customer"


customer["attention_category"] = customer.apply(
    classify_customer,
    axis=1
)


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("## 📡 Telecom IQ")
st.sidebar.caption("Customer Churn & Business Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Customer Analytics",
        "Usage Analytics",
        "Recharge Analytics",
        "Churn Analytics",
        "Complaint Analytics"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Global Filters")


cities = sorted(
    customers["city"].dropna().astype(str).unique()
)

plans = sorted(
    subscriptions["plan_type"].dropna().astype(str).unique()
)

city_filter = st.sidebar.selectbox(
    "City",
    ["All cities"] + cities
)

plan_filter = st.sidebar.selectbox(
    "Subscription Plan",
    ["All plans"] + plans
)

status_filter = st.sidebar.selectbox(
    "Customer Status",
    ["All statuses", "Churned", "Retained", "Unknown"]
)


# Date filter

all_dates = pd.concat([
    usage["usage_date"],
    recharge["recharge_date"],
    complaints["complaint_date"],
    churn["churn_date"]
]).dropna()


if not all_dates.empty:

    min_date = all_dates.min().date()
    max_date = all_dates.max().date()

    date_range = st.sidebar.date_input(
        "Event Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

else:
    start_date = end_date = None


# =====================================================
# 8. APPLY GLOBAL FILTERS
# =====================================================

filtered_customer = customer.copy()

if city_filter != "All cities":

    filtered_customer = filtered_customer[
        filtered_customer["city"].astype(str) == city_filter
    ]


if plan_filter != "All plans":

    filtered_customer = filtered_customer[
        filtered_customer["plan_type"].astype(str) == plan_filter
    ]


if status_filter != "All statuses":

    filtered_customer = filtered_customer[
        filtered_customer["churn_status"] == status_filter
    ]


selected_ids = set(
    filtered_customer["customer_id"]
)


def filter_events(df, date_column):

    result = df[
        df["customer_id"].isin(selected_ids)
    ].copy()

    if start_date is not None:

        result = result[
            result[date_column].dt.date.between(
                start_date, end_date
            )
        ]

    return result


f_usage = filter_events(usage, "usage_date")
f_recharge = filter_events(recharge, "recharge_date")
f_complaints = filter_events(complaints, "complaint_date")
f_churn = filter_events(churn, "churn_date")


# =====================================================
# 9. CHART HELPER FUNCTIONS
# =====================================================

def polish(fig, height=340):

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="#141f33",
        plot_bgcolor="#141f33",
        font=dict(
            family="DM Sans",
            color="#d5dfed",
            size=12
        ),
        title=dict(
            font=dict(size=15, color="#f1f5fb")
        ),
        legend=dict(
            orientation="h",
            y=1.02,
            x=1,
            xanchor="right"
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#34445d"
        ),
        yaxis=dict(
            gridcolor="#293950",
            zeroline=False
        )
    )

    return fig


def heading(title, description=None):

    st.subheader(title)

    if description:
        st.markdown(
            f'<div class="section-note">{description}</div>',
            unsafe_allow_html=True
        )


def metric_row(items):

    columns = st.columns(len(items))

    for col, (label, value, help_text) in zip(columns, items):

        col.metric(
            label,
            value,
            help=help_text
        )


def money(value):
    return f"₹{value:,.2f}"


def no_data():
    st.info("No data available for the selected filters.")



# =====================================================
# DASHBOARD HEADER
# =====================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">BIG DATA ANALYTICS • TELECOMMUNICATIONS</div>
        <h1>Telecom Customer Intelligence</h1>
        <p>Explore customer activity, recharge behavior, service experience and historical churn patterns.</p>
    </div>
    """.replace("\n", ""),
    unsafe_allow_html=True
)

st.caption(
    f"Current Section: {page} | "
    f"Customers: {len(filtered_customer):,}"
)


# =====================================================
# 11. EXECUTIVE OVERVIEW
# =====================================================

if page == "Executive Overview":

    recognized = filtered_customer[
        filtered_customer["churn_status"].isin(
            ["Churned", "Retained"]
        )
    ]

    churned = (
        recognized["churn_status"] == "Churned"
    ).sum()

    retained = (
        recognized["churn_status"] == "Retained"
    ).sum()

    churn_rate = (
        churned / len(recognized) * 100
        if len(recognized) else 0
    )

    metric_row([
        ("Total Customers", f"{len(filtered_customer):,}", "Filtered customers"),
        ("Churned Customers", f"{churned:,}", "Historical churn"),
        ("Churn Rate", f"{churn_rate:.2f}%", "Unknown statuses excluded"),
        ("Recharge Revenue", money(f_recharge["recharge_amount"].sum()), "Transaction revenue"),
        ("Complaints", f"{len(f_complaints):,}", "Selected event period")
    ])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        heading("Customer Retention Distribution")

        d = recognized["churn_status"].value_counts().reset_index()
        d.columns = ["Status", "Customers"]

        if not d.empty:

            fig = px.pie(
                d,
                names="Status",
                values="Customers",
                hole=0.6,
                color="Status",
                color_discrete_map={
                    "Churned": "#e76f51",
                    "Retained": "#2a9d8f"
                }
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            st.plotly_chart(
                polish(fig),
                use_container_width=True
            )

        else:
            no_data()

    with col2:

        heading("Top 10 Cities by Customers")

        d = (
            filtered_customer["city"]
            .value_counts()
            .head(10)
            .sort_values()
            .reset_index()
        )

        d.columns = ["City", "Customers"]

        if not d.empty:

            fig = px.bar(
                d,
                x="Customers",
                y="City",
                orientation="h",
                text="Customers"
            )

            fig.update_traces(
                marker_color="#4776a8",
                textposition="outside"
            )

            st.plotly_chart(
                polish(fig),
                use_container_width=True
            )

    col1, col2 = st.columns(2)

    with col1:

        heading("Monthly Recharge Revenue")

        if not f_recharge.empty:

            tmp = f_recharge.copy()
            tmp["Month"] = tmp["recharge_date"].dt.to_period("M").astype(str)

            d = tmp.groupby("Month", as_index=False)[
                "recharge_amount"
            ].sum()

            fig = px.line(
                d,
                x="Month",
                y="recharge_amount",
                markers=True
            )

            fig.update_yaxes(title="Revenue (₹)")

            st.plotly_chart(
                polish(fig),
                use_container_width=True
            )

        else:
            no_data()

    with col2:

        heading("Monthly Data Consumption")

        if not f_usage.empty:

            tmp = f_usage.copy()
            tmp["Month"] = tmp["usage_date"].dt.to_period("M").astype(str)

            d = tmp.groupby("Month", as_index=False)[
                "data_usage_gb"
            ].sum()

            fig = px.area(
                d,
                x="Month",
                y="data_usage_gb"
            )

            fig.update_yaxes(title="Data Usage (GB)")

            st.plotly_chart(
                polish(fig),
                use_container_width=True
            )

        else:
            no_data()

    st.markdown(
        f"""
        <div class="insight">
        <b>Business Overview:</b>
        {retained:,} customers are historically retained and
        {churned:,} are historically churned in the selected population.
        These are historical descriptive results, not future predictions.
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 12. CUSTOMER ANALYTICS
# =====================================================

elif page == "Customer Analytics":

    heading(
        "Customer Analytics",
        "Explore demographics, subscription plans and customer segments."
    )

    metric_row([
        ("Customers", f"{len(filtered_customer):,}", "Filtered customers"),
        ("Average Monthly Charge",
         money(filtered_customer["monthly_charges"].mean()),
         "Average subscription charge"),
        ("Average Complaints",
         f"{filtered_customer['complaint_count'].mean():.2f}",
         "Complaints per customer"),
        ("No Recorded Recharge",
         f"{(filtered_customer['recharge_count'] == 0).sum():,}",
         "No recharge records found")
    ])

    col1, col2 = st.columns(2)

    with col1:

        heading("Subscription Plan Distribution")

        d = filtered_customer["plan_type"].value_counts().reset_index()
        d.columns = ["Plan", "Customers"]

        fig = px.bar(
            d,
            x="Plan",
            y="Customers",
            color="Plan"
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(polish(fig), use_container_width=True)

    with col2:

        heading("Customer Attention Categories")

        d = filtered_customer[
            "attention_category"
        ].value_counts().reset_index()

        d.columns = ["Category", "Customers"]

        fig = px.bar(
            d,
            x="Customers",
            y="Category",
            orientation="h",
            color="Category"
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(polish(fig), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        if "gender" in filtered_customer:

            heading("Gender Distribution")

            d = filtered_customer["gender"].value_counts().reset_index()
            d.columns = ["Gender", "Customers"]

            fig = px.pie(
                d,
                names="Gender",
                values="Customers",
                hole=.55
            )

            st.plotly_chart(polish(fig), use_container_width=True)

    with col2:

        if "contract_type" in filtered_customer:

            heading("Contract Type Distribution")

            d = filtered_customer["contract_type"].value_counts().reset_index()
            d.columns = ["Contract", "Customers"]

            fig = px.pie(
                d,
                names="Contract",
                values="Customers",
                hole=.55
            )

            st.plotly_chart(polish(fig), use_container_width=True)

    heading("Customer Records")

    columns = [
        c for c in [
            "customer_id", "city", "gender",
            "plan_type", "contract_type",
            "monthly_charges", "churn_status",
            "total_data_usage_gb", "recharge_count",
            "complaint_count", "attention_category"
        ]
        if c in filtered_customer.columns
    ]

    st.dataframe(
        filtered_customer[columns],
        use_container_width=True,
        hide_index=True,
        height=420
    )

    st.download_button(
        "Download Customer Records",
        filtered_customer[columns].to_csv(index=False),
        "customer_records.csv",
        "text/csv"
    )


# =====================================================
# 13. USAGE ANALYTICS
# =====================================================

elif page == "Usage Analytics":

    heading(
        "Usage Analytics",
        "Analyze data, voice and SMS consumption trends."
    )

    if f_usage.empty:

        no_data()

    else:

        metric_row([
            ("Data Consumed", f"{f_usage['data_usage_gb'].sum():,.2f} GB", "Total data"),
            ("Call Minutes", f"{f_usage['call_minutes'].sum():,.0f}", "Total calls"),
            ("SMS", f"{f_usage['sms_count'].sum():,.0f}", "Total SMS"),
            ("Active Customers", f"{f_usage['customer_id'].nunique():,}", "Unique usage customers")
        ])

        tmp = f_usage.copy()

        tmp["Month"] = (
            tmp["usage_date"].dt.to_period("M").astype(str)
        )

        monthly = tmp.groupby("Month", as_index=False).agg(
            Data_GB=("data_usage_gb", "sum"),
            Call_Minutes=("call_minutes", "sum"),
            SMS=("sms_count", "sum")
        )

        for column, title, unit in [
            ("Data_GB", "Monthly Data Consumption", "GB"),
            ("Call_Minutes", "Monthly Voice Usage", "Minutes"),
            ("SMS", "Monthly SMS Usage", "Messages")
        ]:

            heading(title)

            fig = px.line(
                monthly,
                x="Month",
                y=column,
                markers=True
            )

            fig.update_yaxes(title=unit)

            st.plotly_chart(
                polish(fig, 300),
                use_container_width=True
            )

        heading("Usage Records")

        st.dataframe(
            tmp.sort_values("usage_date", ascending=False),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 14. RECHARGE ANALYTICS
# =====================================================

elif page == "Recharge Analytics":

    heading(
        "Recharge Analytics",
        "Analyze recharge transactions and payment behavior."
    )

    if f_recharge.empty:

        no_data()

    else:

        metric_row([
            ("Recharge Revenue", money(f_recharge["recharge_amount"].sum()), "Total recharge amount"),
            ("Transactions", f"{len(f_recharge):,}", "Recharge records"),
            ("Unique Rechargers", f"{f_recharge['customer_id'].nunique():,}", "Unique customers"),
            ("Average Recharge", money(f_recharge["recharge_amount"].mean()), "Average transaction")
        ])

        tmp = f_recharge.copy()

        tmp["Month"] = (
            tmp["recharge_date"].dt.to_period("M").astype(str)
        )

        monthly = tmp.groupby("Month", as_index=False).agg(
            Revenue=("recharge_amount", "sum"),
            Transactions=("customer_id", "count")
        )

        col1, col2 = st.columns(2)

        with col1:

            heading("Monthly Recharge Revenue")

            fig = px.bar(
                monthly,
                x="Month",
                y="Revenue"
            )

            fig.update_yaxes(title="Revenue (₹)")

            st.plotly_chart(polish(fig), use_container_width=True)

        with col2:

            heading("Monthly Recharge Transactions")

            fig = px.line(
                monthly,
                x="Month",
                y="Transactions",
                markers=True
            )

            st.plotly_chart(polish(fig), use_container_width=True)

        if "payment_method" in tmp.columns:

            col1, col2 = st.columns(2)

            with col1:

                heading("Payment Method Distribution")

                d = tmp["payment_method"].value_counts().reset_index()
                d.columns = ["Method", "Transactions"]

                fig = px.pie(
                    d,
                    names="Method",
                    values="Transactions",
                    hole=.55
                )

                st.plotly_chart(polish(fig), use_container_width=True)

            with col2:

                heading("Revenue by Payment Method")

                d = tmp.groupby(
                    "payment_method",
                    as_index=False
                )["recharge_amount"].sum()

                fig = px.bar(
                    d,
                    x="payment_method",
                    y="recharge_amount"
                )

                fig.update_yaxes(title="Revenue (₹)")

                st.plotly_chart(polish(fig), use_container_width=True)

        heading("Recharge Transactions")

        st.dataframe(
            tmp.sort_values("recharge_date", ascending=False),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 15. CHURN ANALYTICS
# =====================================================

elif page == "Churn Analytics":

    heading(
        "Historical Churn Analytics",
        "Analyze recorded churn patterns. No ML prediction is performed."
    )

    recognized = filtered_customer[
        filtered_customer["churn_status"].isin(
            ["Churned", "Retained"]
        )
    ]

    churned = (
        recognized["churn_status"] == "Churned"
    ).sum()

    retained = (
        recognized["churn_status"] == "Retained"
    ).sum()

    churn_rate = (
        churned / len(recognized) * 100
        if len(recognized) else 0
    )

    metric_row([
        ("Churned Customers", f"{churned:,}", "Historical churn"),
        ("Retained Customers", f"{retained:,}", "Historical retention"),
        ("Churn Rate", f"{churn_rate:.2f}%", "Unknown statuses excluded"),
        ("Unknown Status", f"{(filtered_customer['churn_status'] == 'Unknown').sum():,}", "Unrecognized labels")
    ])

    col1, col2 = st.columns(2)

    with col1:

        heading("Churn by Subscription Plan")

        d = recognized.groupby(
            ["plan_type", "churn_status"]
        ).size().reset_index(name="Customers")

        if not d.empty:

            fig = px.bar(
                d,
                x="plan_type",
                y="Customers",
                color="churn_status",
                barmode="group",
                color_discrete_map={
                    "Churned": "#e76f51",
                    "Retained": "#2a9d8f"
                }
            )

            st.plotly_chart(polish(fig), use_container_width=True)

        else:
            no_data()

    with col2:

        heading("Monthly Historical Churn")

        events = f_churn[
            (f_churn["churn_status"] == "Churned")
            & f_churn["churn_date"].notna()
        ].copy()

        if not events.empty:

            events["Month"] = (
                events["churn_date"].dt.to_period("M").astype(str)
            )

            d = events.groupby("Month")[
                "customer_id"
            ].nunique().reset_index(name="Churned Customers")

            fig = px.line(
                d,
                x="Month",
                y="Churned Customers",
                markers=True
            )

            st.plotly_chart(polish(fig), use_container_width=True)

        else:
            no_data()

    heading(
        "Churned vs Retained Customer Behavior",
        "Descriptive averages; these differences do not establish causation."
    )

    comparison = recognized.groupby("churn_status").agg(
        Customers=("customer_id", "nunique"),
        Average_Data_Usage=("total_data_usage_gb", "mean"),
        Average_Recharges=("recharge_count", "mean"),
        Average_Recharge_Amount=("total_recharge_amount", "mean"),
        Average_Complaints=("complaint_count", "mean")
    ).reset_index()

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="insight">
    <b>Interpretation:</b>
    Compare the historical averages of churned and retained customers.
    These are descriptive associations, not proof of causation or
    predictions of future customer behavior.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 16. COMPLAINT ANALYTICS
# =====================================================

elif page == "Complaint Analytics":

    heading(
        "Complaint Analytics",
        "Analyze complaint categories and resolution performance."
    )

    if f_complaints.empty:

        no_data()

    else:

        resolved_mask = (
            f_complaints["resolution_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["resolved", "closed"])
        )

        resolved = f_complaints[resolved_mask]

        avg_days = pd.to_numeric(
            resolved["resolution_days"],
            errors="coerce"
        ).mean()

        metric_row([
            ("Total Complaints", f"{len(f_complaints):,}", "Complaint records"),
            ("Resolved / Closed", f"{len(resolved):,}", "Resolved complaints"),
            ("Other Statuses", f"{len(f_complaints)-len(resolved):,}", "Non-resolved or other labels"),
            ("Average Resolution", f"{avg_days:.2f} days" if pd.notna(avg_days) else "N/A", "Resolved complaint average")
        ])

        col1, col2 = st.columns(2)

        with col1:

            heading("Complaints by Category")

            d = (
                f_complaints["complaint_type"]
                .value_counts()
                .reset_index()
            )

            d.columns = ["Category", "Complaints"]

            fig = px.bar(
                d,
                x="Category",
                y="Complaints"
            )

            st.plotly_chart(polish(fig), use_container_width=True)

        with col2:

            heading("Complaint Resolution Status")

            d = (
                f_complaints["resolution_status"]
                .fillna("Not specified")
                .value_counts()
                .reset_index()
            )

            d.columns = ["Status", "Complaints"]

            fig = px.pie(
                d,
                names="Status",
                values="Complaints",
                hole=.55
            )

            st.plotly_chart(polish(fig), use_container_width=True)

        tmp = f_complaints.dropna(
            subset=["complaint_date"]
        ).copy()

        tmp["Month"] = (
            tmp["complaint_date"].dt.to_period("M").astype(str)
        )

        monthly = tmp.groupby("Month").size().reset_index(
            name="Complaints"
        )

        heading("Monthly Complaint Volume")

        fig = px.line(
            monthly,
            x="Month",
            y="Complaints",
            markers=True
        )

        st.plotly_chart(polish(fig), use_container_width=True)

        heading("Complaint Records")

        st.dataframe(
            f_complaints.sort_values(
                "complaint_date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="small-muted">
Telecom Customer Intelligence |
Academic Big Data Analytics Project |
Synthetic Demonstration Dataset |
Historical Descriptive Analytics
</div>
""", unsafe_allow_html=True)

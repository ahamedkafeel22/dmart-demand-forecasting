import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime

st.set_page_config(
    page_title="DMart Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

API_URL = "http://127.0.0.1:8001"

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🛒 DMart India — Demand Forecasting Dashboard")
st.markdown("**Prophet + ARIMA models | 6 stores | 4 years of data**")
st.markdown("---")

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/dmart_sales.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_daily():
    df = load_data()
    daily = df.groupby("date")["sales"].sum().reset_index()
    return daily

df = load_data()
daily = load_daily()

# ─── KPI Cards ────────────────────────────────────────────────────────────────
summary = requests.get(f"{API_URL}/summary").json()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📦 Total Records",    f"{summary['total_records']:,}")
col2.metric("🏪 Stores",           summary["stores"])
col3.metric("📊 Categories",       summary["categories"])
col4.metric("🔮 Prophet MAPE",     summary["prophet_mape"])
col5.metric("📈 ARIMA MAPE",       summary["arima_mape"])

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Historical Trends",
    "🔮 Demand Forecast",
    "🏪 Store Analysis",
    "📊 Model Comparison"
])

# ── Tab 1: Historical Trends ──────────────────────────────────────────────────
with tab1:
    st.subheader("Historical Sales Trends")

    col1, col2 = st.columns([3, 1])
    with col2:
        view = st.selectbox("View by", ["All Stores", "By Store", "By Category"])
        year_filter = st.multiselect("Year", [2021, 2022, 2023, 2024],
                                     default=[2021, 2022, 2023, 2024])

    filtered = df[df["date"].dt.year.isin(year_filter)]

    with col1:
        if view == "All Stores":
            d = filtered.groupby("date")["sales"].sum().reset_index()
            fig = px.line(d, x="date", y="sales",
                         title="Total Daily Sales — All Stores",
                         labels={"sales": "Sales (₹)", "date": "Date"},
                         color_discrete_sequence=["#1f77b4"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif view == "By Store":
            d = filtered.groupby(["date", "store"])["sales"].sum().reset_index()
            fig = px.line(d, x="date", y="sales", color="store",
                         title="Daily Sales by Store",
                         labels={"sales": "Sales (₹)", "date": "Date"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        else:
            d = filtered.groupby(["date", "category"])["sales"].sum().reset_index()
            fig = px.line(d, x="date", y="sales", color="category",
                         title="Daily Sales by Category",
                         labels={"sales": "Sales (₹)", "date": "Date"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Monthly heatmap
    st.subheader("Monthly Sales Heatmap")
    monthly = df.copy()
    monthly["month"] = monthly["date"].dt.month
    monthly["year"]  = monthly["date"].dt.year
    heat = monthly.groupby(["year", "month"])["sales"].sum().reset_index()
    heat_pivot = heat.pivot(index="year", columns="month", values="sales")
    heat_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_heat = px.imshow(heat_pivot / 1e6, text_auto=".0f",
                         color_continuous_scale="Blues",
                         title="Monthly Sales Heatmap (₹ Millions)",
                         labels={"color": "₹ Millions"})
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 2: Demand Forecast ────────────────────────────────────────────────────
with tab2:
    st.subheader("Future Demand Forecast")

    col1, col2 = st.columns([1, 3])
    with col1:
        forecast_days = st.slider("Forecast Days", 7, 365, 90)
        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Forecasting..."):
                response = requests.post(
                    f"{API_URL}/forecast",
                    json={"days": forecast_days}
                )
                forecast_data = response.json()
                st.session_state["forecast"] = forecast_data
                st.success(f"✅ {forecast_days}-day forecast ready!")

    with col2:
        if "forecast" in st.session_state:
            fc = st.session_state["forecast"]
            preds = pd.DataFrame(fc["predictions"])
            preds["date"] = pd.to_datetime(preds["date"])

            # Historical + forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["sales"] / 1e6,
                name="Historical", line=dict(color="#1f77b4", width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=preds["date"], y=preds["predicted_sales"] / 1e6,
                name="Forecast", line=dict(color="#ff7f0e", width=2, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([preds["date"], preds["date"][::-1]]),
                y=pd.concat([preds["upper_bound"] / 1e6, preds["lower_bound"][::-1] / 1e6]),
                fill="toself", fillcolor="rgba(255,127,14,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval"
            ))
            fig.update_layout(
                title=f"DMart {forecast_days}-Day Demand Forecast",
                xaxis_title="Date", yaxis_title="Sales (₹ Millions)",
                height=450, hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            st.subheader("📋 Forecast Table")
            display = preds.copy()
            display["predicted_sales"] = display["predicted_sales"].apply(lambda x: f"₹{x:,.0f}")
            display["lower_bound"]     = display["lower_bound"].apply(lambda x: f"₹{x:,.0f}")
            display["upper_bound"]     = display["upper_bound"].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(display, use_container_width=True)
        else:
            st.info("👈 Set forecast days and click Generate Forecast")

# ── Tab 3: Store Analysis ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Store-Level Analysis & Forecast")

    stores = requests.get(f"{API_URL}/stores").json()["stores"]
    col1, col2 = st.columns([1, 3])

    with col1:
        selected_store = st.selectbox("Select Store", stores)
        store_days = st.slider("Forecast Days", 7, 180, 30)
        if st.button("🏪 Forecast Store", type="primary"):
            with st.spinner(f"Forecasting {selected_store}..."):
                response = requests.get(
                    f"{API_URL}/forecast/store/{selected_store}",
                    params={"days": store_days}
                )
                st.session_state["store_forecast"] = response.json()
                st.success("✅ Done!")

        # Store KPIs
        store_data = df[df["store"] == selected_store]
        st.metric("Total Sales", f"₹{store_data['sales'].sum()/1e6:.1f}M")
        st.metric("Avg Daily",   f"₹{store_data.groupby('date')['sales'].sum().mean():,.0f}")
        st.metric("City",        selected_store.split("_")[0])

    with col2:
        if "store_forecast" in st.session_state:
            sf = st.session_state["store_forecast"]
            preds = pd.DataFrame(sf["predictions"])
            preds["date"] = pd.to_datetime(preds["date"])

            store_hist = df[df["store"] == selected_store].groupby("date")["sales"].sum().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=store_hist["date"], y=store_hist["sales"] / 1e6,
                name="Historical", line=dict(color="#1f77b4")
            ))
            fig.add_trace(go.Scatter(
                x=preds["date"], y=preds["predicted_sales"] / 1e6,
                name="Forecast", line=dict(color="#ff7f0e", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([preds["date"], preds["date"][::-1]]),
                y=pd.concat([preds["upper_bound"]/1e6, preds["lower_bound"][::-1]/1e6]),
                fill="toself", fillcolor="rgba(255,127,14,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="Confidence"
            ))
            fig.update_layout(
                title=f"{selected_store} — {store_days}-Day Forecast",
                height=400, hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Category breakdown for store
        cat_sales = store_data.groupby("category")["sales"].sum().reset_index()
        fig_pie = px.pie(cat_sales, values="sales", names="category",
                        title=f"{selected_store} — Sales by Category")
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

# ── Tab 4: Model Comparison ───────────────────────────────────────────────────
with tab4:
    st.subheader("Prophet vs ARIMA — Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prophet MAPE", "9.47%",  delta="-2.09% vs ARIMA", delta_color="normal")
    col2.metric("Prophet MAE",  "₹9.25L", delta="Better", delta_color="normal")
    col3.metric("Prophet RMSE", "₹11.9L", delta="Better", delta_color="normal")

    # Load comparison chart
    try:
        comp = pd.read_csv("outputs/model_comparison.csv")
        fig = go.Figure()
        metrics = ["MAE", "RMSE"]
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=comp["model"],
                y=comp[metric] / 1e6,
                text=(comp[metric] / 1e6).apply(lambda x: f"₹{x:.2f}M"),
                textposition="outside"
            ))
        fig.update_layout(
            title="Model Error Comparison (Lower is Better)",
            yaxis_title="Error (₹ Millions)",
            barmode="group", height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Run forecasting.py first to generate comparison data")

    # Forecast images
    st.subheader("📊 Forecast Visualizations")
    col1, col2 = st.columns(2)
    try:
        col1.image("outputs/forecast_comparison.png", caption="Model Comparison")
        col2.image("outputs/future_forecast.png",     caption="180-Day Forecast")
    except:
        st.info("Run forecasting.py to generate charts")

    # Why Prophet wins
    st.subheader("🏆 Why Prophet Outperforms ARIMA")
    st.markdown("""
    | Feature | Prophet | ARIMA |
    |---|---|---|
    | Indian Festival Effects | ✅ Built-in holiday modeling | ❌ Not supported |
    | Trend Changes | ✅ Automatic changepoint detection | ⚠️ Manual differencing |
    | Seasonality | ✅ Multiple seasonality (yearly + weekly) | ⚠️ Single seasonality |
    | Missing Data | ✅ Handles gaps | ❌ Requires complete series |
    | MAPE | **9.47%** | 11.56% |
    """)
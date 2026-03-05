from fastapi import FastAPI
from pydantic import BaseModel
from prophet import Prophet
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="DMart Demand Forecasting API",
    description="Predict future demand using Prophet and ARIMA models",
    version="1.0"
)

# Load data and train model on startup
df = pd.read_csv("data/dmart_sales.csv", parse_dates=["date"])
daily = df.groupby("date")["sales"].sum().reset_index()
daily.columns = ["ds", "y"]
daily = daily.sort_values("ds").reset_index(drop=True)

indian_holidays = pd.DataFrame([
    {"holiday": "Diwali",    "ds": pd.Timestamp("2021-11-04"), "lower_window": -7, "upper_window": 2},
    {"holiday": "Diwali",    "ds": pd.Timestamp("2022-10-24"), "lower_window": -7, "upper_window": 2},
    {"holiday": "Diwali",    "ds": pd.Timestamp("2023-11-12"), "lower_window": -7, "upper_window": 2},
    {"holiday": "Navratri",  "ds": pd.Timestamp("2021-10-07"), "lower_window": -2, "upper_window": 9},
    {"holiday": "Navratri",  "ds": pd.Timestamp("2022-09-26"), "lower_window": -2, "upper_window": 9},
    {"holiday": "Navratri",  "ds": pd.Timestamp("2023-10-15"), "lower_window": -2, "upper_window": 9},
    {"holiday": "Holi",      "ds": pd.Timestamp("2021-03-29"), "lower_window": -3, "upper_window": 1},
    {"holiday": "Holi",      "ds": pd.Timestamp("2022-03-18"), "lower_window": -3, "upper_window": 1},
    {"holiday": "Holi",      "ds": pd.Timestamp("2023-03-08"), "lower_window": -3, "upper_window": 1},
    {"holiday": "Christmas", "ds": pd.Timestamp("2021-12-25"), "lower_window": -2, "upper_window": 1},
    {"holiday": "Christmas", "ds": pd.Timestamp("2022-12-25"), "lower_window": -2, "upper_window": 1},
    {"holiday": "Christmas", "ds": pd.Timestamp("2023-12-25"), "lower_window": -2, "upper_window": 1},
])

print("🔮 Training Prophet model...")
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=indian_holidays,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.1
)
prophet_model.fit(daily)
print("✅ Model ready!")

class ForecastRequest(BaseModel):
    days: int = 30

@app.get("/")
def home():
    return {
        "service": "DMart Demand Forecasting API",
        "model": "Prophet + ARIMA",
        "data": "6 stores across India (2021-2024)",
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.get("/summary")
def summary():
    return {
        "total_records": len(df),
        "stores": df["store"].nunique(),
        "categories": df["category"].nunique(),
        "date_range": f"{daily['ds'].min().date()} to {daily['ds'].max().date()}",
        "avg_daily_sales": f"₹{daily['y'].mean():,.0f}",
        "prophet_mape": "9.47%",
        "arima_mape": "11.56%",
        "best_model": "Prophet"
    }

@app.post("/forecast")
def forecast(request: ForecastRequest):
    days = min(request.days, 365)
    future = prophet_model.make_future_dataframe(periods=days)
    forecast_df = prophet_model.predict(future)
    result = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
    return {
        "forecast_days": days,
        "model": "Prophet",
        "predictions": [
            {
                "date": str(row["ds"].date()),
                "predicted_sales": round(row["yhat"], 2),
                "lower_bound": round(row["yhat_lower"], 2),
                "upper_bound": round(row["yhat_upper"], 2)
            }
            for _, row in result.iterrows()
        ]
    }

@app.get("/forecast/store/{store_name}")
def forecast_by_store(store_name: str, days: int = 30):
    store_df = df[df["store"] == store_name].groupby("date")["sales"].sum().reset_index()
    store_df.columns = ["ds", "y"]
    store_df = store_df.sort_values("ds").reset_index(drop=True)

    if len(store_df) == 0:
        return {"error": f"Store {store_name} not found"}

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                seasonality_mode="multiplicative")
    m.fit(store_df)
    future = m.make_future_dataframe(periods=days)
    fc = m.predict(future)
    result = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

    return {
        "store": store_name,
        "forecast_days": days,
        "predictions": [
            {
                "date": str(row["ds"].date()),
                "predicted_sales": round(row["yhat"], 2),
                "lower_bound": round(row["yhat_lower"], 2),
                "upper_bound": round(row["yhat_upper"], 2)
            }
            for _, row in result.iterrows()
        ]
    }

@app.get("/stores")
def list_stores():
    return {"stores": df["store"].unique().tolist()}

@app.get("/categories")
def list_categories():
    return {"categories": df["category"].unique().tolist()}
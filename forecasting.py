import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─── 1. Load & Prepare Data ───────────────────────────────────────────────────
print("📦 Loading data...")
df = pd.read_csv("data/dmart_sales.csv", parse_dates=["date"])

# Aggregate total daily sales across all stores and categories
daily = df.groupby("date")["sales"].sum().reset_index()
daily.columns = ["ds", "y"]
daily = daily.sort_values("ds").reset_index(drop=True)

print(f"✅ Daily records: {len(daily)}")
print(f"📅 Range: {daily['ds'].min().date()} → {daily['ds'].max().date()}")
print(f"💰 Avg daily sales: ₹{daily['y'].mean():,.0f}")

# ─── 2. Train/Test Split ──────────────────────────────────────────────────────
# Last 90 days = test set
FORECAST_DAYS = 90
train = daily[:-FORECAST_DAYS]
test  = daily[-FORECAST_DAYS:]

print(f"\n📊 Train size: {len(train)} days")
print(f"🧪 Test size:  {len(test)} days")

# ─── 3. Prophet Model ─────────────────────────────────────────────────────────
print("\n🔮 Training Prophet model...")

# Indian holidays for Prophet
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
    {"holiday": "Eid",       "ds": pd.Timestamp("2021-05-13"), "lower_window": -3, "upper_window": 1},
    {"holiday": "Eid",       "ds": pd.Timestamp("2022-05-02"), "lower_window": -3, "upper_window": 1},
    {"holiday": "Eid",       "ds": pd.Timestamp("2023-04-21"), "lower_window": -3, "upper_window": 1},
    {"holiday": "Christmas", "ds": pd.Timestamp("2021-12-25"), "lower_window": -2, "upper_window": 1},
    {"holiday": "Christmas", "ds": pd.Timestamp("2022-12-25"), "lower_window": -2, "upper_window": 1},
    {"holiday": "Christmas", "ds": pd.Timestamp("2023-12-25"), "lower_window": -2, "upper_window": 1},
])

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=indian_holidays,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.1
)

prophet_model.fit(train)

# Predict on test period
future_prophet = prophet_model.make_future_dataframe(periods=FORECAST_DAYS)
forecast_prophet = prophet_model.predict(future_prophet)
pred_prophet = forecast_prophet["yhat"].values[-FORECAST_DAYS:]
pred_prophet = np.maximum(pred_prophet, 0)

# ─── 4. ARIMA Model ───────────────────────────────────────────────────────────
print("📈 Training ARIMA model...")

arima_model = ARIMA(train["y"].values, order=(7, 1, 2))
arima_fit   = arima_model.fit()
pred_arima  = arima_fit.forecast(steps=FORECAST_DAYS)
pred_arima  = np.maximum(pred_arima, 0)

# ─── 5. Metrics ───────────────────────────────────────────────────────────────
actual = test["y"].values

def get_metrics(actual, predicted, model_name):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"\n📊 {model_name} Performance:")
    print(f"   MAE:  ₹{mae:>15,.0f}")
    print(f"   RMSE: ₹{rmse:>15,.0f}")
    print(f"   MAPE: {mape:>14.2f}%")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape}

results = []
results.append(get_metrics(actual, pred_prophet, "Prophet"))
results.append(get_metrics(actual, pred_arima,   "ARIMA"))

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/model_comparison.csv", index=False)
print(f"\n🏆 Best Model (MAPE): {results_df.loc[results_df['MAPE'].idxmin(), 'model']}")

# ─── 6. Plots ─────────────────────────────────────────────────────────────────
print("\n📊 Generating plots...")

# Plot 1 — Full historical + forecast
fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.suptitle("DMart India — Demand Forecasting", fontsize=16, fontweight="bold")

# Historical trend
axes[0].plot(daily["ds"], daily["y"] / 1e6, color="#1f77b4", linewidth=1.2, label="Actual Sales")
axes[0].set_title("Historical Daily Sales (All Stores)")
axes[0].set_ylabel("Sales (₹ Millions)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Prophet vs ARIMA on test set
axes[1].plot(test["ds"].values, actual / 1e6,        color="#1f77b4", linewidth=2,   label="Actual",  alpha=0.9)
axes[1].plot(test["ds"].values, pred_prophet / 1e6,  color="#ff7f0e", linewidth=1.5, label="Prophet", linestyle="--")
axes[1].plot(test["ds"].values, pred_arima / 1e6,    color="#2ca02c", linewidth=1.5, label="ARIMA",   linestyle=":")
axes[1].set_title("Model Comparison — Test Period (Last 90 Days)")
axes[1].set_ylabel("Sales (₹ Millions)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Model metrics bar chart
x = np.arange(len(results_df))
width = 0.35
axes[2].bar(x - width/2, results_df["MAE"] / 1e6,  width, label="MAE",  color="#1f77b4", alpha=0.8)
axes[2].bar(x + width/2, results_df["RMSE"] / 1e6, width, label="RMSE", color="#ff7f0e", alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(results_df["model"])
axes[2].set_title("Model Performance Comparison")
axes[2].set_ylabel("Error (₹ Millions)")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("outputs/forecast_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2 — Future 180 day forecast
print("🔮 Generating 180-day future forecast...")
future_180 = prophet_model.make_future_dataframe(periods=180)
forecast_180 = prophet_model.predict(future_180)

fig2, ax = plt.subplots(figsize=(16, 6))
ax.plot(daily["ds"], daily["y"] / 1e6,
        color="#1f77b4", linewidth=1, label="Historical", alpha=0.7)
ax.plot(forecast_180["ds"].tail(180), forecast_180["yhat"].tail(180) / 1e6,
        color="#ff7f0e", linewidth=2, label="Forecast (180 days)", linestyle="--")
ax.fill_between(
    forecast_180["ds"].tail(180),
    forecast_180["yhat_lower"].tail(180) / 1e6,
    forecast_180["yhat_upper"].tail(180) / 1e6,
    alpha=0.2, color="#ff7f0e", label="Confidence Interval"
)
ax.axvline(daily["ds"].max(), color="red", linestyle=":", linewidth=1.5, label="Forecast Start")
ax.set_title("DMart — 180-Day Demand Forecast (Prophet)", fontsize=14, fontweight="bold")
ax.set_ylabel("Total Sales (₹ Millions)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/future_forecast.png", dpi=150, bbox_inches="tight")
plt.close()

# Save forecast data
forecast_180[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(180).to_csv(
    "outputs/future_forecast.csv", index=False
)

print("\n✅ All outputs saved to outputs/")
print("   - forecast_comparison.png")
print("   - future_forecast.png")
print("   - model_comparison.csv")
print("   - future_forecast.csv")
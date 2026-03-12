# 🛒 DMart India — Demand Forecasting System

> Retail demand forecasting for DMart India using Prophet and ARIMA models, served via FastAPI and visualized in an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-0099CC?style=for-the-badge&logo=facebook&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 🏢 Business Understanding

| | |
|---|---|
| **Business Problem** | DMart and large retail chains face significant revenue loss from two inventory problems: overstocking (capital tied up, wastage, storage costs) and understocking (lost sales, poor customer experience). Without accurate demand forecasting, procurement decisions are based on intuition rather than data. |
| **Business Objective** | Develop time-series forecasting models that accurately predict product demand at store level — enabling procurement teams to make data-driven restocking decisions weeks in advance, reducing both overstock waste and stockout revenue loss. |
| **Business Constraint** | Forecasts must be available 2–4 weeks ahead of procurement deadlines, models must handle Indian retail demand patterns (Diwali spikes, monsoon slowdowns, festival seasons), and results must be presented in a dashboard accessible to non-technical procurement managers. |
| **Business Success Criteria** | Procurement managers can view demand forecasts for the next 4 weeks via a dashboard — replacing manual estimation and enabling data-driven purchase orders before each replenishment cycle. |
| **ML Success Criteria** | Best forecasting model achieves MAPE (Mean Absolute Percentage Error) < 10% on held-out test data across all product categories. |
| **Economic Success Criteria** | A 10% reduction in overstock waste on ₹500 Cr annual inventory = ₹50 Cr saved. A 5% reduction in stockout-driven lost sales on ₹1,000 Cr revenue = ₹50 Cr recovered. Combined potential impact: ₹100 Cr+ annually for a large DMart operation. |

## 🎯 Project Overview

This project builds an end-to-end demand forecasting system for DMart India — one of India's largest retail chains with 415+ stores nationwide.

**Focus areas:**
- Time series forecasting with Prophet and ARIMA
- Indian festival and seasonality modeling (Diwali, Navratri, Holi, Eid)
- REST API for forecast serving via FastAPI
- Interactive 4-tab Streamlit dashboard

---

## 🖼️ Dashboard Screenshots

| Historical Trends | Demand Forecast |
|---|---|
| ![Historical](docs/Historical%20Trends.png) | ![Forecast](docs/Demand%20Forecast.png) |

| Store Analysis | Model Comparison |
|---|---|
| ![Store](docs/Store%20Analysis.png) | ![Model](docs/Model%20Comparison.png) |

---

## 1. System Architecture

```
Dataset (52,596 records)
        ↓
Feature Engineering
(festivals, seasonality, weekends, growth trend)
        ↓
Forecasting Models
(Prophet + ARIMA)
        ↓
Model Evaluation
(MAE, RMSE, MAPE on 90-day test set)
        ↓
FastAPI Service
(REST endpoints for predictions)
        ↓
Streamlit Dashboard
(4-tab interactive visualization)
```

---

## 2. Dataset

Synthetic dataset generated using statistical patterns derived from Indian retail seasonality and DMart's historical growth rates (publicly reported in annual filings).

| Feature | Details |
|---|---|
| Date Range | 2021-01-01 → 2024-12-31 |
| Total Records | 52,596 |
| Stores | 6 (Mumbai, Pune, Hyderabad, Bangalore, Chennai, Ahmedabad) |
| Categories | Groceries, FMCG, Dairy, Apparel, Electronics, Home Care |
| Avg Daily Sales | ~₹63 Lakhs across all stores |

**Patterns built into data:**
- 📈 ~20% YoY growth trend (aligned with DMart's reported revenue growth)
- 🪔 Festival boosts — Diwali (+80%), Navratri (+40%), Holi (+30%), Eid (+30%)
- 🌧️ Monsoon slowdown — June-August (-18%)
- 📅 Weekend spike — Saturday (+20%), Sunday (+35%)
- 🗓️ Monthly seasonality — Oct-Nov festive peak, January post-holiday dip

---

## 3. Forecasting Pipeline

```
Raw Data (52,596 records)
        ↓
Aggregate to daily total sales
        ↓
Train/Test Split (last 90 days = test)
        ↓
Prophet Model (with Indian holiday effects)
        ↓
ARIMA(7,1,2) baseline model
        ↓
Evaluate — MAE, RMSE, MAPE
        ↓
Generate 180-day future forecast
        ↓
Serve via FastAPI
        ↓
Visualize in Streamlit Dashboard
```

---

## 4. Model Performance

| Metric | Prophet | ARIMA |
|---|---|---|
| **MAPE** | **9.47%** | 11.56% |
| MAE | ₹9,24,612 | ₹13,20,792 |
| RMSE | ₹11,93,484 | ₹22,11,130 |
| Indian Holidays | ✅ Built-in | ❌ Not supported |
| Weekly Seasonality | ✅ | ⚠️ Limited |
| **Winner** | 🏆 **Prophet** | — |

**ARIMA(7,1,2)** — order selected after inspecting ACF/PACF plots and minimizing AIC. AR(7) captures the weekly cycle, one differencing for stationarity, MA(2) for short-term error correction.

**Prophet wins** because it natively models Indian festival effects, multiple seasonalities, and automatic trend changepoint detection.

---

## 5. Project Structure

```
dmart-demand-forecasting/
│
├── app/
│   └── forecast_api.py      # FastAPI endpoints
│
├── data/
│   └── dmart_sales.csv      # Generated dataset
│
├── outputs/
│   ├── forecast_comparison.png
│   ├── future_forecast.png
│   ├── future_forecast.csv
│   └── model_comparison.csv
│
├── docs/                    # Dashboard screenshots
├── generate_data.py         # Synthetic data generator
├── forecasting.py           # Model training + evaluation
├── dashboard.py             # Streamlit dashboard
└── README.md
```

---

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/summary` | Model metrics + dataset info |
| POST | `/forecast` | Forecast next N days |
| GET | `/forecast/store/{store}` | Store-level forecast |
| GET | `/stores` | List all stores |
| GET | `/categories` | List all categories |

### Example Request
```bash
curl -X POST http://127.0.0.1:8001/forecast \
  -H "Content-Type: application/json" \
  -d '{"days": 30}'
```

---

## 7. How to Run

```bash
# Clone repo
git clone https://github.com/ahamedkafeel22/dmart-demand-forecasting.git
cd dmart-demand-forecasting

# Install dependencies
pip install prophet statsmodels fastapi uvicorn streamlit pandas numpy matplotlib plotly scikit-learn python-multipart

# Generate dataset
python generate_data.py

# Train models + generate outputs
python forecasting.py

# Terminal 1 — Start API
uvicorn app.forecast_api:app --reload --port 8001

# Terminal 2 — Start Dashboard
streamlit run dashboard.py --server.port 8502
```

---

## 8. Dashboard Features

**Tab 1 — Historical Trends:**
- Interactive line chart (All Stores / By Store / By Category)
- Year filter
- Monthly sales heatmap

**Tab 2 — Demand Forecast:**
- Adjustable forecast horizon (7–365 days)
- Historical + forecast chart with confidence intervals
- Full forecast table with lower/upper bounds

**Tab 3 — Store Analysis:**
- Per-store forecast
- Store KPIs (total sales, avg daily, city)
- Category breakdown pie chart

**Tab 4 — Model Comparison:**
- Prophet vs ARIMA metrics
- Error comparison bar chart
- 180-day forecast visualization
- Feature comparison table

---

## 9. Key Insights

- **Diwali** is the strongest sales driver — up to ~80% above baseline in the pre-festival window
- **Monsoon months** (June–August) show a consistent ~18% dip across all stores
- **Mumbai Andheri** is the highest revenue store with ~₹850K average daily sales
- **Groceries + FMCG** account for ~60% of total category sales
- Prophet achieves **9.47% MAPE** — strong accuracy for retail demand forecasting

---

## 👤 Author

**Syed Kafeel Ahamed**

Finance professional with 6+ years of accounting experience transitioning into Data Science and AI.

🔗 [LinkedIn](https://www.linkedin.com/in/syed-kafeel-ahamed-ab465036b) | [GitHub](https://github.com/ahamedkafeel22)

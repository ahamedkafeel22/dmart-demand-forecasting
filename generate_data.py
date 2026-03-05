import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Configuration
START_DATE = "2021-01-01"
END_DATE   = "2024-12-31"
STORES = {
    "Mumbai_Andheri":   {"base": 850000, "size": "large"},
    "Pune_Kothrud":     {"base": 620000, "size": "medium"},
    "Hyderabad_Madhapur": {"base": 700000, "size": "large"},
    "Bangalore_Koramangala": {"base": 780000, "size": "large"},
    "Chennai_Velachery": {"base": 590000, "size": "medium"},
    "Ahmedabad_Satellite": {"base": 640000, "size": "medium"},
}

CATEGORIES = ["Groceries", "FMCG", "Dairy", "Apparel", "Electronics", "Home_Care"]

# Indian festivals (month, day, boost multiplier)
FESTIVALS = {
    "Diwali":    [(10, 24, 2026), (11, 12, 2023), (10, 24, 2022), (11, 4, 2021)],
    "Navratri":  [(10, 3,  2025), (10, 15, 2023), (9, 26, 2022),  (10, 7, 2021)],
    "Holi":      [(3, 25, 2024),  (3, 8, 2023),   (3, 18, 2022),  (3, 29, 2021)],
    "Eid":       [(4, 10, 2024),  (4, 21, 2023),  (5, 2, 2022),   (5, 13, 2021)],
    "Christmas": [(12, 25, 2024), (12, 25, 2023), (12, 25, 2022), (12, 25, 2021)],
    "NewYear":   [(1, 1, 2024),   (1, 1, 2023),   (1, 1, 2022),   (1, 1, 2021)],
}

def get_festival_boost(date):
    boost = 1.0
    for festival, dates in FESTIVALS.items():
        for month, day, year in dates:
            try:
                fest_date = datetime(year, month, day)
                delta = abs((date - fest_date).days)
                if delta <= 7:
                    boost = max(boost, 1.8 - (delta * 0.1))
                elif delta <= 14:
                    boost = max(boost, 1.3 - (delta * 0.02))
            except:
                pass
    return boost

def get_monsoon_factor(date):
    # Monsoon slowdown June-August (people shop less)
    if date.month in [6, 7, 8]:
        return 0.82
    return 1.0

def get_weekend_boost(date):
    if date.weekday() == 6:  # Sunday
        return 1.35
    elif date.weekday() == 5:  # Saturday
        return 1.20
    return 1.0

def get_month_factor(month):
    # January post-holiday dip, Oct-Nov festive peak
    factors = {
        1: 0.85, 2: 0.90, 3: 1.05, 4: 1.00,
        5: 0.95, 6: 0.85, 7: 0.82, 8: 0.85,
        9: 1.05, 10: 1.25, 11: 1.30, 12: 1.20
    }
    return factors.get(month, 1.0)

# Generate data
records = []
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="D")

for store_name, store_info in STORES.items():
    base_sales = store_info["base"]
    yearly_growth = 1.0

    for date in date_range:
        # Yearly growth trend (DMart grows ~20% YoY)
        if date.year == 2021: yearly_growth = 1.00
        elif date.year == 2022: yearly_growth = 1.18
        elif date.year == 2023: yearly_growth = 1.38
        elif date.year == 2024: yearly_growth = 1.55

        festival_boost  = get_festival_boost(date)
        monsoon_factor  = get_monsoon_factor(date)
        weekend_boost   = get_weekend_boost(date)
        month_factor    = get_month_factor(date.month)
        noise           = np.random.normal(1.0, 0.06)

        total_sales = (
            base_sales
            * yearly_growth
            * festival_boost
            * monsoon_factor
            * weekend_boost
            * month_factor
            * noise
        )

        # Split by category
        category_split = {
            "Groceries": 0.35,
            "FMCG":      0.25,
            "Dairy":     0.15,
            "Apparel":   0.10,
            "Electronics": 0.08,
            "Home_Care": 0.07,
        }

        for category, share in category_split.items():
            cat_noise = np.random.normal(1.0, 0.04)
            # Apparel spikes during festivals
            if category == "Apparel" and festival_boost > 1.3:
                cat_noise *= 1.5
            # Electronics spike during Diwali
            if category == "Electronics" and festival_boost > 1.6:
                cat_noise *= 2.0

            records.append({
                "date":          date,
                "store":         store_name,
                "city":          store_name.split("_")[0],
                "category":      category,
                "sales":         round(total_sales * share * cat_noise, 2),
                "transactions":  int(np.random.normal(2800, 300) * weekend_boost * festival_boost),
                "is_festival":   1 if festival_boost > 1.2 else 0,
                "is_weekend":    1 if date.weekday() >= 5 else 0,
                "month":         date.month,
                "year":          date.year,
                "day_of_week":   date.weekday(),
            })

df = pd.DataFrame(records)
df.to_csv("data/dmart_sales.csv", index=False)
print(f"✅ Dataset generated: {len(df):,} records")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
print(f"🏪 Stores: {df['store'].nunique()}")
print(f"📦 Categories: {df['category'].nunique()}")
print(df.head())
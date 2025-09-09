"""Generate sample Amazon search data for AutoML training."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_amazon_data(num_records: int = 10000) -> pd.DataFrame:
    """Generate sample Amazon search data for time-series forecasting."""
    np.random.seed(42)

    # Sample product categories and search terms
    categories = [
        "Electronics",
        "Books",
        "Clothing",
        "Home",
        "Sports",
        "Beauty",
        "Toys",
    ]
    products = [
        "wireless headphones",
        "laptop stand",
        "coffee maker",
        "running shoes",
        "mystery novel",
        "winter jacket",
        "yoga mat",
        "face cream",
        "board game",
        "smartphone case",
        "desk lamp",
        "protein powder",
        "hiking boots",
        "cookbook",
        "winter coat",
        "resistance bands",
        "moisturizer",
        "puzzle",
    ]

    # Generate date range (last 2 years for time-series)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    data = []
    for i in range(num_records):
        # Generate timestamp
        random_days = np.random.randint(0, 730)
        timestamp = start_date + timedelta(days=random_days)

        # Generate search metrics
        category = np.random.choice(categories)
        product = np.random.choice(products)

        # Simulate seasonal trends and weekly patterns
        day_of_week = timestamp.weekday()
        month = timestamp.month

        # Higher search volume on weekends and holidays
        base_searches = np.random.poisson(100)
        weekend_boost = 1.3 if day_of_week >= 5 else 1.0
        holiday_boost = 1.5 if month in [11, 12] else 1.0

        search_volume = int(base_searches * weekend_boost * holiday_boost)
        click_through_rate = np.random.normal(0.15, 0.05)
        click_through_rate = max(0.01, min(0.5, click_through_rate))

        conversion_rate = np.random.normal(0.08, 0.03)
        conversion_rate = max(0.001, min(0.3, conversion_rate))

        # Revenue follows power law distribution
        avg_order_value = np.random.lognormal(3.5, 0.5)
        revenue = search_volume * click_through_rate * conversion_rate * avg_order_value

        data.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "date": timestamp.strftime("%Y-%m-%d"),
                "category": category,
                "product_search_term": product,
                "search_volume": search_volume,
                "click_through_rate": round(click_through_rate, 4),
                "conversion_rate": round(conversion_rate, 4),
                "avg_order_value": round(avg_order_value, 2),
                "revenue": round(revenue, 2),
                "day_of_week": day_of_week,
                "month": month,
                "quarter": (month - 1) // 3 + 1,
            }
        )

    return pd.DataFrame(data).sort_values("timestamp").reset_index(drop=True)


def save_sample_data(output_dir: str = "data"):
    """Generate and save sample data files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate sample data
    df = generate_sample_amazon_data()

    # Save full dataset
    full_path = output_path / "amazon_search_data.csv"
    df.to_csv(full_path, index=False)
    print(f"Generated {len(df)} records saved to {full_path}")

    # Create aggregated daily data for time-series forecasting
    daily_agg = (
        df.groupby(["date", "category"])
        .agg(
            {
                "search_volume": "sum",
                "revenue": "sum",
                "click_through_rate": "mean",
                "conversion_rate": "mean",
            }
        )
        .reset_index()
    )

    # Add time-based features
    daily_agg["day_of_week"] = pd.to_datetime(daily_agg["date"]).dt.dayofweek
    daily_agg["month"] = pd.to_datetime(daily_agg["date"]).dt.month
    daily_agg["quarter"] = pd.to_datetime(daily_agg["date"]).dt.quarter

    daily_path = output_path / "amazon_search_daily.csv"
    daily_agg.to_csv(daily_path, index=False)
    print(f"Daily aggregated data saved to {daily_path}")

    return str(full_path), str(daily_path)


if __name__ == "__main__":
    save_sample_data()

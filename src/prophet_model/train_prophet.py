"""
src/prophet_model/train_prophet.py — Bước 2: Tầng 1 cốt lõi với Prophet.

Huấn luyện Prophet trên Clean_Revenue / Clean_COGS.
- Yearly + Weekly seasonality (Fourier)
- Custom holidays: end-of-month spike, Tết Nguyên Đán, Black Friday
- Trend changepoint detection tự động
"""
import pandas as pd
import numpy as np
import logging
from prophet import Prophet

logger = logging.getLogger("gridbreaker")


def _build_holidays() -> pd.DataFrame:
    """Tạo bảng holidays cho Prophet: end-of-month spikes & các ngày lễ đặc biệt."""
    rows = []

    # End-of-month (ngày 28-31 thường có spike doanh thu)
    for year in range(2012, 2025):
        for month in range(1, 13):
            # Last 3 days of month
            if month == 12 and year == 2024:
                continue
            try:
                eom = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
                for delta in range(4):
                    d = eom - pd.Timedelta(days=delta)
                    rows.append({"holiday": "end_of_month", "ds": d})
            except:
                pass

    # First 3 days of month (post-spike recovery)
    for year in range(2012, 2025):
        for month in range(1, 13):
            for day in [1, 2, 3]:
                try:
                    rows.append({"holiday": "start_of_month", "ds": pd.Timestamp(year, month, day)})
                except:
                    pass

    # Tết Nguyên Đán (xấp xỉ, các năm 2012-2024)
    tet_dates = [
        "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19",
        "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
        "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
    ]
    for td in tet_dates:
        base = pd.Timestamp(td)
        for delta in range(-2, 8):  # 10 ngày quanh Tết
            rows.append({"holiday": "tet", "ds": base + pd.Timedelta(days=delta)})

    return pd.DataFrame(rows)


def fit_prophet(series: pd.Series, target_name: str = "Revenue") -> Prophet:
    """
    Huấn luyện Prophet trên chuỗi sạch.
    
    Args:
        series: pd.Series(index=DatetimeIndex, values=Clean_Revenue hoặc Clean_COGS)
        target_name: tên biến (để log)
    Returns:
        model: Prophet đã fit
    """
    logger.info(f"  Fitting Prophet cho {target_name}...")

    train_df = pd.DataFrame({"ds": series.index, "y": series.values})
    holidays = _build_holidays()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.15,   # linh hoạt hơn default (0.05)
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_range=0.9,
    )
    model.fit(train_df)

    logger.info(f"  Prophet {target_name}: fitted with {len(train_df)} datapoints, {len(holidays)} holiday rows.")
    return model


def predict_prophet(model: Prophet, start: str, end: str) -> pd.DataFrame:
    """
    Dự báo Prophet trên khoảng [start, end].
    
    Returns:
        DataFrame: index=Date, columns=['prophet_pred', 'prophet_trend']
    """
    future = model.make_future_dataframe(
        periods=(pd.Timestamp(end) - model.history["ds"].max()).days,
        freq="D"
    )
    forecast = model.predict(future)
    forecast = forecast.set_index("ds")
    result = forecast.loc[start:end, ["yhat", "trend"]].rename(
        columns={"yhat": "prophet_pred", "trend": "prophet_trend"}
    )
    return result

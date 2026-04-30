"""
src/denoising/denoise.py — Bước 1: Khử nhiễu mục tiêu (Target Denoising).

Mục tiêu: Tạo Clean_Revenue và Clean_COGS bằng cách:
  1. Impute ngày stockout (hết hàng) bằng trung bình lân cận.
  2. Cap spike bất thường (> mean + 3σ trong sliding window 30 ngày),
     nhưng BẢO TỒN spike mang tính hệ thống (end-of-month, lặp hàng năm).
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("gridbreaker")


def _detect_recurring_spikes(series: pd.Series, threshold: float = 0.7) -> pd.Series:
    """
    Phát hiện spike lặp lại hàng năm (seasonal signature).
    Nếu cùng (month, day) có spike trong >= threshold*tổng số năm → giữ lại.
    """
    df = pd.DataFrame({"val": series, "month": series.index.month, "day": series.index.day, "year": series.index.year})
    # Tính z-score theo năm
    annual_mean = df.groupby("year")["val"].transform("mean")
    annual_std = df.groupby("year")["val"].transform("std")
    df["zscore"] = (df["val"] - annual_mean) / annual_std
    df["is_spike"] = df["zscore"] > 2.0

    # Tỷ lệ năm có spike cho mỗi (month, day)
    spike_rate = df.groupby(["month", "day"])["is_spike"].mean()
    recurring = spike_rate[spike_rate >= threshold].index
    is_recurring = pd.Series(False, index=series.index)
    for m, d in recurring:
        mask = (series.index.month == m) & (series.index.day == d)
        is_recurring[mask] = True
    return is_recurring


def _cap_outliers(series: pd.Series, window: int = 30, n_sigma: float = 3.0) -> pd.Series:
    """Cap spike bất thường, giữ nguyên recurring spikes."""
    recurring = _detect_recurring_spikes(series)
    rolling_mean = series.rolling(window, center=True, min_periods=7).mean()
    rolling_std = series.rolling(window, center=True, min_periods=7).std()
    upper = rolling_mean + n_sigma * rolling_std
    is_outlier = (series > upper) & (~recurring)
    cleaned = series.copy()
    cleaned[is_outlier] = upper[is_outlier]
    n_capped = is_outlier.sum()
    logger.info(f"  Capped {n_capped} non-recurring outlier(s).")
    return cleaned


def _impute_stockout_days(series: pd.Series, stockout_flags: pd.Series) -> pd.Series:
    """Impute ngày stockout bằng rolling mean 7 ngày lân cận."""
    aligned = stockout_flags.reindex(series.index).fillna(0)
    is_stockout = aligned == 1
    if is_stockout.sum() == 0:
        logger.info("  Không có ngày stockout nào khớp với sales data.")
        return series
    cleaned = series.copy()
    rolling_fill = series.rolling(7, center=True, min_periods=1).mean()
    cleaned[is_stockout] = rolling_fill[is_stockout]
    logger.info(f"  Imputed {is_stockout.sum()} stockout day(s).")
    return cleaned


def denoise_target(df: pd.DataFrame, stockout_flags: pd.Series = None) -> pd.DataFrame:
    """
    Bước 1: Tạo Clean_Revenue và Clean_COGS.
    
    Args:
        df: DataFrame với index=Date, columns=['Revenue','COGS']
        stockout_flags: Series với index=Date, values=0/1
    Returns:
        df with new columns: Clean_Revenue, Clean_COGS
    """
    logger.info("=" * 50)
    logger.info("BƯỚC 1: TARGET DENOISING")
    logger.info("=" * 50)

    result = df.copy()
    for col in ["Revenue", "COGS"]:
        logger.info(f"Processing {col}:")
        cleaned = df[col].copy()

        # 1. Impute stockout days
        if stockout_flags is not None:
            cleaned = _impute_stockout_days(cleaned, stockout_flags)

        # 2. Cap non-recurring outliers
        cleaned = _cap_outliers(cleaned.dropna())

        result[f"Clean_{col}"] = cleaned

    logger.info(f"  Revenue: mean={result['Revenue'].mean():,.0f} → Clean mean={result['Clean_Revenue'].mean():,.0f}")
    logger.info(f"  COGS:    mean={result['COGS'].mean():,.0f} → Clean mean={result['Clean_COGS'].mean():,.0f}")
    return result

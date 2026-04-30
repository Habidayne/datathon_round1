"""
src/utils/helpers.py — Các hàm tiện ích dùng chung cho toàn bộ pipeline.
"""
import logging
import os
import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..")  # project root


def setup_logger(log_path: str = None) -> logging.Logger:
    """Tạo logger ghi ra cả console và file log.txt."""
    if log_path is None:
        log_path = os.path.join(DATA_DIR, "log.txt")

    logger = logging.getLogger("gridbreaker")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_sales(path: str = None) -> pd.DataFrame:
    """Load sales.csv, parse dates, set Date index, reindex daily, interpolate."""
    if path is None:
        path = os.path.join(DATA_DIR, "sales.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    full_idx = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(full_idx)
    df["Revenue"] = df["Revenue"].interpolate(method="time")
    df["COGS"] = df["COGS"].interpolate(method="time")
    return df


def load_inventory_flags(path: str = None) -> pd.DataFrame:
    """Load inventory.csv, tổng hợp stockout_flag theo ngày."""
    if path is None:
        path = os.path.join(DATA_DIR, "inventory.csv")
    inv = pd.read_csv(path)
    # Tạo date column từ year+month (snapshot ở tháng)
    inv["snapshot_date"] = pd.to_datetime(
        inv["year"].astype(str) + "-" + inv["month"].astype(str) + "-01"
    )
    # Tổng hợp: ngày nào có ≥ 1 product stockout → flag = 1
    daily_stockout = (
        inv.groupby("snapshot_date")["stockout_flag"]
        .max()
        .reset_index()
        .rename(columns={"snapshot_date": "Date"})
        .set_index("Date")
    )
    return daily_stockout

"""
src/postprocess/blend.py — Bước 4: Hậu xử lý và Blend kết quả.

Final_Forecast = Prophet_Prediction + LightGBM_Residual_Prediction
Sau đó clip âm → 0 và đảm bảo đúng format submission.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("gridbreaker")


def blend_forecasts(
    prophet_pred: pd.Series,
    lgbm_residual_pred: pd.Series,
) -> pd.Series:
    """
    Tổng hợp: Final = Prophet + LGBM_Residual
    """
    final = prophet_pred + lgbm_residual_pred
    return final


def postprocess(
    revenue_pred: pd.Series,
    cogs_pred: pd.Series,
    sample_submission_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Hậu xử lý và xuất submission.csv:
    - Clip giá trị âm → 0
    - Đảm bảo COGS ≤ Revenue (nếu vi phạm, set COGS = Revenue * 0.85)
    - Đúng format sample_submission.csv
    """
    logger.info("=" * 50)
    logger.info("BƯỚC 4: HẬU XỬ LÝ & XUẤT SUBMISSION")
    logger.info("=" * 50)

    sample = pd.read_csv(sample_submission_path)
    n_expected = len(sample)

    sub = pd.DataFrame({
        "Date": revenue_pred.index.strftime("%Y-%m-%d"),
        "Revenue": revenue_pred.values,
        "COGS": cogs_pred.values,
    })

    # Clip âm
    n_neg = (sub[["Revenue", "COGS"]] < 0).sum().sum()
    sub["Revenue"] = sub["Revenue"].clip(lower=0)
    sub["COGS"] = sub["COGS"].clip(lower=0)
    if n_neg > 0:
        logger.info(f"  Clipped {n_neg} negative value(s) → 0.")

    # COGS ≤ Revenue (logic kinh doanh)
    violate = sub["COGS"] > sub["Revenue"]
    if violate.any():
        sub.loc[violate, "COGS"] = sub.loc[violate, "Revenue"] * 0.85
        logger.info(f"  Fixed {violate.sum()} row(s) where COGS > Revenue.")

    # Round
    sub["Revenue"] = sub["Revenue"].round(2)
    sub["COGS"] = sub["COGS"].round(2)

    # Verify length
    assert len(sub) == n_expected, f"Submission has {len(sub)} rows, expected {n_expected}!"

    sub.to_csv(output_path, index=False)
    logger.info(f"  ✅ Saved submission: {output_path} ({len(sub)} rows)")
    logger.info(f"     Revenue: mean={sub['Revenue'].mean():,.0f}, min={sub['Revenue'].min():,.0f}, max={sub['Revenue'].max():,.0f}")
    logger.info(f"     COGS:    mean={sub['COGS'].mean():,.0f}, min={sub['COGS'].min():,.0f}, max={sub['COGS'].max():,.0f}")

    return sub

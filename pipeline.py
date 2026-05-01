"""
pipeline.py — 🔥 The Gridbreaker — Main Orchestrator

Pipeline dự đoán Revenue & COGS hàng ngày cho kỳ test 01/01/2023 → 01/07/2024.
4 bước:
  1. Target Denoising   (src/denoising)
  2. Prophet             (src/prophet_model)
  3. LightGBM Residuals  (src/lgbm_model)
  4. Blend & Postprocess (src/postprocess)

Usage:
    python pipeline.py
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Setup paths ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils import setup_logger, mape, load_sales, load_inventory_flags, SEED
from src.denoising import denoise_target
from src.prophet_model import fit_prophet, predict_prophet
from src.lgbm_model import fit_lgbm_residual, predict_lgbm_residual, make_future_safe_features
from src.postprocess import blend_forecasts, postprocess

np.random.seed(SEED)

# ── Config ───────────────────────────────────────────────
CSV_DIR         = os.path.join(ROOT, "csv")
SALES_FILE      = os.path.join(CSV_DIR, "sales.csv")
INVENTORY_FILE  = os.path.join(CSV_DIR, "inventory.csv")
SAMPLE_SUB_FILE = os.path.join(CSV_DIR, "sample_submission.csv")
OUTPUT_FILE     = os.path.join(ROOT, "submission.csv")
LOG_FILE        = os.path.join(ROOT, "log.txt")

TRAIN_END  = "2022-12-31"
VAL_START  = "2021-01-01"
VAL_END    = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END   = "2024-07-01"

# ── Logger ───────────────────────────────────────────────
logger = setup_logger(LOG_FILE)
logger.info("🔥 THE GRIDBREAKER PIPELINE — BẮT ĐẦU")
logger.info(f"Train: ... → {TRAIN_END}")
logger.info(f"Test:  {TEST_START} → {TEST_END}")


def run_pipeline():
    # ═══════════════════════════════════════════════════════
    # LOAD DATA
    # ═══════════════════════════════════════════════════════
    logger.info("Loading sales data...")
    df = load_sales(SALES_FILE)
    logger.info(f"Sales: {df.index.min().date()} → {df.index.max().date()} ({len(df)} rows)")

    logger.info("Loading inventory stockout flags...")
    stockout_flags = load_inventory_flags(INVENTORY_FILE)
    logger.info(f"Stockout flags: {len(stockout_flags)} monthly snapshots")

    # ═══════════════════════════════════════════════════════
    # BƯỚC 1: TARGET DENOISING
    # ═══════════════════════════════════════════════════════
    df = denoise_target(df, stockout_flags.get("stockout_flag", None))

    train = df.loc[:TRAIN_END]
    # Extend full index to cover test period
    full_idx = pd.date_range(df.index.min(), TEST_END)
    df_extended = df.reindex(full_idx)

    # ═══════════════════════════════════════════════════════
    # BƯỚC 2: PROPHET
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 50)
    logger.info("BƯỚC 2: PROPHET — Tầng 1 cốt lõi")
    logger.info("=" * 50)

    results = {}
    for target in ["Revenue"]:
        clean_col = f"Clean_{target}"
        train_series = train[clean_col].dropna()

        # Fit Prophet
        model = fit_prophet(train_series, target_name=target)

        # Predict full range (train + test)
        prophet_preds = predict_prophet(model, str(df.index.min().date()), TEST_END)

        # Tính Residual trên train
        train_prophet = prophet_preds.loc[:TRAIN_END, "prophet_pred"]
        residual_train = train[target].reindex(train_prophet.index) - train_prophet

        # ═══════════════════════════════════════════════════
        # BƯỚC 3: LGBM RESIDUAL
        # ═══════════════════════════════════════════════════
        logger.info("=" * 50)
        logger.info(f"BƯỚC 3: LGBM RESIDUAL — {target}")
        logger.info("=" * 50)

        # Build features cho toàn bộ range
        full_residuals = pd.Series(np.nan, index=full_idx, name="residual")
        full_residuals.update(residual_train)

        full_trend = prophet_preds["prophet_trend"].reindex(full_idx)

        X_full = make_future_safe_features(full_residuals, full_trend, full_idx)

        # Train LGBM trên train period
        X_train = X_full.loc[:TRAIN_END]
        y_train = residual_train.reindex(X_train.index)
        lgbm_model = fit_lgbm_residual(X_train, y_train, target_name=target)

        # ═══════════════════════════════════════════════════
        # BLEND & ROLLING FORECAST
        # ═══════════════════════════════════════════════════
        logger.info("=" * 50)
        logger.info("BƯỚC 3.5: ROLLING FORECAST (TEST PERIOD)")
        logger.info("=" * 50)
        
        test_years = range(
            pd.Timestamp(TEST_START).year,
            pd.Timestamp(TEST_END).year + 1
        )

        all_predictions = []
        residuals_rolling = full_residuals.copy()

        for year in test_years:
            year_start = f"{year}-01-01"
            year_end   = f"{year}-12-31" if year < pd.Timestamp(TEST_END).year else TEST_END

            # Rebuild X_full với residuals hiện có (cập nhật sau mỗi năm)
            X_full_iter = make_future_safe_features(residuals_rolling, full_trend, full_idx)
            
            # Predict năm hiện tại
            test_prophet_yr = prophet_preds.loc[year_start:year_end, "prophet_pred"]
            X_test_yr       = X_full_iter.loc[year_start:year_end]
            test_resid_yr   = predict_lgbm_residual(lgbm_model, X_test_yr)
            test_revenue_yr = blend_forecasts(test_prophet_yr, test_resid_yr)

            all_predictions.append(test_revenue_yr)

            # Extend residuals với năm vừa predict (proxy cho năm tiếp theo)
            proxy = test_revenue_yr - prophet_preds.loc[year_start:year_end, "prophet_pred"]
            residuals_rolling.update(proxy)

        # Ghép tất cả năm lại
        final_pred = pd.concat(all_predictions)
        final_pred = final_pred.loc[TEST_START:TEST_END]

        results[target] = final_pred

        # ── Validation: metrics trên 2021-2022 (in-sample — LGBM đã thấy data này) ──
        # Lưu ý: đây là in-sample validation vì VAL trong TRAIN_END.
        # Dùng để ước lượng hiệu suất, không phải holdout thực sự.
        val_prophet = prophet_preds.loc[VAL_START:VAL_END, "prophet_pred"]
        X_val = X_full.loc[VAL_START:VAL_END]
        val_resid = predict_lgbm_residual(lgbm_model, X_val)
        val_final = blend_forecasts(val_prophet, val_resid)
        actual_val = df.loc[VAL_START:VAL_END, target].reindex(val_final.index).dropna()
        val_aligned = val_final.reindex(actual_val.index)

        mae_v  = mean_absolute_error(actual_val, val_aligned)
        rmse_v = np.sqrt(mean_squared_error(actual_val, val_aligned))
        r2_v   = r2_score(actual_val, val_aligned)
        mape_v = mape(actual_val.values, val_aligned.values)
        logger.info(f"  📊 {target} Validation (2021-2022):")
        logger.info(f"     MAE  = {mae_v/1e6:.3f}M VND")
        logger.info(f"     RMSE = {rmse_v/1e6:.3f}M VND")
        logger.info(f"     R²   = {r2_v:.4f}")
        logger.info(f"     MAPE = {mape_v:.2f}%")

    # ═══════════════════════════════════════════════════════
    # BƯỚC 4: POSTPROCESS & SUBMISSION
    # ═══════════════════════════════════════════════════════
    sub = postprocess(
        revenue_pred=results["Revenue"],
        gross_margin=0.175,
        sample_submission_path=SAMPLE_SUB_FILE,
        output_path=OUTPUT_FILE,
    )

    logger.info("🏁 THE GRIDBREAKER PIPELINE — HOÀN TẤT!")
    return sub


if __name__ == "__main__":
    run_pipeline()

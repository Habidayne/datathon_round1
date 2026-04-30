"""
src/lgbm_model/train_lgbm.py — Bước 3: LightGBM trên Residuals.

LightGBM học phần dư (Residual = Actual - Prophet_Prediction).
Chỉ sử dụng features 100% future-safe:
  - lag_365 / lag_364 / lag_366 của Residual
  - rolling_4w_lag365 của Residual
  - prophet_trend (component từ Prophet)
  - Calendar: month, dayofweek, is_month_end, weekofyear, quarter
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging

logger = logging.getLogger("gridbreaker")

SEED = 42
FEAT_COLS = [
    "resid_lag365", "resid_lag364", "resid_lag366",
    "resid_roll28_lag365",
    "prophet_trend",
    "month", "dayofweek", "is_month_end", "weekofyear", "quarter", "dayofyear",
]


def make_future_safe_features(
    residuals: pd.Series,
    prophet_trend: pd.Series,
    index: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Tạo feature matrix 100% future-safe từ residuals và prophet trend.
    
    Args:
        residuals: Series chứa Residual (index=Date).
                   Có thể chứa NaN cho kỳ test.
        prophet_trend: Series chứa trend component từ Prophet.
        index: DatetimeIndex cuối cùng (nếu None, dùng residuals.index).
    """
    idx = index if index is not None else residuals.index
    X = pd.DataFrame(index=idx)

    # Lag features (từ residuals — lag 365 luôn có trong train data)
    for lag in [365, 364, 366]:
        X[f"resid_lag{lag}"] = residuals.shift(lag).reindex(idx)

    # Rolling mean 28 ngày ở lag 365
    X["resid_roll28_lag365"] = residuals.shift(365).rolling(28, min_periods=1).mean().reindex(idx)

    # Prophet trend
    X["prophet_trend"] = prophet_trend.reindex(idx)

    # Calendar features
    X["month"] = idx.month
    X["dayofweek"] = idx.dayofweek
    X["is_month_end"] = idx.is_month_end.astype(int)
    X["weekofyear"] = idx.isocalendar().week.astype(int).values
    X["quarter"] = idx.quarter
    X["dayofyear"] = idx.dayofyear

    return X


def fit_lgbm_residual(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_name: str = "Revenue",
) -> lgb.LGBMRegressor:
    """Huấn luyện LightGBM trên residuals."""
    logger.info(f"  Fitting LightGBM Residual cho {target_name}...")

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
    )

    # Drop NaN rows (lag_365 cần >= 1 năm data)
    mask = X_train[FEAT_COLS].notna().all(axis=1) & y_train.notna()
    X_clean = X_train.loc[mask, FEAT_COLS]
    y_clean = y_train.loc[mask]

    model.fit(X_clean, y_clean)
    logger.info(f"  LightGBM {target_name}: trained on {len(X_clean)} samples, {len(FEAT_COLS)} features.")

    # Log top features
    importance = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    logger.info(f"  Top 5 features: {dict(importance.head().items())}")

    return model


def predict_lgbm_residual(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
) -> pd.Series:
    """Dự đoán residual bằng LightGBM."""
    X_filled = X[FEAT_COLS].fillna(0)
    pred = model.predict(X_filled)
    return pd.Series(pred, index=X.index, name="lgbm_residual_pred")

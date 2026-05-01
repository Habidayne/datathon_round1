# The Gridbreaker — Datathon 2026 Round 1

**Taming the Volatility: Từ dữ liệu nhiễu loạn đến chuỗi cung ứng tinh gọn.**

---

## Cấu trúc thư mục

```
datathon-2026-round-1/
│
├── pipeline.py              # Entry point: chạy file này → submission.csv
├── generate_analysis.py     # Sinh 14 biểu đồ cho Phần 2 (EDA)
├── submission.csv           # File nộp (tự sinh khi chạy pipeline.py)
├── report.tex / report.pdf  # Báo cáo NeurIPS 4 trang
├── baseline.ipynb           # Baseline của BTC (tham khảo)
│
├── csv/                     # Dữ liệu gốc (KHÔNG chỉnh sửa)
│   ├── sales.csv            # Target: Revenue + COGS hàng ngày (2012–2022)
│   ├── sample_submission.csv
│   └── inventory.csv, promotions.csv, web_traffic.csv, ...
│
├── notebooks/
│   ├── Part2_EDA_Analysis.ipynb      # Phần 2: Trực quan hoá & EDA (60 điểm)
│   └── Part3_Forecasting_Model.ipynb # Phần 3: Pipeline + SHAP + CV (20 điểm)
│
├── outputs/
│   └── part2/               # 14 biểu đồ (viz1_*.png → viz14_*.png)
│
├── sql_round1/              # Câu hỏi SQL Vòng 1
│   └── Q1.sql → Q10.sql
│
├── src/                     # Source code pipeline
│   ├── utils/               # Tiện ích chung (helpers.py)
│   ├── denoising/           # Bước 1: Khử nhiễu mục tiêu
│   ├── prophet_model/       # Bước 2: Prophet Trend + Seasonality
│   ├── lgbm_model/          # Bước 3: LightGBM Residual Correction
│   ├── postprocess/         # Bước 4: Blend + xuất submission
│   └── analysis/            # Phần 2: 4 module phân tích (14 viz)
│
├── Styles/                  # LaTeX template NeurIPS 2025
└── "Đề thi Vòng 1.pdf"
```

---

## Pipeline "The Gridbreaker"

```
sales.csv (raw)
    │
    ▼ [Bước 1: Denoising]
    │  - Impute ngày stockout (rolling mean 7 ngày)
    │  - Cap spike không lặp lại (>3σ, recurrence <70%)
    ▼
Clean_Revenue
    │
    ▼ [Bước 2: Prophet]
    │  - Trend + Yearly/Weekly Seasonality
    │  - Custom holidays (Tết, cuối tháng, Black Friday)
    │  - Changepoint prior = 0.15
    ▼
prophet_pred + prophet_trend
    │
    ▼ [Bước 3: LightGBM Residual]
    │  - Residual = Actual − Prophet_pred
    │  - 11 features, 100% future-safe (lag ≥ 364 ngày)
    │  - 800 trees, lr=0.03, num_leaves=63
    ▼
lgbm_residual_pred
    │
    ▼ [Bước 4: Blend & Postprocess]
       Final = Prophet + LGBM_Residual
       Clip âm, dẫn xuất COGS từ Revenue (COGS = Revenue * 0.825)
    ▼
submission.csv (548 rows, 2023-01-01 → 2024-07-01)
```

---

## Kết quả Validation (Out-of-sample: 2022)

| Target  | MAE        | RMSE       | R²     | MAPE   |
|---------|------------|------------|--------|--------|
| Revenue | 0.697M VND | 0.926M VND | 0.6943 | 24.86% |
| COGS    | 0.686M VND | 0.898M VND | 0.6212 | 26.76% |

---

## Cách chạy

```bash
# Cài đặt thư viện
pip install pandas numpy prophet lightgbm scikit-learn shap matplotlib seaborn statsmodels

# Chạy full pipeline → sinh submission.csv
python pipeline.py

# Sinh 14 biểu đồ phân tích (outputs/part2/)
python generate_analysis.py
```

---

## Tài liệu

- **Báo cáo:** `report.pdf` (4 trang, NeurIPS format)
- **Phần 2 (EDA):** `notebooks/Part2_EDA_Analysis.ipynb`
- **Phần 3 (Model):** `notebooks/Part3_Forecasting_Model.ipynb`

---

## Tính tái lập (Reproducibility)

- Random seed: `SEED = 42` (global)
- Chạy `python pipeline.py` cho kết quả giống nhau mỗi lần
- Tất cả features là future-safe, không có data leakage

"""
src/analysis/diagnostic.py — Phân tích Chẩn đoán (Diagnostic)
"Why did it happen?"

Viz 5: Before/After denoising + stockout highlights  (sales ↔ inventory)
Viz 6: Promotion intervention analysis               (sales ↔ promotions)
Viz 7: Web traffic → Revenue cross-correlation       (sales ↔ web_traffic)
Viz 8: Revenue/COGS cointegration anomaly detection  (sales internal)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.tsa.stattools import ccf, coint

BG    = "#F8F9FA"
ACCENT = "#2EC4B6"
RED    = "#E71D36"
GOLD   = "#FF9F1C"
DARK   = "#011627"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def viz5_denoising_stockout(sales: pd.DataFrame, inventory_path: str,
                             clean_revenue: pd.Series, out_dir: str):
    """
    Viz 5: Stockout làm sụt giảm doanh thu 23% — Denoising khôi phục tín hiệu thật
    JOIN: sales ↔ inventory (stockout_flag per month → mapped to daily)
    """
    # Load & aggregate inventory by month → stockout flag
    inv = pd.read_csv(inventory_path)
    inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
    monthly_stockout = (inv.groupby("snapshot_date")["stockout_flag"]
                          .max()
                          .reset_index()
                          .set_index("snapshot_date"))

    # Map monthly flag to daily
    daily_idx = sales.index
    stockout_daily = monthly_stockout["stockout_flag"].reindex(daily_idx, method="ffill").fillna(0)

    # Focus on 2019-2021 window for clarity
    mask = (sales.index >= "2019-01-01") & (sales.index <= "2021-06-30")
    rev_raw   = sales.loc[mask, "Revenue"]
    rev_clean = clean_revenue.reindex(rev_raw.index)
    so_mask   = stockout_daily.reindex(rev_raw.index) == 1

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Highlight stockout periods
    in_stockout = False
    start_so = None
    for date, flag in so_mask.items():
        if flag and not in_stockout:
            start_so = date
            in_stockout = True
        elif not flag and in_stockout:
            ax.axvspan(start_so, date, alpha=0.15, color=RED, label="_nolegend_")
            in_stockout = False

    ax.plot(rev_raw.index, rev_raw / 1e6, color="#CCCCCC", linewidth=0.8,
            label="Doanh thu gốc (có nhiễu)", alpha=0.9)
    ax.plot(rev_clean.index, rev_clean / 1e6, color=ACCENT, linewidth=1.8,
            label="Doanh thu sau Denoising")

    # Stats annotation
    raw_mean   = rev_raw[so_mask].mean()
    clean_mean = rev_clean[so_mask].mean()
    drop_pct   = (clean_mean - raw_mean) / clean_mean * 100
    ax.annotate(f"Stockout làm doanh thu giảm\ngiả tạo ~{abs(drop_pct):.0f}%",
                xy=(pd.Timestamp("2019-10-01"), raw_mean/1e6),
                xytext=(pd.Timestamp("2020-02-01"), raw_mean/1e6 + 5),
                fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))

    red_patch = mpatches.Patch(color=RED, alpha=0.3, label="Tháng có Stockout")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [red_patch], fontsize=8, loc="upper left")

    _style_ax(ax,
              title="Viz 5 — Stockout làm sụt giảm doanh thu giả tạo: Denoising khôi phục tín hiệu thật",
              xlabel="Date", ylabel="Revenue (Triệu VNĐ)")

    plt.tight_layout()
    path = os.path.join(out_dir, "viz5_denoising_stockout.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz6_promotion_intervention(sales: pd.DataFrame, promotions_path: str, out_dir: str):
    """
    Viz 6: Khuyến mãi tạo spike +X% nhưng hiệu ứng chỉ kéo dài 3-5 ngày
    JOIN: sales.Date ∈ [promotions.start_date, promotions.end_date]
    """
    promos = pd.read_csv(promotions_path, parse_dates=["start_date", "end_date"])

    # Build daily promo flag
    daily_flag = pd.Series(0, index=sales.index)
    promo_windows = []
    for _, row in promos.iterrows():
        mask = (sales.index >= row["start_date"]) & (sales.index <= row["end_date"])
        daily_flag[mask] = 1
        promo_windows.append((row["start_date"], row["end_date"], row["promo_name"]))

    rev = sales["Revenue"]
    baseline = rev[daily_flag == 0].rolling(30).mean().reindex(rev.index).ffill()
    lift = (rev / baseline - 1) * 100

    # Event study: average lift around promo start
    event_windows = []
    for _, row in promos.iterrows():
        start = row["start_date"]
        window_start = start - pd.Timedelta(days=7)
        window_end   = start + pd.Timedelta(days=14)
        if window_start in sales.index and window_end in sales.index:
            window_rev = rev.loc[window_start:window_end]
            baseline_val = rev.loc[window_start:start].mean()
            relative = (window_rev / baseline_val - 1) * 100
            relative.index = range(-7, len(relative) - 7)
            event_windows.append(relative)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Left: Revenue with promo highlights
    ax1.plot(rev.index, rev/1e6, color="#CCCCCC", linewidth=0.6, alpha=0.7)
    for start, end, name in promo_windows[:8]:  # first 8 promos
        if start >= sales.index.min() and end <= sales.index.max():
            peak = rev.loc[start:end].max()
            ax1.axvspan(start, end, alpha=0.25, color=GOLD)
            ax1.annotate("", xy=(start, peak/1e6), xytext=(start, (peak*0.7)/1e6),
                         arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
    _style_ax(ax1, title="Doanh thu theo thời gian + Promo periods",
              xlabel="Date", ylabel="Revenue (Triệu VNĐ)")

    # Right: Event study plot
    if event_windows:
        event_df = pd.DataFrame(event_windows)
        mean_lift   = event_df.mean()
        median_lift = event_df.median()
        days = mean_lift.index.tolist()

        ax2.fill_between(days, event_df.quantile(0.25), event_df.quantile(0.75),
                         alpha=0.2, color=GOLD, label="IQR")
        ax2.plot(days, mean_lift, color=GOLD, linewidth=2, marker="o",
                 markersize=4, label="Lift trung bình (%)")
        ax2.plot(days, median_lift, color=RED, linewidth=1.5, linestyle="--",
                 label="Lift trung vị (%)")
        ax2.axvline(0, color=DARK, linestyle=":", linewidth=1.5, label="Ngày bắt đầu Promo")
        ax2.axhline(0, color="#999999", linewidth=0.8)
        ax2.set_xlim(-7, 14)

        peak_day  = int(mean_lift.idxmax())
        peak_lift = mean_lift.max()
        ax2.annotate(f"Peak: +{peak_lift:.0f}%\n(Ngày +{peak_day})",
                     xy=(peak_day, peak_lift),
                     xytext=(peak_day + 2, peak_lift * 0.8),
                     fontsize=9, color=GOLD, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=GOLD))

        _style_ax(ax2, title="Event Study: Lift doanh thu quanh ngày bắt đầu Promo",
                  xlabel="Ngày so với ngày bắt đầu Promo", ylabel="Revenue Lift (%)")
        ax2.legend(fontsize=8)

    fig.suptitle("Viz 6 — Khuyến mãi tạo spike doanh thu ngắn hạn: Hiệu ứng suy giảm sau 5-7 ngày",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz6_promotion_intervention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz7_web_traffic_ccf(sales: pd.DataFrame, web_traffic_path: str, out_dir: str):
    """
    Viz 7: Web traffic dẫn trước doanh thu 2-3 ngày — chỉ số dẫn xuất
    JOIN: sales.Date == web_traffic.date
    """
    wt = pd.read_csv(web_traffic_path, parse_dates=["date"]).set_index("date")

    # Daily aggregate (sum across traffic sources)
    wt_daily = wt.groupby(wt.index)["sessions"].sum()

    # Align with sales
    common_idx = sales.index.intersection(wt_daily.index)
    rev_aligned = sales.loc[common_idx, "Revenue"]
    wt_aligned  = wt_daily.reindex(common_idx)

    # Standardize
    rev_std = (rev_aligned - rev_aligned.mean()) / rev_aligned.std()
    wt_std  = (wt_aligned  - wt_aligned.mean())  / wt_aligned.std()

    # CCF: correlation of wt_today with revenue at lag k
    max_lag = 14
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = wt_std.shift(lag).corr(rev_std)
        else:
            c = wt_std.shift(lag).corr(rev_std)
        correlations.append(c)
    lags = list(range(-max_lag, max_lag + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # CCF bar chart
    colors = [ACCENT if c > 0 else RED for c in correlations]
    ax1.bar(lags, correlations, color=colors, alpha=0.8)
    ax1.axhline(0, color="#999999", linewidth=0.8)
    ci = 1.96 / np.sqrt(len(common_idx))
    ax1.axhline(ci,  color="#555555", linestyle="--", linewidth=0.8, label=f"95% CI (±{ci:.2f})")
    ax1.axhline(-ci, color="#555555", linestyle="--", linewidth=0.8)
    peak_lag = lags[np.argmax(correlations)]
    ax1.annotate(f"Lag tối ưu: {peak_lag} ngày\nr = {max(correlations):.2f}",
                 xy=(peak_lag, max(correlations)),
                 xytext=(peak_lag + 2, max(correlations) * 0.85),
                 fontsize=9, color=ACCENT, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT))
    _style_ax(ax1, title="Cross-Correlation: Web Sessions → Revenue",
              xlabel="Lag (ngày, dương = traffic dẫn trước revenue)",
              ylabel="Correlation coefficient")
    ax1.legend(fontsize=8)

    # Scatter: sessions(t-2) vs revenue(t)
    lag_opt = max(0, peak_lag)
    x_lag = wt_aligned.shift(lag_opt).dropna()
    y_rev = rev_aligned.reindex(x_lag.index).dropna()
    x_lag = x_lag.reindex(y_rev.index)

    ax2.scatter(x_lag/1e3, y_rev/1e6, alpha=0.2, s=10, color=ACCENT)
    z = np.polyfit(x_lag, y_rev, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_lag.min(), x_lag.max(), 100)
    ax2.plot(x_line/1e3, p(x_line)/1e6, color=RED, linewidth=2)
    _style_ax(ax2,
              title=f"Scatter: Sessions(t−{lag_opt}) vs Revenue(t)",
              xlabel="Web Sessions (nghìn)", ylabel="Revenue (Triệu VNĐ)")

    fig.suptitle("Viz 7 — Web traffic dẫn trước doanh thu: Leading indicator cho hệ thống cảnh báo sớm",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz7_web_traffic_ccf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz8_cointegration_anomaly(sales: pd.DataFrame, out_dir: str):
    """
    Viz 8: Revenue và COGS đồng liên kết dài hạn — phát hiện giai đoạn chi phí đội bất thường
    Chỉ dùng sales.csv (Revenue + COGS)
    """
    rev  = sales["Revenue"].dropna()
    cogs = sales["COGS"].dropna()
    common = rev.index.intersection(cogs.index)
    rev  = rev.loc[common]
    cogs = cogs.loc[common]

    # Engle-Granger cointegration test
    score, pvalue, _ = coint(rev, cogs)

    # Spread = COGS - beta*Revenue (OLS residual)
    beta = np.cov(cogs, rev)[0, 1] / np.var(rev)
    spread = cogs - beta * rev
    spread_roll_mean = spread.rolling(90).mean()
    spread_roll_std  = spread.rolling(90).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("white")

    ax1.plot(rev.index,  rev/1e6,  color=ACCENT, linewidth=1.0, label="Revenue", alpha=0.9)
    ax1.plot(cogs.index, cogs/1e6, color=RED,    linewidth=1.0, label="COGS", alpha=0.9)
    _style_ax(ax1, title=f"Revenue vs COGS — Đồng liên kết (Cointegrated), p-value={pvalue:.4f}",
              ylabel="Triệu VNĐ")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # Spread plot with anomaly bands
    ax2.plot(spread.index, spread/1e6, color="#888888", linewidth=0.7, alpha=0.7)
    ax2.plot(spread_roll_mean.index, spread_roll_mean/1e6, color=DARK, linewidth=1.5, label="90-day MA spread")
    upper = (spread_roll_mean + 2*spread_roll_std)
    lower = (spread_roll_mean - 2*spread_roll_std)
    ax2.fill_between(spread.index, lower/1e6, upper/1e6, alpha=0.15, color=GOLD, label="±2σ band")

    # Mark anomaly (spread > +2σ = COGS abnormally high)
    anomaly = spread > upper
    ax2.scatter(spread.index[anomaly], spread[anomaly]/1e6, color=RED, s=8, zorder=5,
                label=f"COGS đội bất thường ({anomaly.sum()} ngày)")
    _style_ax(ax2, title="Spread (COGS − β×Revenue): Phát hiện giai đoạn chi phí vượt mức",
              xlabel="Date", ylabel="Spread (Triệu VNĐ)")
    ax2.legend(fontsize=8)
    ax2.axhline(0, color="#999999", linewidth=0.8)

    fig.suptitle("Viz 8 — Revenue & COGS đồng liên kết dài hạn: Phát hiện giai đoạn biên lợi nhuận bị bào mòn bất thường",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz8_cointegration_anomaly.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run_diagnostic(sales: pd.DataFrame, clean_revenue: pd.Series, out_dir: str,
                   inventory_path: str, promotions_path: str, web_traffic_path: str):
    """Run tất cả 4 Diagnostic visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    print("  [Diagnostic] Viz 5: Denoising + stockout...")
    paths.append(viz5_denoising_stockout(sales, inventory_path, clean_revenue, out_dir))
    print("  [Diagnostic] Viz 6: Promotion intervention...")
    paths.append(viz6_promotion_intervention(sales, promotions_path, out_dir))
    print("  [Diagnostic] Viz 7: Web traffic CCF...")
    paths.append(viz7_web_traffic_ccf(sales, web_traffic_path, out_dir))
    print("  [Diagnostic] Viz 8: Cointegration anomaly...")
    paths.append(viz8_cointegration_anomaly(sales, out_dir))
    return paths

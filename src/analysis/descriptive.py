"""
src/analysis/descriptive.py — Phân tích Mô tả (Descriptive)
"What happened?"

Viz 1: Revenue time series + rolling volatility band
Viz 2: Monthly profit margin heatmap (year × month)
Viz 3: STL Decomposition 3-panel
Viz 4: Revenue by product category (top categories)
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
import seaborn as sns
from statsmodels.tsa.seasonal import STL

# ── Style ─────────────────────────────────────────────────
PALETTE = ["#2EC4B6", "#E71D36", "#FF9F1C", "#011627", "#FDFFFC"]
ACCENT  = "#2EC4B6"
RED     = "#E71D36"
GOLD    = "#FF9F1C"
BG      = "#F8F9FA"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def viz1_revenue_trend(sales: pd.DataFrame, out_dir: str):
    """
    Viz 1: Doanh thu tăng trưởng 2.8× trong 10 năm nhưng biến động gia tăng cùng tốc độ
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("white")

    rev = sales["Revenue"].copy()
    roll_mean = rev.rolling(90).mean()
    roll_std  = rev.rolling(90).std()

    ax1.fill_between(rev.index, roll_mean - 2*roll_std, roll_mean + 2*roll_std,
                     alpha=0.15, color=ACCENT, label="±2σ band (90-day)")
    ax1.plot(rev.index, rev, color="#CCCCCC", linewidth=0.6, alpha=0.7)
    ax1.plot(roll_mean.index, roll_mean, color=ACCENT, linewidth=2, label="90-day MA")

    # Annotate YoY growth
    for year in [2015, 2018, 2020, 2022]:
        y_val = rev.loc[str(year)].mean()
        ax1.annotate(f"{year}", xy=(pd.Timestamp(f"{year}-07-01"), y_val),
                     fontsize=7, color="#555555")

    _style_ax(ax1,
              title="Viz 1 — Doanh thu tăng trưởng 2.8× (2012–2022) nhưng biến động ngày càng lớn",
              ylabel="Revenue (VNĐ)")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    # Bottom: rolling volatility
    ax2.fill_between(roll_std.index, 0, roll_std, color=RED, alpha=0.6)
    _style_ax(ax2, ylabel="Rolling Std", xlabel="Date")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    plt.tight_layout()
    path = os.path.join(out_dir, "viz1_revenue_trend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz2_profit_margin_heatmap(sales: pd.DataFrame, out_dir: str):
    """
    Viz 2: Biên lợi nhuận gộp dao động theo mùa — phát hiện tháng 8 margin âm bất thường
    """
    df = sales.copy()
    df["margin"] = (df["Revenue"] - df["COGS"]) / df["Revenue"] * 100
    df["year"]   = df.index.year
    df["month"]  = df.index.month

    pivot = df.groupby(["year", "month"])["margin"].mean().unstack()
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    # Use explicit vmin/vmax to handle extreme negative margins (Aug = -42%)
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=-45, vmax=25, center=0, linewidths=0.5,
                annot_kws={"size": 9, "fontweight": "bold"}, ax=ax,
                cbar_kws={"label": "Gross Margin %", "shrink": 0.8})

    ax.set_title("Viz 2 — Biên lợi nhuận gộp (%): Tháng 8 có margin âm bất thường, Q1 cao nhất",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tháng", fontsize=10)
    ax.set_ylabel("Năm", fontsize=10)
    ax.tick_params(labelsize=9)

    # Highlight anomalous cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val) and val < 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor=RED, linewidth=2.5))

    plt.tight_layout()
    path = os.path.join(out_dir, "viz2_profit_margin_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz3_stl_decomposition(sales: pd.DataFrame, out_dir: str):
    """
    Viz 3: Phân rã STL — Mùa vụ chiếm ~35% biến động
    """
    rev_daily = sales["Revenue"].resample("D").mean().interpolate()
    stl = STL(rev_daily, period=365, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Viz 3 — STL Decomposition: Mùa vụ + Xu hướng chiếm 65% tổng biến động của doanh thu",
                 fontsize=12, fontweight="bold", y=1.01)

    components = [
        (rev_daily,     "Observed (Doanh thu gốc)",   ACCENT),
        (res.trend,     "Trend (Xu hướng dài hạn)",   "#011627"),
        (res.seasonal,  "Seasonal (Mùa vụ hàng năm)", GOLD),
        (res.resid,     "Residual (Nhiễu / Bất thường)", RED),
    ]
    for ax, (data, label, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    axes[-1].set_xlabel("Date", fontsize=9)

    # Annotate variance share
    var_seasonal = float(np.var(res.seasonal))
    var_total    = float(np.var(rev_daily))
    pct = var_seasonal / var_total * 100
    axes[2].annotate(f"Seasonal variance: {pct:.0f}% of total",
                     xy=(rev_daily.index[100], float(res.seasonal.iloc[100])),
                     fontsize=8, color=GOLD, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "viz3_stl_decomposition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz4_revenue_by_category(order_items_path: str, products_path: str,
                              orders_path: str, out_dir: str):
    """
    Viz 4: Top categories chiếm 72% doanh thu — JOIN order_items + products + orders
    """
    oi = pd.read_csv(order_items_path)
    pr = pd.read_csv(products_path)[["product_id", "category", "cogs"]]
    od = pd.read_csv(orders_path)[["order_id", "order_date"]]
    od["order_date"] = pd.to_datetime(od["order_date"])
    od["year"] = od["order_date"].dt.year

    merged = oi.merge(pr, on="product_id").merge(od, on="order_id")
    merged["line_revenue"] = merged["quantity"] * merged["unit_price"] - merged["discount_amount"]
    merged["line_cogs"]    = merged["quantity"] * merged["cogs"]

    cat_rev = merged.groupby("category")["line_revenue"].sum().sort_values(ascending=False)
    total   = cat_rev.sum()
    cumpct  = cat_rev.cumsum() / total * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")

    # Bar chart
    colors = [ACCENT if i < 3 else "#AAAAAA" for i in range(len(cat_rev))]
    bars = ax1.barh(cat_rev.index[::-1], cat_rev.values[::-1] / 1e9, color=colors[::-1])
    _style_ax(ax1,
              title="Doanh thu theo Category (2012–2022)",
              xlabel="Revenue (Tỷ VNĐ)", ylabel="")
    for bar, val in zip(bars, cat_rev.values[::-1]):
        pct = val / total * 100
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{pct:.1f}%", va="center", fontsize=8)

    # Yearly trend by top 3 categories
    top3 = cat_rev.index[:3].tolist()
    yearly = merged[merged["category"].isin(top3)].groupby(["year", "category"])["line_revenue"].sum().unstack()
    for cat in top3:
        if cat in yearly.columns:
            ax2.plot(yearly.index, yearly[cat]/1e9, marker="o", markersize=4, label=cat)

    _style_ax(ax2, title="Xu hướng Top 3 Category theo năm",
              xlabel="Năm", ylabel="Revenue (Tỷ VNĐ)")
    ax2.legend(fontsize=8)

    fig.suptitle("Viz 4 — Top categories chiếm >70% doanh thu: Tập trung nguồn lực vào phân khúc dẫn đầu",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz4_revenue_by_category.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run_descriptive(sales: pd.DataFrame, out_dir: str,
                    order_items_path: str, products_path: str, orders_path: str):
    """Run tất cả 4 Descriptive visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    print("  [Descriptive] Viz 1: Revenue trend...")
    paths.append(viz1_revenue_trend(sales, out_dir))
    print("  [Descriptive] Viz 2: Profit margin heatmap...")
    paths.append(viz2_profit_margin_heatmap(sales, out_dir))
    print("  [Descriptive] Viz 3: STL decomposition...")
    paths.append(viz3_stl_decomposition(sales, out_dir))
    print("  [Descriptive] Viz 4: Revenue by category...")
    paths.append(viz4_revenue_by_category(order_items_path, products_path, orders_path, out_dir))
    return paths

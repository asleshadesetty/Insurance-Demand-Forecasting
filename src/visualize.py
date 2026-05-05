"""
visualize.py
------------
Produces four publication-quality charts replicating the
Tableau Public dashboard:

  1. Demand Forecast Over Time       — per-category forecast with CI band
  2. Actual vs Predicted Comparison  — scatter + line of best fit
  3. Resource Utilization Breakdown  — stacked area chart over time
  4. Medication Category Breakdown   — heatmap of weekly demand by category

All charts saved to visuals/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = ["#1F3864", "#2E75B6", "#70AD47", "#ED7D31", "#FFC000"]
BG_COLOR  = "#F7F9FC"
GRID_CLR  = "#E0E6EF"
FONT      = "DejaVu Sans"

plt.rcParams.update({
    "font.family":        FONT,
    "axes.facecolor":     BG_COLOR,
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.color":         GRID_CLR,
    "grid.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelcolor":    "#333333",
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
})

OUTPUT_DIR = "visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
def load():
    forecasts = pd.read_csv("outputs/forecasts.csv", parse_dates=["ds"])
    metrics   = pd.read_csv("outputs/metrics.csv")
    return forecasts, metrics


# ── Chart 1: Demand Forecast Over Time ───────────────────────────────────────
def plot_forecast_over_time(forecasts):
    categories = sorted(forecasts["category"].unique())
    fig, axes  = plt.subplots(3, 2, figsize=(16, 14), sharex=False)
    axes       = axes.flatten()

    for idx, (cat, color) in enumerate(zip(categories, PALETTE)):
        ax  = axes[idx]
        sub = forecasts[forecasts["category"] == cat].sort_values("ds")

        hist   = sub[sub["actual"].notna()]
        future = sub[sub["actual"].isna()]

        ax.fill_between(sub["ds"], sub["yhat_lower"], sub["yhat_upper"],
                        alpha=0.18, color=color, label="95% CI")
        ax.plot(sub["ds"], sub["yhat"], color=color, lw=2.0,
                label="Forecast", zorder=3)
        ax.plot(hist["ds"], hist["actual"], color="#333333", lw=1.2,
                alpha=0.75, label="Actual", zorder=4)
        if not future.empty:
            ax.axvline(future["ds"].min(), color="#888888",
                       ls="--", lw=1, alpha=0.6, label="Forecast start")

        ax.set_title(cat, fontsize=13, fontweight="bold", color="#1F3864", pad=8)
        ax.set_ylabel("Weekly Demand (units)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=8, loc="upper left")

    # Hide unused subplot
    if len(categories) < len(axes):
        for i in range(len(categories), len(axes)):
            axes[i].set_visible(False)

    fig.suptitle("Insurance Medication & Resource Demand Forecast",
                 fontsize=17, fontweight="bold", color="#1F3864", y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/1_forecast_over_time.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Chart 2: Actual vs Predicted ─────────────────────────────────────────────
def plot_actual_vs_predicted(forecasts):
    df = forecasts[forecasts["actual"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    for cat, color in zip(sorted(df["category"].unique()), PALETTE):
        sub = df[df["category"] == cat]
        ax.scatter(sub["actual"], sub["yhat"], alpha=0.55, s=25,
                   color=color, label=cat, zorder=3)

    # Perfect prediction line
    lims = [
        min(df["actual"].min(), df["yhat"].min()) * 0.97,
        max(df["actual"].max(), df["yhat"].max()) * 1.03,
    ]
    ax.plot(lims, lims, "k--", lw=1.4, alpha=0.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)

    # Line of best fit
    m, b  = np.polyfit(df["actual"], df["yhat"], 1)
    x_fit = np.linspace(lims[0], lims[1], 200)
    ax.plot(x_fit, m * x_fit + b, color="#ED7D31", lw=1.8,
            alpha=0.8, label=f"Best fit  (slope={m:.2f})")

    ax.set_xlabel("Actual Demand (units)", fontsize=12)
    ax.set_ylabel("Predicted Demand (units)", fontsize=12)
    ax.set_title("Actual vs Predicted Demand — All Categories",
                 fontsize=14, fontweight="bold", color="#1F3864")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/2_actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Chart 3: Resource Utilization Breakdown ───────────────────────────────────
def plot_resource_utilization(forecasts):
    df = (
        forecasts[["ds", "category", "yhat"]]
        .copy()
        .sort_values("ds")
    )
    pivot = df.pivot_table(index="ds", columns="category",
                           values="yhat", aggfunc="sum").fillna(0)
    pivot.columns.name = None

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot.area(ax=ax, color=PALETTE, alpha=0.82, linewidth=0)

    ax.set_title("Weekly Resource Utilization by Category (Forecast)",
                 fontsize=14, fontweight="bold", color="#1F3864")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Total Weekly Demand (units)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=9)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/3_resource_utilization.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Chart 4: Medication Category Heatmap ─────────────────────────────────────
def plot_category_heatmap(forecasts):
    df = forecasts[forecasts["actual"].notna()].copy()
    df["month_year"] = df["ds"].dt.to_period("M").astype(str)

    pivot = df.pivot_table(index="category", columns="month_year",
                           values="actual", aggfunc="sum")

    # Keep last 24 months for readability
    pivot = pivot.iloc[:, -24:]

    fig, ax = plt.subplots(figsize=(18, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="#CCCCCC",
        annot=False,
        fmt=".0f",
        cbar_kws={"label": "Monthly Demand (units)", "shrink": 0.8},
    )
    ax.set_title("Medication Category Demand Heatmap (Monthly)",
                 fontsize=14, fontweight="bold", color="#1F3864")
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       ha="right", fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/4_category_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    forecasts, metrics = load()
    print("\nGenerating visualizations...\n")
    plot_forecast_over_time(forecasts)
    plot_actual_vs_predicted(forecasts)
    plot_resource_utilization(forecasts)
    plot_category_heatmap(forecasts)
    print("\nAll charts saved to visuals/")

"""
Generate four benchmark plots from the results CSV.

Plot 1 — Accuracy vs. model size (param count proxy)
Plot 2 — Energy / cost per query
Plot 3 — Cumulative CO2 over one year at varying daily-usage levels,
          local-vs-cloud crossover highlighted
Plot 4 — Latency vs. accuracy scatter

All outputs saved as PNG to plots/.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
PLOTS_DIR = REPO_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------

LOCAL_COLOR = "#2196F3"   # blue
CLOUD_COLOR = "#F44336"   # red
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]


def _model_label(name: str) -> str:
    return name.replace(" [mock]", "\n(mock)")


# --------------------------------------------------------------------------
# Shared summary builder
# --------------------------------------------------------------------------

def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-example rows to per-model summary stats."""
    agg = (
        df.groupby(["model", "model_type", "model_params_b"])
        .agg(
            accuracy=("score_overall", "mean"),
            parse_rate=("parse_ok", "mean"),
            latency_mean=("latency_s", "mean"),
            latency_p95=("latency_s", lambda x: x.quantile(0.95)),
            co2_kg_per_query=("co2_kg", "mean"),
            cost_usd_per_query=("cost_usd", "mean"),
            n=("score_overall", "count"),
        )
        .reset_index()
        .sort_values("model_params_b")
    )
    return agg


# --------------------------------------------------------------------------
# Plot 1: Accuracy vs. model size
# --------------------------------------------------------------------------

def plot_accuracy_vs_size(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    summary = _summarise(df)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, row in enumerate(summary.itertuples()):
        color = CLOUD_COLOR if row.model_type == "cloud" else LOCAL_COLOR
        marker = "★" if row.model_type == "cloud" else "o"
        ax.scatter(
            row.model_params_b,
            row.accuracy,
            s=180,
            color=color,
            zorder=3,
            label=f"{_model_label(row.model)} ({row.accuracy:.2f})",
        )
        ax.annotate(
            _model_label(row.model),
            xy=(row.model_params_b, row.accuracy),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Model size (B parameters, log scale)", fontsize=11)
    ax.set_ylabel("Mean accuracy (weighted field score)", fontsize=11)
    ax.set_title("Accuracy vs. Model Size", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.4)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LOCAL_COLOR, markersize=10, label="Local (Ollama)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLOUD_COLOR, markersize=10, label="Cloud (Gemini)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = out_dir / "plot1_accuracy_vs_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Plot 2: Energy / cost per query
# --------------------------------------------------------------------------

def plot_energy_cost(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    summary = _summarise(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = [_model_label(m) for m in summary["model"]]
    colors = [CLOUD_COLOR if t == "cloud" else LOCAL_COLOR for t in summary["model_type"]]
    x = np.arange(len(models))

    # Left: CO2 per query (g)
    ax = axes[0]
    co2_g = summary["co2_kg_per_query"] * 1000
    bars = ax.bar(x, co2_g, color=colors, width=0.6, zorder=2)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("g CO₂eq per query (estimated)", fontsize=10)
    ax.set_title("Carbon per Query", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.annotate(
        "⚠ Estimates only — see README",
        xy=(0.5, 0.97), xycoords="axes fraction",
        ha="center", va="top", fontsize=7, color="grey",
    )

    # Right: Cost per query (USD)
    ax2 = axes[1]
    cost_m = summary["cost_usd_per_query"] * 1000  # milli-dollars
    bars2 = ax2.bar(x, cost_m, color=colors, width=0.6, zorder=2)
    ax2.bar_label(bars2, fmt="%.4f", fontsize=8, padding=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("Cost (milli-USD per query)", fontsize=10)
    ax2.set_title("Monetary Cost per Query", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.annotate(
        "Local = $0 marginal (electricity excluded)",
        xy=(0.5, 0.97), xycoords="axes fraction",
        ha="center", va="top", fontsize=7, color="grey",
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=LOCAL_COLOR, label="Local (Ollama)"),
        Patch(facecolor=CLOUD_COLOR, label="Cloud (Gemini)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    out = out_dir / "plot2_energy_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Plot 3: Cumulative CO2 over one year — local vs cloud crossover
# --------------------------------------------------------------------------

def plot_cumulative_co2(
    df: pd.DataFrame,
    daily_usage_levels: list[int] | None = None,
    out_dir: Path = PLOTS_DIR,
) -> Path:
    """
    Cumulative annual CO2 at several daily query rates.

    - If both local and cloud models are present: one line per model type per
      daily-usage level, with a crossover marker where they intersect.
    - If only local models are present: one line per model at 20 queries/day,
      showing relative carbon cost across model sizes.
    """
    if daily_usage_levels is None:
        daily_usage_levels = [5, 20, 50, 100]

    summary = _summarise(df)
    local_rows = summary[summary["model_type"] == "local"].sort_values("model_params_b")
    cloud_rows = summary[summary["model_type"] == "cloud"]

    days = np.arange(1, 366)
    fig, ax = plt.subplots(figsize=(10, 6))
    line_styles = ["-", "--", "-.", ":"]

    has_cloud = not cloud_rows.empty
    has_local = not local_rows.empty

    if not has_local:
        print("  [plot3] No local model rows — skipping.")
        plt.close(fig)
        return out_dir / "plot3_cumulative_co2.png"

    if has_cloud:
        # Local vs cloud comparison across daily usage levels
        best_local = local_rows.sort_values("accuracy", ascending=False).iloc[0]
        cloud = cloud_rows.iloc[0]
        crossover_noted = False

        for i, daily in enumerate(daily_usage_levels):
            ls = line_styles[i % len(line_styles)]
            local_co2 = days * daily * best_local["co2_kg_per_query"]
            cloud_co2 = days * daily * cloud["co2_kg_per_query"]
            ax.plot(days, local_co2, color=LOCAL_COLOR, linestyle=ls, linewidth=2,
                    label=f"Local — {_model_label(best_local['model'])} ({daily}/day)")
            ax.plot(days, cloud_co2, color=CLOUD_COLOR, linestyle=ls, linewidth=2,
                    label=f"Cloud — Gemini ({daily}/day)")

            diff = local_co2 - cloud_co2
            if np.any(diff > 0) and not crossover_noted:
                idx = np.where(diff > 0)[0][0]
                ax.axvline(days[idx], color="grey", linestyle=":", linewidth=1.5)
                ax.annotate(
                    f"Crossover\n(day {days[idx]})",
                    xy=(days[idx], local_co2[idx]),
                    xytext=(10, 10), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="grey"),
                )
                crossover_noted = True

        title = (
            f"Cumulative CO₂: Local ({_model_label(best_local['model'])}) vs Cloud (Gemini)\n"
            "Per-query CO₂ from harness measurements · 365 days/year"
        )
    else:
        # Local-only: one line per model at a representative usage level
        daily = 20
        colors_local = PALETTE[: len(local_rows)]
        for i, row in enumerate(local_rows.itertuples()):
            annual_co2 = days * daily * row.co2_kg_per_query
            ax.plot(days, annual_co2, color=colors_local[i], linewidth=2,
                    label=f"{_model_label(row.model)} ({row.model_params_b:.0f}B params)")
        ax.set_title(  # placeholder — set below
            "", fontsize=11)
        title = (
            f"Cumulative CO₂ — Local Models at {daily} queries/day\n"
            "Per-query CO₂ estimated via TDP × latency · 365 days/year\n"
            "Run with --cloud to add Gemini comparison"
        )

    ax.set_xlabel("Day of year", fontsize=11)
    ax.set_ylabel("Cumulative CO₂eq (kg, estimated)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.annotate(
        "⚠ CO₂ figures are ballpark estimates — not lab measurements. See README.",
        xy=(0.5, -0.13), xycoords="axes fraction",
        ha="center", fontsize=8, color="grey",
    )
    fig.tight_layout()
    out = out_dir / "plot3_cumulative_co2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Plot 4: Latency vs. accuracy
# --------------------------------------------------------------------------

def plot_latency_vs_accuracy(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    summary = _summarise(df)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, row in enumerate(summary.itertuples()):
        color = CLOUD_COLOR if row.model_type == "cloud" else LOCAL_COLOR
        ax.scatter(
            row.latency_mean,
            row.accuracy,
            s=200,
            color=color,
            zorder=3,
            alpha=0.85,
        )
        # Error bar for p95 latency
        ax.errorbar(
            row.latency_mean,
            row.accuracy,
            xerr=[[0], [row.latency_p95 - row.latency_mean]],
            fmt="none",
            color=color,
            alpha=0.4,
            capsize=4,
        )
        ax.annotate(
            _model_label(row.model),
            xy=(row.latency_mean, row.accuracy),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean latency per query (s)", fontsize=11)
    ax.set_ylabel("Mean accuracy (weighted field score)", fontsize=11)
    ax.set_title(
        "Latency vs. Accuracy\n(error bar = p95 latency; lower-left is best)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.4)

    # "Ideal" annotation
    ax.annotate(
        "← faster, more accurate",
        xy=(0.02, 0.97), xycoords="axes fraction",
        ha="left", va="top", fontsize=8, color="grey",
        arrowprops=None,
    )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LOCAL_COLOR, markersize=10, label="Local (Ollama)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLOUD_COLOR, markersize=10, label="Cloud (Gemini)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = out_dir / "plot4_latency_vs_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Convenience: generate all four plots
# --------------------------------------------------------------------------

def generate_all(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> list[Path]:
    paths = [
        plot_accuracy_vs_size(df, out_dir),
        plot_energy_cost(df, out_dir),
        plot_cumulative_co2(df, out_dir=out_dir),
        plot_latency_vs_accuracy(df, out_dir),
    ]
    for p in paths:
        print(f"  Saved: {p}")
    return paths
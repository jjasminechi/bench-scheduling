"""
Simple benchmark plots for the edge-LLM tradeoff study.

Plot 1 — Accuracy by model (grouped bar: per-benchmark + average)
Plot 2 — Carbon emissions per query (bar chart, local = CodeCarbon, cloud = EcoLogits)
Plot 3 — Latency per query (bar chart with p90 error bar)
Plot 4 — Accuracy vs. latency scatter (all models labelled)

All outputs saved as PNG to plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent
PLOTS_DIR = REPO_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

LOCAL_COLOR = "#2196F3"   # blue
CLOUD_COLOR = "#F44336"   # red
BM_COLORS   = {"hellaswag": "#4CAF50", "piqa": "#9C27B0"}
BENCHMARKS  = ["hellaswag", "piqa"]


def _label(name: str) -> str:
    return name.replace("gemini/", "").replace(" [mock]", " (mock)")


# ---------------------------------------------------------------------------
# Shared aggregation
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary stats, including per-benchmark accuracy columns."""
    if "benchmark" in df.columns:
        bm_acc = (
            df.groupby(["model", "benchmark"])["correct"]
            .mean()
            .unstack("benchmark", fill_value=float("nan"))
            .rename(columns=lambda c: f"acc_{c}")
        )
    else:
        bm_acc = pd.DataFrame()

    agg = (
        df.groupby(["model", "model_type", "model_params_b"], dropna=False)
        .agg(
            accuracy          = ("correct",    "mean"),
            latency_median    = ("latency_s",  "median"),
            latency_p90       = ("latency_s",  lambda x: x.quantile(0.90)),
            emissions_g_per_q = ("emissions_g","mean"),
            energy_j_per_q    = ("energy_j",   "mean"),
            energy_kwh_per_q  = ("energy_kwh", "mean"),
            cost_usd_per_q    = ("cost_usd",   "mean"),
            is_mock           = ("is_mock",     "any"),
            n                 = ("correct",     "count"),
        )
        .reset_index()
    )
    agg["cost_per_1k"] = agg["cost_usd_per_q"] * 1000

    if not bm_acc.empty:
        agg = agg.join(bm_acc, on="model")

    # Sort: local models by size, then cloud
    local = agg[agg["model_type"] == "local"].sort_values("model_params_b")
    cloud = agg[agg["model_type"] == "cloud"]
    return pd.concat([local, cloud], ignore_index=True)


# ---------------------------------------------------------------------------
# Plot 1: Accuracy by model
# ---------------------------------------------------------------------------

def plot_accuracy(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    """Grouped bar chart: one cluster per model, bars = per-benchmark + average."""
    s = summarise(df)

    benchmarks_present = [b for b in BENCHMARKS if f"acc_{b}" in s.columns]
    bar_labels  = benchmarks_present + ["average"]
    n_bars      = len(bar_labels)
    n_models    = len(s)
    x           = np.arange(n_models)
    width       = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 5))

    colors = list(BM_COLORS.values())[:len(benchmarks_present)] + ["#607D8B"]

    for i, (bm, color) in enumerate(zip(bar_labels, colors)):
        col  = f"acc_{bm}" if bm != "average" else None
        vals = s[col].fillna(0).values if col else s["accuracy"].values
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, color=color,
                      label=bm, zorder=2)
        ax.bar_label(bars, fmt="%.0f%%",
                     labels=[f"{v*100:.0f}%" for v in vals],
                     fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in s["model"]], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (MCQ exact-match)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Accuracy by Model and Benchmark", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", ncol=len(bar_labels))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Shade cloud columns
    for i, row in enumerate(s.itertuples()):
        if row.model_type == "cloud":
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=CLOUD_COLOR, zorder=0)

    fig.tight_layout()
    out = out_dir / "plot1_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 2: Carbon emissions per query
# ---------------------------------------------------------------------------

def plot_carbon(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    """Bar chart: grams CO2 per query. Local = CodeCarbon, cloud = EcoLogits."""
    s = summarise(df)

    has_emissions = s["emissions_g_per_q"].notna().any()

    fig, ax = plt.subplots(figsize=(max(8, len(s) * 1.2), 5))

    colors = [CLOUD_COLOR if t == "cloud" else LOCAL_COLOR for t in s["model_type"]]
    vals   = s["emissions_g_per_q"].fillna(0).values

    bars = ax.bar(range(len(s)), vals, color=colors, width=0.6, zorder=2)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)

    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([_label(m) for m in s["model"]], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("CO₂ per query (g)", fontsize=11)
    ax.set_title(
        "Carbon Emissions per Query\n"
        "Local = CodeCarbon estimate  |  Cloud = EcoLogits estimate",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if not has_emissions:
        ax.text(0.5, 0.5, "No emissions data\n(run with energy tracking enabled)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="grey")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=LOCAL_COLOR, label="Local (CodeCarbon)"),
        Patch(facecolor=CLOUD_COLOR, label="Cloud (EcoLogits)"),
    ], fontsize=9)

    fig.tight_layout()
    out = out_dir / "plot2_carbon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 3: Latency per query
# ---------------------------------------------------------------------------

def plot_latency(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    """Bar chart: median latency with p90 error bar."""
    s = summarise(df)

    fig, ax = plt.subplots(figsize=(max(8, len(s) * 1.2), 5))

    colors  = [CLOUD_COLOR if t == "cloud" else LOCAL_COLOR for t in s["model_type"]]
    medians = s["latency_median"].values
    p90s    = s["latency_p90"].values

    bars = ax.bar(range(len(s)), medians, color=colors, width=0.6, zorder=2)
    ax.bar_label(bars, fmt="%.2fs",
                 labels=[f"{v:.2f}s" for v in medians],
                 fontsize=8, padding=3)

    # p90 error bars
    ax.errorbar(
        range(len(s)), medians,
        yerr=[np.zeros(len(s)), p90s - medians],
        fmt="none", color="black", capsize=5, linewidth=1.5, alpha=0.6,
        label="p90 latency",
    )

    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([_label(m) for m in s["model"]], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Latency per query (s)", fontsize=11)
    ax.set_title(
        "Latency per Query\n(bar = median  |  error bar = p90)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=LOCAL_COLOR, label="Local"),
        Patch(facecolor=CLOUD_COLOR, label="Cloud"),
    ] + [plt.Line2D([0], [0], color="black", linewidth=1.5, label="p90 range")],
    fontsize=9)

    fig.tight_layout()
    out = out_dir / "plot3_latency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 4: Accuracy vs. latency scatter
# ---------------------------------------------------------------------------

def plot_accuracy_vs_latency(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    """Simple scatter: every model labelled, local=blue, cloud=red."""
    s = summarise(df)

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in s.iterrows():
        color  = CLOUD_COLOR if row["model_type"] == "cloud" else LOCAL_COLOR
        marker = "*" if row["model_type"] == "cloud" else "o"
        size   = 250 if row["model_type"] == "cloud" else 100
        ax.scatter(row["latency_median"], row["accuracy"],
                   color=color, marker=marker, s=size, zorder=4)
        ax.annotate(
            _label(row["model"]),
            xy=(row["latency_median"], row["accuracy"]),
            xytext=(7, 4), textcoords="offset points", fontsize=9,
        )

    ax.set_xlabel("Median latency per query (s)", fontsize=11)
    ax.set_ylabel("Accuracy (MCQ exact-match)", fontsize=11)
    ax.set_title("Accuracy vs. Latency", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=LOCAL_COLOR,
               markersize=10, label="Local (Ollama)"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=CLOUD_COLOR,
               markersize=14, label="Cloud (Gemini)"),
    ], fontsize=10)

    fig.tight_layout()
    out = out_dir / "plot4_accuracy_vs_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_all(df: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> list[Path]:
    paths = [
        plot_accuracy(df, out_dir),
        plot_carbon(df, out_dir),
        plot_latency(df, out_dir),
        plot_accuracy_vs_latency(df, out_dir),
    ]
    for p in paths:
        print(f"  Saved: {p}")
    return paths

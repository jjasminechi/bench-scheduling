"""
Four analysis plots for the edge-vs-cloud benchmark.

Energy/carbon plots carry "(estimated)" — values come from CodeCarbon (local)
or EcoLogits (cloud), both TDP/grid-intensity heuristics.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
PLOTS_DIR = REPO_ROOT / "plots"

BENCH_LABELS = {"hellaswag": "HellaSwag", "piqa": "PIQA"}

# Consistent color per model — local blues/greens, cloud warm tones
MODEL_COLORS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "#AEC6E8",   # light blue
    "Qwen/Qwen2.5-1.5B-Instruct": "#4A9EDB",   # medium blue
    "Qwen/Qwen2.5-3B-Instruct":   "#1A5FA8",   # dark blue
    "Qwen/Qwen2.5-7B-Instruct":   "#0A3060",   # navy
    "gemini/gemini-2.5-flash":    "#F4A460",   # sandy orange
    "gemini/gemini-2.5-pro":      "#CC3311",   # red
}
_FALLBACK_LOCAL  = "#4477AA"
_FALLBACK_CLOUD  = "#EE6677"


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, _FALLBACK_CLOUD if "gemini" in model else _FALLBACK_LOCAL)


def _short(model: str) -> str:
    name = model.split("/")[-1]
    return name.replace("-Instruct", "").replace("gemini-", "Gemini ")


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy by model, grouped by benchmark
# ---------------------------------------------------------------------------

def plot_accuracy(df: pd.DataFrame, out_dir: Path) -> Path:
    benchmarks = [b for b in ["hellaswag", "piqa"] if b in df["benchmark"].values]

    # Order: local models by params_b, then cloud models
    local_order = (
        df[df["model_type"] == "local"][["model", "params_b"]]
        .drop_duplicates().sort_values("params_b")["model"].tolist()
    )
    cloud_order = (
        df[df["model_type"] == "cloud"]["model"].unique().tolist()
    )
    all_models = local_order + cloud_order

    # Pivot: rows=model, cols=benchmark
    acc = (
        df.groupby(["model", "benchmark"])["acc"].mean()
        .unstack(fill_value=float("nan"))
        .reindex(all_models)
    )

    n_models = len(all_models)
    n_bench  = len(benchmarks)
    width    = 0.8 / n_bench
    x        = np.arange(n_models)
    offsets  = [(i - (n_bench - 1) / 2) * width for i in range(n_bench)]

    fig, ax = plt.subplots(figsize=(10, 5))

    bench_colors  = {"hellaswag": "#4477AA", "piqa": "#EE6677"}
    bench_hatches = {"hellaswag": "", "piqa": "//"}

    for i, bm in enumerate(benchmarks):
        if bm not in acc.columns:
            continue
        vals = [float(acc.loc[m, bm]) if m in acc.index else float("nan")
                for m in all_models]
        bars = ax.bar(
            x + offsets[i], vals, width,
            label=BENCH_LABELS[bm],
            color=bench_colors[bm], hatch=bench_hatches[bm], alpha=0.85,
        )
        ax.bar_label(bars, fmt=lambda v: f"{v:.0%}", fontsize=7, padding=2)

    # Vertical divider between local and cloud
    if local_order and cloud_order:
        ax.axvline(len(local_order) - 0.5, color="gray", linestyle="--",
                   linewidth=1, alpha=0.5)
        ax.text(len(local_order) - 0.5, 1.02, "cloud →",
                fontsize=8, color="gray", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in all_models], rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Accuracy by Model and Benchmark")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "plot1_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2 — Accuracy vs latency (one point per model, avg over benchmarks)
# ---------------------------------------------------------------------------

def plot_accuracy_vs_latency(df: pd.DataFrame, out_dir: Path) -> Path:
    agg = (
        df.groupby(["model", "model_type"])
        .agg(acc=("acc", "mean"), latency=("latency_median_s", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in agg.iterrows():
        if math.isnan(row["latency"]) or math.isnan(row["acc"]):
            continue
        is_cloud = row["model_type"] == "cloud"
        color  = _color(row["model"])
        marker = "D" if is_cloud else "o"
        ax.scatter(row["latency"], row["acc"], color=color, marker=marker,
                   s=90, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(
            _short(row["model"]), (row["latency"], row["acc"]),
            textcoords="offset points", xytext=(7, 4), fontsize=8,
        )

    # Legend
    handles = [
        mpatches.Patch(color=_color(m), label=_short(m))
        for m in df["model"].unique()
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("Median latency per query (s)")
    ax.set_ylabel("Accuracy (avg over benchmarks)")
    ax.set_title("Accuracy vs Latency")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    path = out_dir / "plot2_accuracy_vs_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 3 — Energy per query (estimated), one bar per model
# ---------------------------------------------------------------------------

def plot_co2(df: pd.DataFrame, out_dir: Path) -> Path:
    agg = (
        df.groupby(["model", "model_type"])["emissions_g_per_query"]
        .mean()
        .reset_index()
    )
    agg = agg[agg["emissions_g_per_query"].apply(
        lambda v: not math.isnan(float(v))
    )].copy()

    if agg.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("CO2 per query (estimated) — no data yet")
        path = out_dir / "plot3_co2.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # Sort: local by params_b, then cloud
    local_order = (
        df[df["model_type"] == "local"][["model", "params_b"]]
        .drop_duplicates().sort_values("params_b")["model"].tolist()
    )
    cloud_order = agg[agg["model_type"] == "cloud"]["model"].tolist()
    order = [m for m in local_order if m in agg["model"].values] + \
            [m for m in cloud_order if m not in local_order]

    agg    = agg.set_index("model").loc[order].reset_index()
    labels = [_short(m) for m in agg["model"]]
    colors = [_color(m) for m in agg["model"]]
    co2_mg = agg["emissions_g_per_query"] * 1000  # g → mg

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, co2_mg, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt=lambda v: f"{v:.2f} mg", fontsize=7, padding=3)
    ax.set_yscale("log")
    ax.set_ylabel("CO2 per query (mg, estimated — log scale)")
    ax.set_title("CO2 Emissions per Query by Model (estimated)")
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=_color(m), label=_short(m)) for m in agg["model"]]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()

    path = out_dir / "plot3_co2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all(
    df: pd.DataFrame,
    queries_per_day: list[int] | None = None,
    electricity_rate: float = 0.12,
) -> list[Path]:
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"\n  Local cost assumption : ${electricity_rate}/kWh")
    print("  Energy/carbon         : ESTIMATED (CodeCarbon / EcoLogits)")

    return [
        plot_accuracy(df, PLOTS_DIR),
        plot_accuracy_vs_latency(df, PLOTS_DIR),
        plot_co2(df, PLOTS_DIR),
    ]

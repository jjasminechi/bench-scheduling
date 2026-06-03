"""
Generate the four analysis plots.

All plots that show energy or carbon carry "(estimated)" in their title
because energy values come from CodeCarbon (local) or EcoLogits (cloud),
both of which use TDP/grid-intensity heuristics, not hardware measurements.

Usage assumptions (configurable via kwargs to generate_all):
  queries_per_day  : [5, 20, 50, 100]   — for CO2 crossover plot
  days_per_year    : 365
  electricity_rate : 0.12 USD/kWh       — for $ cost plot (local)
"""

from __future__ import annotations

import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).parent.parent
PLOTS_DIR  = REPO_ROOT / "plots"

BENCHMARK_LABELS = {
    "hellaswag": "HellaSwag",
    "piqa":      "PIQA",
}
BENCH_COLORS = {
    "hellaswag": "#4477AA",
    "piqa":      "#EE6677",
}
CHANCE_LEVEL = {"hellaswag": 0.25, "piqa": 0.50}


# ---------------------------------------------------------------------------
# Wilson CI helper
# ---------------------------------------------------------------------------

def _wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Returns (lower, center, upper) for Wilson CI at z sigma."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p     = k / n
    denom = 1 + z ** 2 / n
    ctr   = (p + z ** 2 / (2 * n)) / denom
    half  = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, ctr - half), ctr, min(1.0, ctr + half)


def _get_err(row: pd.Series) -> tuple[float, float]:
    """Asymmetric error bars (below, above) from Wilson CI."""
    lo, ctr, hi = _wilson_ci(row["n_correct_norm"], int(row["n_examples"]))
    return ctr - lo, hi - ctr


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def _pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of Pareto-optimal points (minimise x, maximise y).
    """
    mask = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        dominated = False
        for j in range(len(xs)):
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                dominated = True
                break
        mask[i] = not dominated
    return mask


# ---------------------------------------------------------------------------
# Plot 1 — acc_norm vs model size
# ---------------------------------------------------------------------------

def plot_accuracy_vs_size(df: pd.DataFrame, out_dir: Path) -> Path:
    local = df[df["model_type"] == "local"].copy()
    cloud = df[df["model_type"] == "cloud"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for bm, color in BENCH_COLORS.items():
        sub = local[local["benchmark"] == bm].sort_values("params_b")
        if sub.empty:
            continue
        yerr_lo, yerr_hi = zip(*sub.apply(_get_err, axis=1))
        ax.errorbar(
            sub["params_b"], sub["acc_norm"],
            yerr=[list(yerr_lo), list(yerr_hi)],
            fmt="o-", color=color, capsize=4, linewidth=1.5, markersize=6,
            label=BENCHMARK_LABELS.get(bm, bm),
        )
        # Chance line
        ax.axhline(
            CHANCE_LEVEL[bm], color=color, linestyle=":", linewidth=0.8, alpha=0.5
        )

    # Cloud reference (dashed vertical band per benchmark)
    if not cloud.empty:
        for bm, color in BENCH_COLORS.items():
            csub = cloud[cloud["benchmark"] == bm]
            if not csub.empty:
                for _, row in csub.iterrows():
                    _, ctr, _ = _wilson_ci(row["n_correct_norm"], int(row["n_examples"]))
                    ax.axhline(
                        ctr, color=color, linestyle="--", linewidth=1.2, alpha=0.7
                    )

    ax.set_xlabel("Parameters (B)", fontsize=12)
    ax.set_ylabel("acc_norm (95% Wilson CI)", fontsize=12)
    ax.set_title("Accuracy vs Model Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    path = out_dir / "plot1_acc_vs_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2 — acc_norm vs energy  AND  acc_norm vs latency  (Pareto)
# ---------------------------------------------------------------------------

def plot_pareto(df: pd.DataFrame, out_dir: Path) -> Path:
    # Aggregate over benchmarks: macro-average acc_norm, sum correct/n for CI
    agg = (
        df.groupby(["model", "model_type", "params_b"])
        .agg(
            acc_norm=("acc_norm", "mean"),
            n_correct_norm=("n_correct_norm", "sum"),
            n_examples=("n_examples", "sum"),
            energy_kwh=("energy_kwh_per_query", "mean"),
            latency=("latency_median_s", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (xcol, xlabel) in zip(axes, [
        ("energy_kwh", "Energy / query (kWh, estimated)"),
        ("latency",    "Median latency / query (s)"),
    ]):
        valid = agg.dropna(subset=[xcol, "acc_norm"])
        if valid.empty:
            ax.set_title(f"acc_norm vs {xcol}\n(no data)")
            continue

        xs = valid[xcol].values
        ys = valid["acc_norm"].values
        pareto = _pareto_front(xs, ys)

        for _, row in valid.iterrows():
            is_cloud = row["model_type"] == "cloud"
            marker   = "*" if is_cloud else "o"
            color    = "#AA3377" if is_cloud else "#4477AA"
            lo, ctr, hi = _wilson_ci(row["n_correct_norm"], int(row["n_examples"]))
            ax.errorbar(
                row[xcol], row["acc_norm"],
                yerr=[[ctr - lo], [hi - ctr]],
                fmt=marker, color=color, capsize=3, markersize=9,
                label=row["model"].split("/")[-1],
            )

        # Draw Pareto frontier
        pf_xs = xs[pareto]
        pf_ys = ys[pareto]
        order = np.argsort(pf_xs)
        ax.step(pf_xs[order], pf_ys[order], where="post",
                color="orange", linewidth=1.5, linestyle="--",
                label="Pareto frontier", zorder=0)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("acc_norm (macro-avg, 95% CI)", fontsize=11)
        title_suffix = " (estimated)" if "energy" in xcol else ""
        ax.set_title(f"acc_norm vs {xcol.split('_')[0]}{title_suffix}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()

    path = out_dir / "plot2_pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 3 — Cumulative CO2 over one year  (estimated)
# ---------------------------------------------------------------------------

def plot_co2_crossover(
    df: pd.DataFrame,
    out_dir: Path,
    queries_per_day: list[int] | None = None,
    days: int = 365,
) -> Path:
    if queries_per_day is None:
        queries_per_day = [5, 20, 50, 100]

    # Per-query CO2 averaged over benchmarks
    co2 = (
        df.groupby(["model", "model_type"])["emissions_g_per_query"]
        .mean()
        .reset_index()
    )
    local = co2[co2["model_type"] == "local"]
    cloud = co2[co2["model_type"] == "cloud"]

    if local.empty or cloud.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title("Cumulative CO2 over 1 year (estimated)\n(insufficient data)")
        path = out_dir / "plot3_co2_crossover.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    t = np.arange(0, days + 1)

    fig, axes = plt.subplots(
        1, len(queries_per_day), figsize=(4 * len(queries_per_day), 5),
        sharey=True,
    )
    if len(queries_per_day) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")

    for ax, qpd in zip(axes, queries_per_day):
        for idx, (_, row) in enumerate(local.iterrows()):
            g_per_day = row["emissions_g_per_query"] * qpd
            if math.isnan(g_per_day):
                continue
            ax.plot(
                t, t * g_per_day,
                label=row["model"].split("/")[-1],
                color=cmap(idx), linewidth=1.5,
            )

        for _, row in cloud.iterrows():
            g_per_day = row["emissions_g_per_query"] * qpd
            if math.isnan(g_per_day):
                continue
            ax.plot(
                t, t * g_per_day,
                label=row["model"].split("/")[-1],
                color="purple", linewidth=2, linestyle="--",
            )

        ax.set_xlabel("Day", fontsize=11)
        ax.set_title(f"{qpd} queries/day", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Cumulative CO2 (g, estimated)", fontsize=11)
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Cumulative CO2 over 1 year — local vs cloud (ESTIMATED)\n"
        f"Usage assumptions: {queries_per_day} queries/day scenarios",
        fontsize=12,
    )
    fig.tight_layout()

    path = out_dir / "plot3_co2_crossover.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 4 — acc_norm vs $ per 1k queries
# ---------------------------------------------------------------------------

def plot_cost_accuracy(
    df: pd.DataFrame,
    out_dir: Path,
    electricity_rate: float = 0.12,
) -> Path:
    agg = (
        df.groupby(["model", "model_type", "params_b"])
        .agg(
            acc_norm=("acc_norm", "mean"),
            n_correct_norm=("n_correct_norm", "sum"),
            n_examples=("n_examples", "sum"),
            cost_usd=("cost_usd_per_query", "mean"),
            energy_kwh=("energy_kwh_per_query", "mean"),
        )
        .reset_index()
    )

    # Fill missing local cost from energy
    for idx, row in agg.iterrows():
        if row["model_type"] == "local" and math.isnan(row["cost_usd"]):
            if not math.isnan(row["energy_kwh"]):
                agg.at[idx, "cost_usd"] = row["energy_kwh"] * electricity_rate

    agg["cost_per_1k"] = agg["cost_usd"] * 1000

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in agg.iterrows():
        if math.isnan(row["cost_per_1k"]) or math.isnan(row["acc_norm"]):
            continue
        is_cloud = row["model_type"] == "cloud"
        color    = "#AA3377" if is_cloud else "#4477AA"
        marker   = "*" if is_cloud else "o"
        lo, ctr, hi = _wilson_ci(row["n_correct_norm"], int(row["n_examples"]))
        ax.errorbar(
            row["cost_per_1k"], row["acc_norm"],
            yerr=[[ctr - lo], [hi - ctr]],
            fmt=marker, color=color, capsize=3, markersize=9,
            label=row["model"].split("/")[-1],
        )
        ax.annotate(
            row["model"].split("/")[-1],
            (row["cost_per_1k"], row["acc_norm"]),
            textcoords="offset points", xytext=(6, 3), fontsize=7,
        )

    ax.set_xlabel(
        f"$ per 1k queries  (local = electricity @ ${electricity_rate}/kWh, cloud = API pricing)",
        fontsize=11,
    )
    ax.set_ylabel("acc_norm (macro-avg, 95% CI)", fontsize=11)
    ax.set_title("Accuracy vs Cost per 1k queries", fontsize=13)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.4f}"))
    fig.tight_layout()

    path = out_dir / "plot4_cost_accuracy.png"
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
    """
    Generate all four plots.  Returns list of saved paths.

    Prints the usage assumptions so readers know what the plots assume.
    """
    PLOTS_DIR.mkdir(exist_ok=True)
    qpd = queries_per_day or [5, 20, 50, 100]

    print(f"\nPlot assumptions:")
    print(f"  CO2 crossover : {qpd} queries/day scenarios, {365} days/year")
    print(f"  Local $ cost  : ${electricity_rate}/kWh electricity rate")
    print("  Energy/carbon : ESTIMATED (CodeCarbon local, EcoLogits cloud)")

    paths = [
        plot_accuracy_vs_size(df, PLOTS_DIR),
        plot_pareto(df, PLOTS_DIR),
        plot_co2_crossover(df, PLOTS_DIR, qpd),
        plot_cost_accuracy(df, PLOTS_DIR, electricity_rate),
    ]
    return paths

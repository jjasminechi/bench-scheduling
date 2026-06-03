"""
CSV schema and write helpers.

One row per (model, benchmark).  Incrementally flushed so a mid-run crash
doesn't lose already-completed work.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

REPO_ROOT    = Path(__file__).parent.parent
RESULTS_DIR  = REPO_ROOT / "results"
RESULTS_CSV  = RESULTS_DIR / "results.csv"

CSV_FIELDS = [
    "model", "model_type", "params_b", "benchmark",
    "n_examples", "n_correct_acc", "n_correct_norm",
    "acc", "acc_norm", "acc_stderr",
    "latency_median_s", "latency_p90_s",
    "energy_kwh_per_query", "emissions_g_per_query", "energy_method",
    "cost_usd_per_query",
    "prompt_tokens_avg", "completion_tokens_avg",
    "is_mock",
]


def open_csv(path: Path = RESULTS_CSV) -> tuple[Any, Any]:
    """
    Open CSV for append.  Writes header if file is new or schema changed.
    Returns (file_handle, DictWriter).
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    is_new = not path.exists()
    if not is_new:
        with open(path, newline="") as fh:
            existing = fh.readline().strip().split(",")
        if existing != CSV_FIELDS:
            print(
                f"  [results] Schema changed — replacing {path.name} "
                f"(old had {len(existing)} cols, new has {len(CSV_FIELDS)})."
            )
            path.unlink()
            is_new = True

    fh     = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return fh, writer


def fill_local_cost(
    rows: list[dict[str, Any]],
    electricity_rate: float | None = None,
) -> list[dict[str, Any]]:
    """
    Compute cost_usd_per_query for local rows from energy + electricity rate.
    Reads ELECTRICITY_RATE_USD_PER_KWH from env (default 0.12) if rate not given.
    """
    if electricity_rate is None:
        try:
            electricity_rate = float(os.getenv("ELECTRICITY_RATE_USD_PER_KWH", "0.12"))
        except ValueError:
            electricity_rate = 0.12

    for row in rows:
        if row.get("model_type") == "local":
            kwh = row.get("energy_kwh_per_query", float("nan"))
            if not math.isnan(kwh):
                row["cost_usd_per_query"] = kwh * electricity_rate
    return rows


def fmt_float(v: Any, decimals: int = 6) -> Any:
    """Format NaN as empty string for CSV readability."""
    try:
        f = float(v)
        return "" if math.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return v

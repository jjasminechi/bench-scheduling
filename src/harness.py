from __future__ import annotations

import concurrent.futures
import contextlib
import csv
import math
import os
import time
from pathlib import Path
from typing import Any

from .benchmarks import parse_answer, score_answer
from .models import MODEL_PARAMS_B, GeminiModel, OllamaModel
from .power import check_power_available, measure_power

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_CSV  = RESULTS_DIR / "results.csv"

SYSTEM_PROMPT = (
    "You are answering multiple-choice questions. "
    "Follow the instructions in each question exactly."
)

CSV_FIELDS = [
    "model", "model_type", "model_params_b", "benchmark", "example_id",
    "gold", "predicted", "correct", "latency_s",
    "energy_j", "energy_kwh", "energy_method",
    "emissions_g",
    "cost_usd", "prompt_tokens", "completion_tokens",
    "is_mock", "error",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _electricity_rate() -> float:
    try:
        return float(os.getenv("ELECTRICITY_RATE_USD_PER_KWH", "0.12"))
    except ValueError:
        return 0.12


def _nan() -> float:
    return float("nan")


# ---------------------------------------------------------------------------
# Local model runner
# ---------------------------------------------------------------------------

def run_local_benchmark(
    model: OllamaModel,
    benchmark: str,
    examples: list[dict],
    use_power: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Run one local model on all examples for one benchmark.
    One powermetrics session wraps the full loop; latency is timed per example.
    """
    rate = _electricity_rate()
    meter = measure_power() if use_power else contextlib.nullcontext()

    per_ex: list[dict] = []

    with meter:
        for i, ex in enumerate(examples):
            t0    = time.perf_counter()
            resp  = model.call(ex["prompt"], system=SYSTEM_PROMPT)
            lat   = time.perf_counter() - t0

            pred    = parse_answer(resp.raw_text, benchmark)
            correct = score_answer(pred, ex["gold"])

            per_ex.append({
                "predicted": pred,
                "correct":   correct,
                "latency_s": lat,
                "error":     resp.error or "",
            })

            if verbose:
                status = "✓" if correct else "✗"
                print(
                    f"    [{i+1:3d}/{len(examples)}] {ex['id']:20s} "
                    f"gold={ex['gold']} pred={str(pred):4s} {status} {lat:.2f}s",
                    flush=True,
                )

    n = len(examples)
    if use_power and hasattr(meter, "joules") and meter.joules > 0:
        energy_j    = meter.joules / n
        energy_kwh  = meter.kwh()  / n
        emissions_g = getattr(meter, "emissions_g", _nan()) / n
        method      = getattr(meter, "method", "CodeCarbon")
        cost_usd    = energy_kwh * rate
    else:
        energy_j    = _nan()
        energy_kwh  = _nan()
        emissions_g = _nan()
        method      = "none"
        cost_usd    = _nan()

    rows = []
    for ex, r in zip(examples, per_ex):
        rows.append({
            "model":            model.name,
            "model_type":       "local",
            "model_params_b":   MODEL_PARAMS_B.get(model.model_tag),
            "benchmark":        benchmark,
            "example_id":       ex["id"],
            "gold":             ex["gold"],
            "predicted":        r["predicted"],
            "correct":          r["correct"],
            "latency_s":        r["latency_s"],
            "energy_j":         energy_j,
            "energy_kwh":       energy_kwh,
            "energy_method":    method,
            "emissions_g":      emissions_g,
            "cost_usd":         cost_usd,
            "prompt_tokens":    0,
            "completion_tokens":0,
            "is_mock":          False,
            "error":            r["error"],
        })
    return rows


# ---------------------------------------------------------------------------
# Cloud model runner
# ---------------------------------------------------------------------------

def run_cloud_benchmark(
    model: GeminiModel,
    benchmark: str,
    examples: list[dict],
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Run the cloud model on one benchmark. Measures latency and token counts
    per example. Energy is not reported (not externally accessible).
    """
    def _evaluate(i_ex):
        i, ex = i_ex
        resp    = model.call(ex["prompt"], system=SYSTEM_PROMPT)
        pred    = parse_answer(resp.raw_text, benchmark)
        correct = score_answer(pred, ex["gold"])
        cost    = model.cost_usd(resp.prompt_tokens, resp.completion_tokens)

        if verbose:
            status = "✓" if correct else ("~" if resp.is_mock else "✗")
            print(
                f"    [{i+1:3d}/{len(examples)}] {ex['id']:20s} "
                f"gold={ex['gold']} pred={str(pred):4s} {status} "
                f"{resp.latency_s:.2f}s  ${cost:.6f}",
                flush=True,
            )

        return {
            "model":            model.name,
            "model_type":       "cloud",
            "model_params_b":   None,
            "benchmark":        benchmark,
            "example_id":       ex["id"],
            "gold":             ex["gold"],
            "predicted":        pred,
            "correct":          correct,
            "latency_s":        resp.latency_s,
            "energy_j":         resp.energy_j,
            "energy_kwh":       resp.energy_j / 3_600_000 if not math.isnan(resp.energy_j) else _nan(),
            "energy_method":    "EcoLogits (estimated)" if not math.isnan(resp.energy_j) else "N/A",
            "emissions_g":      resp.emissions_g,
            "cost_usd":         cost,
            "prompt_tokens":    resp.prompt_tokens,
            "completion_tokens":resp.completion_tokens,
            "is_mock":          resp.is_mock,
            "error":            resp.error or "",
        }

    max_workers = 10 if not model.is_mock else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(_evaluate, enumerate(examples)))

    return rows


# ---------------------------------------------------------------------------
# CSV writer (incremental — safe against mid-run crashes)
# ---------------------------------------------------------------------------

def _open_csv(path: Path) -> tuple[Any, Any]:
    """Open CSV for append; write header if file is new or has a stale schema."""
    is_new = not path.exists()
    if not is_new:
        # Check that the existing file's header matches our schema
        with open(path, newline="") as check:
            existing_header = check.readline().strip().split(",")
        if existing_header != CSV_FIELDS:
            print(
                f"  [harness] WARNING: {path.name} has a different schema "
                f"(old project?). Replacing it."
            )
            path.unlink()
            is_new = True
    fh     = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return fh, writer


# ---------------------------------------------------------------------------
# Full eval entry point
# ---------------------------------------------------------------------------

def run_eval(
    local_tags:    list[str],
    subset:        dict[str, list[dict]],
    include_cloud: bool = True,
    use_power:     bool = True,
    results_csv:   Path = RESULTS_CSV,
    verbose:       bool = True,
) -> None:
    """
    Run all models over the full benchmark subset and write to CSV incrementally.

    Parameters
    ----------
    local_tags    : Ollama model tags to benchmark.
    subset        : {benchmark: [example, ...]} from benchmarks.build_subset().
    include_cloud : Whether to run the Gemini cloud reference.
    use_power     : If False, skip energy measurement (accuracy/latency only).
    results_csv   : Output CSV path.
    verbose       : Print per-example progress.
    """
    if use_power:
        ok, msg = check_power_available()
        if not ok:
            raise RuntimeError(
                f"Energy measurement requested but hardware power source unavailable:\n"
                f"  {msg}\n\n"
                "Either run with sudo, or pass --no-power to skip energy measurement."
            )

    fh, writer = _open_csv(results_csv)

    try:
        # --- Local models ---
        for tag in local_tags:
            model = OllamaModel(tag)
            if verbose:
                print(f"\n{'='*60}")
                print(f"  LOCAL: {model.name}")
                print(f"{'='*60}")

            for bm, examples in subset.items():
                if verbose:
                    print(f"\n  Benchmark: {bm} ({len(examples)} examples)")

                rows = run_local_benchmark(
                    model, bm, examples,
                    use_power=use_power,
                    verbose=verbose,
                )
                writer.writerows(rows)
                fh.flush()

                if verbose and rows:
                    acc = sum(r["correct"] for r in rows) / len(rows)
                    if use_power and not math.isnan(rows[0]["energy_j"]):
                        print(
                            f"  → acc={acc:.1%}  "
                            f"energy={rows[0]['energy_j']:.4f} J/query  "
                            f"(method: {rows[0]['energy_method']})"
                        )
                    else:
                        print(f"  → acc={acc:.1%}")

        # --- Cloud model ---
        if include_cloud:
            cloud = GeminiModel()
            mock_note = " [MOCK — no credentials]" if cloud.is_mock else ""
            if verbose:
                print(f"\n{'='*60}")
                print(f"  CLOUD: {cloud.name}{mock_note}")
                print(f"{'='*60}")

            for bm, examples in subset.items():
                if verbose:
                    print(f"\n  Benchmark: {bm} ({len(examples)} examples)")

                rows = run_cloud_benchmark(
                    cloud, bm, examples, verbose=verbose
                )
                writer.writerows(rows)
                fh.flush()

                if verbose and rows:
                    acc      = sum(r["correct"] for r in rows) / len(rows)
                    total_cost = sum(
                        r["cost_usd"] for r in rows
                        if not math.isnan(r["cost_usd"])
                    )
                    print(f"  → acc={acc:.1%}  total_cost=${total_cost:.4f}")

    finally:
        fh.close()

    if verbose:
        print(f"\nResults written to {results_csv}")

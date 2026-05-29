"""
Eval harness: runs each model over the eval set and writes a results CSV.

Energy / CO2 tracking:
- Local inference: CodeCarbon EmissionsTracker wraps each Ollama call.
- Cloud inference: EcoLogits patches google-genai at import time (via models.py)
  and adds .impacts.gwp to each response; harness reads co2_kg off ModelResponse.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .models import GeminiModel, OllamaModel, get_cloud_model, get_local_models, MODEL_PARAM_B
from .scoring import extract_json, score_prediction

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
EVAL_PATH = REPO_ROOT / "data" / "eval_set.json"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Energy helpers
# ---------------------------------------------------------------------------

def _codecarbon_tracker(model_tag: str):
    """Return a CodeCarbon tracker or a no-op context manager."""
    try:
        from codecarbon import EmissionsTracker  # type: ignore
        return EmissionsTracker(
            project_name=f"bench-{model_tag}",
            output_dir=str(RESULTS_DIR),
            log_level="error",
            save_to_file=False,
        )
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()



# ---------------------------------------------------------------------------
# Per-call runner
# ---------------------------------------------------------------------------

def _run_one_local(
    model: OllamaModel,
    example: dict[str, Any],
    use_codecarbon: bool = True,
) -> dict[str, Any]:
    user_text = example["input"]
    example_id = example["id"]
    gold = example["expected"]

    co2_kg: float = 0.0

    if use_codecarbon:
        tracker = _codecarbon_tracker(model.model_tag)
        try:
            tracker.start()  # type: ignore[union-attr]
        except Exception:
            tracker = None
    else:
        tracker = None

    response = model.call(user_text)

    if tracker is not None:
        try:
            emissions = tracker.stop()  # type: ignore[union-attr]
            co2_kg = float(emissions) if emissions else 0.0
        except Exception:
            co2_kg = 0.0

    pred_dict = extract_json(response.raw_text)
    scores = score_prediction(pred_dict, gold)

    return {
        "example_id": example_id,
        "model": model.name,
        "model_params_b": MODEL_PARAM_B.get(model.model_tag, None),
        "model_type": "local",
        "input": user_text,
        "intent_gold": gold["intent"],
        "raw_output": response.raw_text[:500],
        "parse_ok": scores["parse_ok"],
        "score_overall": scores["overall"],
        "score_intent": scores["intent"],
        "score_title": scores["title"],
        "score_start_time": scores["start_time"],
        "score_end_time": scores["end_time"],
        "score_duration": scores["duration_minutes"],
        "score_attendees": scores["attendees"],
        "score_recurrence": scores["recurrence"],
        "score_priority": scores["priority"],
        "latency_s": response.latency_s,
        "co2_kg": co2_kg,
        "cost_usd": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "is_mock": False,
        "error": response.error or "",
    }


def _run_one_cloud(
    model: GeminiModel,
    example: dict[str, Any],
) -> dict[str, Any]:
    user_text = example["input"]
    example_id = example["id"]
    gold = example["expected"]

    response = model.call(user_text)

    co2_kg = response.co2_kg
    cost_usd = model.estimate_cost_usd(response.prompt_tokens, response.completion_tokens)

    pred_dict = extract_json(response.raw_text)
    scores = score_prediction(pred_dict, gold)

    return {
        "example_id": example_id,
        "model": model.name,
        "model_params_b": MODEL_PARAM_B.get(model.name, 200.0),
        "model_type": "cloud",
        "input": user_text,
        "intent_gold": gold["intent"],
        "raw_output": response.raw_text[:500],
        "parse_ok": scores["parse_ok"],
        "score_overall": scores["overall"],
        "score_intent": scores["intent"],
        "score_title": scores["title"],
        "score_start_time": scores["start_time"],
        "score_end_time": scores["end_time"],
        "score_duration": scores["duration_minutes"],
        "score_attendees": scores["attendees"],
        "score_recurrence": scores["recurrence"],
        "score_priority": scores["priority"],
        "latency_s": response.latency_s,
        "co2_kg": co2_kg,
        "cost_usd": cost_usd,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "is_mock": response.is_mock,
        "error": response.error or "",
    }


# ---------------------------------------------------------------------------
# Main harness entry point
# ---------------------------------------------------------------------------

def run_eval(
    local_model_tags: list[str] | None = None,
    include_cloud: bool = True,
    cloud_only: bool = False,
    use_codecarbon: bool = True,
    eval_path: Path | None = None,
    results_csv: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full eval harness.

    Args:
        local_model_tags: Ollama model tags to benchmark.
        include_cloud: Whether to include a cloud baseline.
        cloud_only: Skip local inference; replace only the cloud rows in the
            existing CSV and keep all local rows intact.
        cloud_provider: "gemini" (only supported option).
        use_codecarbon: Wrap local calls in CodeCarbon energy tracking.
        eval_path: Path to eval_set.json.
        results_csv: Output path. Defaults to results/results.csv.
        verbose: Print progress to stdout.

    Returns:
        DataFrame of all results.
    """
    eval_path = eval_path or EVAL_PATH
    results_csv = results_csv or RESULTS_DIR / "results.csv"

    with open(eval_path) as f:
        examples = json.load(f)

    cloud_model = get_cloud_model() if include_cloud else None

    rows: list[dict[str, Any]] = []
    done = 0

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if not cloud_only:
        local_models = get_local_models(local_model_tags)
        total = len(examples) * (len(local_models) + (1 if cloud_model else 0))
        for model in local_models:
            _log(f"\n=== {model.name} ===")
            for ex in examples:
                row = _run_one_local(model, ex, use_codecarbon=use_codecarbon)
                rows.append(row)
                done += 1
                status = "✓" if row["parse_ok"] else "✗"
                _log(
                    f"  [{done}/{total}] {ex['id']:20s} "
                    f"score={row['score_overall']:.2f} "
                    f"lat={row['latency_s']:.2f}s {status}"
                )
    else:
        total = len(examples)

    if cloud_model is not None:
        mock_note = " [MOCK — no credentials]" if cloud_model.is_mock else ""
        _log(f"\n=== {cloud_model.name}{mock_note} ===")
        for ex in examples:
            row = _run_one_cloud(cloud_model, ex)
            rows.append(row)
            done += 1
            status = "✓" if row["parse_ok"] else "✗"
            _log(
                f"  [{done}/{total}] {ex['id']:20s} "
                f"score={row['score_overall']:.2f} "
                f"lat={row['latency_s']:.2f}s {status}"
            )

    new_df = pd.DataFrame(rows)

    if cloud_only and results_csv.exists():
        existing = pd.read_csv(results_csv)
        new_bases = {n.replace(" [mock]", "") for n in new_df["model"].unique()}
        keep = existing["model"].apply(
            lambda m: m.replace(" [mock]", "") not in new_bases
        )
        existing = existing[keep]
        df = pd.concat([existing, new_df], ignore_index=True)
        _log(f"  (merged with existing local rows)")
    else:
        df = new_df

    df.to_csv(results_csv, index=False)
    _log(f"\nResults saved to {results_csv}")
    return df
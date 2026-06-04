"""
Local model evaluation via generative multiple-choice.

Both local and cloud use the same method: each example is formatted as a
numbered MCQ prompt, the model generates a short response, and the first
valid digit in that response is the prediction.  This gives a direct
apples-to-apples comparison with cloud_eval.py.

Prompt format
-------------
HellaSwag (4-way):
  "You are given a situation ... Respond only with the number.\n\n
   Context: ...\n1. ending\n2. ending\n3. ending\n4. ending\n\nAnswer: "

PIQA (2-way):
  "You are given a goal and two solutions ... Respond only with the number.\n\n
   Goal: ...\n1. solution\n2. solution\n\nAnswer: "

Energy is tracked by CodeCarbon (machine-level TDP heuristic) — LABELLED AS ESTIMATE.
acc_norm == acc for all rows (no LL normalisation in generative scoring).
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_PARAMS_B: dict[str, float | None] = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B-Instruct":   3.0,
    # "Qwen/Qwen2.5-7B-Instruct":   7.0,
}

CHANCE_LEVEL  = {"hellaswag": 0.25, "piqa": 0.50}
DECENT_THRESH = {"hellaswag": 0.55, "piqa": 0.65}

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
]

VALIDATION_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Maps gold index → expected digit string (same for both benchmarks)
_GOLD_DIGIT = {0: "1", 1: "2", 2: "3", 3: "4"}

# Digits that are valid answers for each benchmark
_VALID_DIGITS = {"hellaswag": frozenset("1234"), "piqa": frozenset("12")}


# ---------------------------------------------------------------------------
# Prompt builder  (mirrors cloud_eval.py format exactly)
# ---------------------------------------------------------------------------

def _build_prompt(benchmark: str, context: str, choices: list[str]) -> str:
    if benchmark == "piqa":
        goal = context.removeprefix("Question: ").removesuffix("\nAnswer:").strip()
        lines = [
            "You are given a goal and two solutions. "
            "Choose the better solution by selecting 1 or 2. "
            "Respond only with the number.",
            "",
            f"Goal: {goal}",
            f"1. {choices[0]}",
            f"2. {choices[1]}",
            "",
            "Answer: ",
        ]
    else:  # hellaswag
        lines = [
            "You are given a situation followed by four possible endings. "
            "Choose the most appropriate ending by selecting the corresponding number. "
            "Respond only with the number of the correct answer.",
            "",
            f"Context: {context}",
        ]
        for i, c in enumerate(choices):
            lines.append(f"{i + 1}. {c}")
        lines += ["", "Answer: "]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer generation + parsing
# ---------------------------------------------------------------------------

def _generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> str:
    """Generate up to 3 new tokens and return the decoded text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _parse_digit(text: str, valid: frozenset[str]) -> str | None:
    """Return first valid digit found in text, or None."""
    for ch in text:
        if ch in valid:
            return ch
    return None


# ---------------------------------------------------------------------------
# Validation check
# ---------------------------------------------------------------------------

def validate_results(
    model_id: str,
    bench_accs: dict[str, float],
    is_validation_model: bool,
) -> None:
    for bm, acc in bench_accs.items():
        chance = CHANCE_LEVEL.get(bm, 0.25)
        if acc <= chance:
            raise RuntimeError(
                f"\n[VALIDATION FAILED] {model_id} on {bm}: "
                f"acc={acc:.1%} ≤ chance={chance:.0%}.\n"
                "This is a scorer/prompt bug, not a weak model.\n"
                "Check: prompt format, digit parsing, gold mapping."
            )
        if is_validation_model and bm in DECENT_THRESH:
            thresh = DECENT_THRESH[bm]
            if acc < thresh:
                print(
                    f"  [WARNING] {model_id} on {bm}: "
                    f"acc={acc:.1%} < expected ≥{thresh:.0%}. "
                    "Above chance but lower than typical — check prompt or model."
                )


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_local_eval(
    model_id: str,
    subset: dict[str, list[dict]],
    use_power: bool = True,
    verbose: bool = True,
    run_validation: bool = False,
) -> list[dict[str, Any]]:
    """
    Evaluate one local model on all benchmarks via generative MCQ.

    Returns a list of per-benchmark result dicts, one per benchmark,
    matching the schema expected by results.write_rows().
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    # --- Device ---
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    dtype = torch.float32 if device_str == "cpu" else torch.float16

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LOCAL: {model_id}")
        print(f"  device={device_str}  dtype={dtype}  power_tracking={use_power}")
        print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    try:
        import accelerate  # type: ignore  # noqa: F401
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
        ).eval()
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True,
        ).to(device_str).eval()

    input_device = torch.device(next(model.parameters()).device)

    # --- CodeCarbon ---
    tracker = None
    if use_power:
        from codecarbon import EmissionsTracker  # type: ignore
        tracker = EmissionsTracker(
            save_to_file=False, log_level="critical", tracking_mode="process"
        )
        tracker.start()

    total_examples = sum(len(v) for v in subset.values())
    global_i       = 0
    is_validation_model = (model_id == VALIDATION_MODEL or run_validation)
    rows: list[dict[str, Any]] = []

    try:
        for benchmark, examples in subset.items():
            if verbose:
                print(f"\n  Benchmark: {benchmark} ({len(examples)} examples)")

            valid = _VALID_DIGITS[benchmark]
            per_ex_lat: list[float] = []
            n_correct = 0
            n_unparseable = 0

            for ex in examples:
                prompt = _build_prompt(benchmark, ex["context"], ex["choices"])
                t0     = time.perf_counter()
                raw    = _generate_answer(model, tokenizer, prompt, input_device)
                lat    = time.perf_counter() - t0
                per_ex_lat.append(lat)
                global_i += 1

                digit    = _parse_digit(raw, valid)
                expected = _GOLD_DIGIT[ex["gold"]]
                correct  = 1.0 if digit == expected else 0.0
                if digit is None:
                    n_unparseable += 1
                n_correct += correct

                if verbose:
                    mark = "✓" if correct else ("?" if digit is None else "✗")
                    print(
                        f"    [{global_i:3d}/{total_examples}] "
                        f"src={ex['source_idx']:5d} "
                        f"gold={expected} pred={str(digit):3s} "
                        f"{mark}  {lat:.2f}s",
                        flush=True,
                    )

            n       = len(examples)
            acc_val = n_correct / n

            if verbose:
                import numpy as np
                print(
                    f"  → {benchmark}: acc={acc_val:.1%}  "
                    f"unparseable={n_unparseable}/{n}  "
                    f"lat_med={float(np.median(per_ex_lat)):.2f}s"
                )

            if run_validation or is_validation_model:
                validate_results(model_id, {benchmark: acc_val}, is_validation_model)

            rows.append({
                "_benchmark":      benchmark,
                "_n":              n,
                "_n_correct":      int(n_correct),
                "_n_unparseable":  n_unparseable,
                "_acc":            acc_val,
                "_latencies":      per_ex_lat,
            })

    finally:
        energy_kwh    = float("nan")
        emissions_g   = float("nan")
        energy_method = "none"

        if tracker is not None:
            tracker.stop()
            data          = tracker.final_emissions_data
            energy_kwh    = float(data.energy_consumed)
            emissions_g   = float(data.emissions) * 1000
            energy_method = "CodeCarbon process (est)"

    import numpy as np

    # Attribute energy proportionally by forward passes per example:
    # HellaSwag generates once per example (1 prompt → 1 generation).
    # PIQA same. Both are 1 forward pass per example in generative mode.
    # (Unlike LL scoring which did 4 and 2 passes respectively.)
    total_examples_actual = sum(r["_n"] for r in rows)

    result_rows = []
    for r in rows:
        bm   = r["_benchmark"]
        lats = r["_latencies"]
        n    = r["_n"]

        energy_per_q   = energy_kwh  / total_examples_actual if not math.isnan(energy_kwh)  else float("nan")
        emissions_per_q = emissions_g / total_examples_actual if not math.isnan(emissions_g) else float("nan")
        acc_val = r["_acc"]

        result_rows.append({
            "model":               model_id,
            "model_type":          "local",
            "params_b":            MODEL_PARAMS_B.get(model_id),
            "benchmark":           bm,
            "n_examples":          n,
            "n_correct_acc":       r["_n_correct"],
            "n_correct_norm":      r["_n_correct"],   # same: no LL normalisation
            "acc":                 acc_val,
            "acc_norm":            acc_val,            # same as acc for generative
            "acc_stderr":          _wilson_stderr(r["_n_correct"], n),
            "latency_median_s":    float(np.median(lats)),
            "latency_p90_s":       float(np.percentile(lats, 90)),
            "energy_kwh_per_query":   energy_per_q,
            "emissions_g_per_query":  emissions_per_q,
            "energy_method":          energy_method,
            "cost_usd_per_query":     float("nan"),
            "prompt_tokens_avg":      0,
            "completion_tokens_avg":  0,
            "is_mock":             False,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilson_stderr(k: float, n: int) -> float:
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1 - p) / n)

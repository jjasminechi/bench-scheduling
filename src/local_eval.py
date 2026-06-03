"""
Local model evaluation via direct log-likelihood (LL) scoring.

Algorithm (identical to lm-evaluation-harness loglikelihood tasks):
  For each (context C, candidate continuation k_i):
    ll_i      = sum of token log-probs for k_i given C
    ll_norm_i = ll_i / len(tokens(k_i))        (length-normalised)
  acc      = 1 if argmax(ll_i)      == gold
  acc_norm = 1 if argmax(ll_norm_i) == gold

acc_norm is the primary metric: it matches published lm-eval numbers and
corrects for choices that differ wildly in token length.

Energy is tracked by CodeCarbon (CPU/RAM TDP heuristic) — LABELLED AS ESTIMATE.
Idle-power is NOT subtracted here; the run wraps inference only (model is
loaded first, then CodeCarbon starts).  Divide total energy by n_examples to
get per-query figures.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Model parameter registry
# ---------------------------------------------------------------------------

MODEL_PARAMS_B: dict[str, float | None] = {
    "Qwen/Qwen2.5-0.5B-Instruct":       0.5,
    "Qwen/Qwen2.5-1.5B-Instruct":       1.5,
    "Qwen/Qwen2.5-3B-Instruct":         3.0,
    "microsoft/Phi-3-mini-4k-instruct": 3.8,
    "Qwen/Qwen2.5-7B-Instruct":         7.0,
}

# At/below-chance triggers a hard stop; these are the 4-way and 2-way levels.
CHANCE_LEVEL = {"hellaswag": 0.25, "piqa": 0.50}

# "Should be well above this" for the validation model (3B / 7B); warn if not.
DECENT_THRESH = {"hellaswag": 0.55, "piqa": 0.65}

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# First model used in validation step (large enough to be reliable)
VALIDATION_MODEL = "Qwen/Qwen2.5-3B-Instruct"


# ---------------------------------------------------------------------------
# Per-example LL scorer
# ---------------------------------------------------------------------------

def _loglikelihood(
    model: Any,
    tokenizer: Any,
    context: str,
    continuation: str,
    device: torch.device,
) -> tuple[float, int]:
    """
    Return (sum_log_prob, n_continuation_tokens).

    Exact lm-eval algorithm:
      1. Move any trailing whitespace from context onto continuation so the
         joint string tokenises the same way the model sees natural text.
      2. Encode context+continuation TOGETHER (avoids boundary artifacts where
         tokenising separately gives different tokens at the split point).
      3. Derive continuation token IDs as whole_enc[len(context_enc):].
      4. Slice logits at the continuation positions and gather per-token log-probs.
    """
    # Step 1: trailing-space transfer
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context      = context[:-n_spaces]

    # Step 2: encode together, then split
    whole_enc   = tokenizer.encode(context + continuation, add_special_tokens=True)
    context_enc = tokenizer.encode(context,                add_special_tokens=True)
    cont_ids    = whole_enc[len(context_enc):]

    if not cont_ids:
        return 0.0, 1

    n_ctx  = len(context_enc)
    n_cont = len(cont_ids)

    full_ids = torch.tensor([whole_enc], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # [n_ctx+n_cont, vocab]

    cont_logits = logits[n_ctx - 1 : n_ctx + n_cont - 1, :]
    log_probs   = torch.log_softmax(cont_logits.float(), dim=-1)
    cont_tensor = torch.tensor(cont_ids, dtype=torch.long).to(device)
    token_lps   = log_probs[torch.arange(n_cont, device=device), cont_tensor]

    return token_lps.sum().item(), n_cont


def score_example(
    model: Any,
    tokenizer: Any,
    context: str,
    choices: list[str],
    device: torch.device,
) -> tuple[int, int, list[float], list[float]]:
    """
    Score one MCQ example.  Returns (pred_acc, pred_norm, raw_lls, norm_lls).
    pred_acc  = argmax of raw log-likelihood
    pred_norm = argmax of CHARACTER-length-normalised log-likelihood  (primary)

    acc_norm convention: character length (lm-eval), NOT token count (Karpathy).
    The two references disagree: lm-eval reports gpt2-xl acc_norm 50.89% using
    character length; Karpathy's script gets 48.93% using token count on the same
    model.  We target lm-eval's published numbers, so character length is correct.
    Do not validate against Karpathy's exact figures.
    """
    lls = []
    for choice in choices:
        ll, _ = _loglikelihood(model, tokenizer, context, choice, device)
        lls.append(ll)

    # Normalise by character length of the choice WITHOUT the prepended space.
    # lm-eval computes len(i) on doc["choices"] (no leading space); our choices
    # have " " prepended for correct tokenisation, so strip before measuring.
    norm_lls  = [ll / max(len(c.lstrip()), 1) for ll, c in zip(lls, choices)]
    pred_acc  = int(max(range(len(lls)),      key=lambda i: lls[i]))
    pred_norm = int(max(range(len(norm_lls)), key=lambda i: norm_lls[i]))
    return pred_acc, pred_norm, lls, norm_lls


# ---------------------------------------------------------------------------
# Validation check
# ---------------------------------------------------------------------------

def validate_results(
    model_id: str,
    bench_accs: dict[str, float],
    is_validation_model: bool,
) -> None:
    """
    Hard-halt on at/below-chance.  Warn (but continue) if above chance but
    below DECENT_THRESH on the designated validation model.
    """
    for bm, acc_norm in bench_accs.items():
        chance = CHANCE_LEVEL.get(bm, 0.25)
        if acc_norm <= chance:
            raise RuntimeError(
                f"\n[VALIDATION FAILED] {model_id} on {bm}: "
                f"acc_norm={acc_norm:.1%} ≤ chance={chance:.0%}.\n"
                "This is an unambiguous scorer bug, not a weak model.\n"
                "Check: context/continuation format, BOS handling, logit slicing."
            )
        if is_validation_model and bm in DECENT_THRESH:
            thresh = DECENT_THRESH[bm]
            if acc_norm < thresh:
                print(
                    f"  [WARNING] {model_id} on {bm}: "
                    f"acc_norm={acc_norm:.1%} < expected ≥{thresh:.0%}. "
                    "Above chance but lower than typical — check format or model download."
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
    Evaluate one local model on all benchmarks in subset.

    Returns a list of per-benchmark result dicts (one dict per benchmark),
    ready to pass to results.write_rows().

    Parameters
    ----------
    model_id       : HuggingFace model ID, e.g. "Qwen/Qwen2.5-3B-Instruct"
    subset         : {benchmark: [example, ...]} from subset.build_subset()
    use_power      : Wrap inference in CodeCarbon (ESTIMATE label applied)
    verbose        : Print per-example progress
    run_validation : If True, apply validate_results() after each benchmark
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    # --- Resolve device ---
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

    # --- Load model (outside CodeCarbon — we measure inference, not loading) ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Use device_map="auto" when accelerate is available; otherwise place manually.
    try:
        import accelerate  # type: ignore  # noqa: F401
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
        ).eval()
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True,
        ).to(device_str).eval()

    # Input device for tensor placement
    input_device = torch.device(next(model.parameters()).device)

    # --- CodeCarbon setup ---
    tracker = None
    if use_power:
        from codecarbon import EmissionsTracker  # type: ignore
        tracker = EmissionsTracker(save_to_file=False, log_level="critical")
        tracker.start()

    total_examples = sum(len(v) for v in subset.values())
    global_i       = 0
    all_latencies  = []
    rows: list[dict[str, Any]] = []
    is_validation_model = (model_id == VALIDATION_MODEL or run_validation)

    try:
        for benchmark, examples in subset.items():
            if verbose:
                print(f"\n  Benchmark: {benchmark} ({len(examples)} examples)")

            per_ex_lat: list[float] = []
            n_correct_acc  = 0
            n_correct_norm = 0

            for ex in examples:
                t0 = time.perf_counter()
                pred_acc, pred_norm, lls, norm_lls = score_example(
                    model, tokenizer, ex["context"], ex["choices"], input_device
                )
                lat = time.perf_counter() - t0
                per_ex_lat.append(lat)
                all_latencies.append(lat)
                global_i += 1

                acc_val  = 1.0 if pred_acc  == ex["gold"] else 0.0
                norm_val = 1.0 if pred_norm == ex["gold"] else 0.0
                n_correct_acc  += acc_val
                n_correct_norm += norm_val

                if verbose:
                    mark = "✓" if norm_val else "✗"
                    print(
                        f"    [{global_i:3d}/{total_examples}] "
                        f"src={ex['source_idx']:5d} "
                        f"gold={ex['gold']} "
                        f"acc={'✓' if acc_val else '✗'} "
                        f"norm={mark}  {lat:.2f}s",
                        flush=True,
                    )

            n = len(examples)
            acc_val  = n_correct_acc  / n
            norm_val = n_correct_norm / n

            if verbose:
                lat_med = _median(per_ex_lat)
                print(
                    f"  → {benchmark}: acc={acc_val:.1%}  "
                    f"acc_norm={norm_val:.1%}  "
                    f"lat_med={lat_med:.2f}s"
                )

            if run_validation or is_validation_model:
                validate_results(model_id, {benchmark: norm_val}, is_validation_model)

            rows.append({
                "_benchmark":        benchmark,
                "_n":                n,
                "_n_correct_acc":    int(n_correct_acc),
                "_n_correct_norm":   int(n_correct_norm),
                "_acc":              acc_val,
                "_acc_norm":         norm_val,
                "_acc_stderr":       _wilson_stderr(n_correct_norm, n),
                "_latencies":        per_ex_lat,
            })

    finally:
        # Always stop CodeCarbon even if something raises
        energy_kwh   = float("nan")
        emissions_g  = float("nan")
        energy_method = "none"

        if tracker is not None:
            tracker.stop()
            data         = tracker.final_emissions_data
            energy_kwh   = float(data.energy_consumed)   # total kWh for all benchmarks
            emissions_g  = float(data.emissions) * 1000  # kg CO2 → g
            energy_method = "CodeCarbon (est)"

    # --- Build final rows (one per benchmark) ---
    # Attribute energy proportionally by forward passes:
    #   HellaSwag = 4 passes/example, PIQA = 2 passes/example.
    # Dividing total energy by total_examples equally would give HellaSwag and
    # PIQA identical per-query figures despite doing 2× as many passes.
    _N_PASSES = {"hellaswag": 4, "piqa": 2}
    total_passes = sum(
        _N_PASSES.get(r["_benchmark"], 1) * r["_n"] for r in rows
    )
    import numpy as np

    result_rows = []
    for r in rows:
        bm   = r["_benchmark"]
        lats = r["_latencies"]
        n    = r["_n"]
        n_passes = _N_PASSES.get(bm, 1)

        if not math.isnan(energy_kwh) and total_passes > 0:
            bm_frac     = (n_passes * n) / total_passes
            energy_per_q   = energy_kwh  * bm_frac / n
            emissions_per_q = emissions_g * bm_frac / n
        else:
            energy_per_q   = float("nan")
            emissions_per_q = float("nan")

        result_rows.append({
            "model":              model_id,
            "model_type":         "local",
            "params_b":           MODEL_PARAMS_B.get(model_id),
            "benchmark":          bm,
            "n_examples":         n,
            "n_correct_acc":      r["_n_correct_acc"],
            "n_correct_norm":     r["_n_correct_norm"],
            "acc":                r["_acc"],
            "acc_norm":           r["_acc_norm"],
            "acc_stderr":         r["_acc_stderr"],
            "latency_median_s":   float(np.median(lats)),
            "latency_p90_s":      float(np.percentile(lats, 90)),
            "energy_kwh_per_query":   energy_per_q,
            "emissions_g_per_query":  emissions_per_q,
            "energy_method":          energy_method,
            "cost_usd_per_query":     float("nan"),  # filled by caller
            "prompt_tokens_avg":      0,
            "completion_tokens_avg":  0,
            "is_mock":            False,
        })

    # --- Clean up GPU/MPS memory ---
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilson_stderr(k: float, n: int, z: float = 1.96) -> float:
    """Standard error from Wilson CI (same formula as lm-eval uses for stderr)."""
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1 - p) / n)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2

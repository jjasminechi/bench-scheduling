"""
Cloud evaluation via Gemini (Vertex AI) — generative MCQ.

IMPORTANT ASYMMETRY WITH LOCAL MODELS:
  Local:  log-likelihood argmax over forced continuations → acc + acc_norm
  Cloud:  generative — model outputs an answer letter/number → acc only

This asymmetry is unavoidable: Gemini does not expose per-token log-probs
over arbitrary forced continuations.  For a capable cloud model, generative
MCQ accuracy is close to log-likelihood accuracy in practice, but the scores
are NOT directly comparable — document this in any analysis.

Energy:  EcoLogits estimate (patches google.genai at import time) — ESTIMATE.
Cost:    EXACT — measured token usage × current pricing.
Latency: EXACT — measured per call.

Gemini 2.5 Flash pricing (Vertex AI, non-thinking, as of 2025-06):
  Input:  $0.30 / 1M tokens
  Output: $2.50 / 1M tokens
Verify at: https://cloud.google.com/vertex-ai/generative-ai/pricing

Requires: GOOGLE_CLOUD_PROJECT in env.  Falls back to a deterministic mock
(always answers "A" / "1") when the env var is absent — safe for offline runs.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import re
import time
import warnings
from typing import Any

# EcoLogits patches generate_content at import time
try:
    from ecologits import EcoLogits  # type: ignore
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EcoLogits.init(providers=["google_genai"])
except Exception:
    pass


# ---------------------------------------------------------------------------
# Gemini pricing  (verify at cloud.google.com/vertex-ai/generative-ai/pricing)
# ---------------------------------------------------------------------------

GEMINI_PRICE_INPUT_PER_M  = 0.30   # USD per 1M input tokens  (non-thinking mode)
GEMINI_PRICE_OUTPUT_PER_M = 2.50   # USD per 1M output tokens

# Mapping from (benchmark, gold_int) → expected answer string for parsing
_GOLD_DIGIT = {
    "hellaswag": {0: "1", 1: "2", 2: "3", 3: "4"},
    "piqa":      {0: "1", 1: "2"},
}


# ---------------------------------------------------------------------------
# Prompt formatter  (MCQ → instruction for Gemini)
# ---------------------------------------------------------------------------

def _build_prompt(benchmark: str, context: str, choices: list[str]) -> str:
    """
    Convert a pre-formatted (context, choices) pair into a generative MCQ prompt.
    """
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
# Answer parser  (last-match to avoid matching choice labels)
# ---------------------------------------------------------------------------

_1234_RE = re.compile(r"\b([1-4])\b")
_12_RE   = re.compile(r"\b([12])\b")


def _parse_answer(raw: str, benchmark: str) -> str | None:
    """Return last matching digit — last-match avoids false hits on numbered lists."""
    raw = raw.strip()
    pattern = _12_RE if benchmark == "piqa" else _1234_RE
    matches = pattern.findall(raw)
    return matches[-1] if matches else None


def _score(predicted: str | None, gold: int, benchmark: str) -> float:
    expected = _GOLD_DIGIT.get(benchmark, {}).get(gold)
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


# ---------------------------------------------------------------------------
# EcoLogits helper
# ---------------------------------------------------------------------------

def _range_midpoint(v: Any) -> float:
    if hasattr(v, "min") and hasattr(v, "max"):
        return (float(v.min) + float(v.max)) / 2
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# GeminiEvaluator
# ---------------------------------------------------------------------------

class GeminiEvaluator:
    """
    Wraps the Vertex AI Gemini client.  Falls back to a mock when
    GOOGLE_CLOUD_PROJECT is not set.
    """

    def __init__(self, model: str | None = None):
        self.model_tag = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        project        = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.location  = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.is_mock   = not bool(project)

        if not self.is_mock:
            try:
                from google import genai  # type: ignore
                self._client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=self.location,
                )
            except Exception as exc:
                self.is_mock    = True
                self._init_error = str(exc)

    @property
    def name(self) -> str:
        tag = f"gemini/{self.model_tag}"
        return f"{tag} [mock]" if self.is_mock else tag

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens      / 1_000_000 * GEMINI_PRICE_INPUT_PER_M
            + completion_tokens / 1_000_000 * GEMINI_PRICE_OUTPUT_PER_M
        )

    def call(self, prompt: str) -> dict[str, Any]:
        """
        Returns dict with keys: raw_text, latency_s, prompt_tokens,
        completion_tokens, energy_j, emissions_g, is_mock, error.
        """
        if self.is_mock:
            return {
                "raw_text":          "A",
                "latency_s":         0.0,
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "energy_j":          float("nan"),
                "emissions_g":       float("nan"),
                "is_mock":           True,
                "error":             "",
            }

        t0 = time.perf_counter()
        try:
            from google.genai import types  # type: ignore
            # thinking_budget=0 is Flash-only; omit for Pro and other models
            cfg_kwargs: dict[str, Any] = {"temperature": 0}
            if "flash" in self.model_tag.lower():
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            cfg = types.GenerateContentConfig(**cfg_kwargs)
            resp    = self._client.models.generate_content(
                model=self.model_tag, contents=prompt, config=cfg
            )
            latency = time.perf_counter() - t0
            usage   = resp.usage_metadata

            prompt_tok   = getattr(usage, "prompt_token_count",     0) or 0
            output_tok   = getattr(usage, "candidates_token_count", 0) or 0
            thinking_tok = getattr(usage, "thoughts_token_count",   0) or 0

            energy_j    = float("nan")
            emissions_g = float("nan")
            impacts     = getattr(resp, "impacts", None)
            if impacts is not None:
                if getattr(impacts, "energy", None) is not None:
                    energy_j = _range_midpoint(impacts.energy.value) * 3_600_000
                if getattr(impacts, "gwp", None) is not None:
                    emissions_g = _range_midpoint(impacts.gwp.value) * 1000

            return {
                "raw_text":          resp.text or "",
                "latency_s":         latency,
                "prompt_tokens":     prompt_tok,
                "completion_tokens": output_tok + thinking_tok,
                "energy_j":          energy_j,
                "emissions_g":       emissions_g,
                "is_mock":           False,
                "error":             "",
            }
        except Exception as exc:
            return {
                "raw_text":          "",
                "latency_s":         time.perf_counter() - t0,
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "energy_j":          float("nan"),
                "emissions_g":       float("nan"),
                "is_mock":           False,
                "error":             str(exc),
            }


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_cloud_eval(
    subset: dict[str, list[dict]],
    model: str | None = None,
    verbose: bool = True,
    max_workers: int = 10,
) -> list[dict[str, Any]]:
    """
    Evaluate one Gemini cloud model on all benchmarks.

    Parameters
    ----------
    model : Gemini model tag, e.g. "gemini-2.5-pro".
            Defaults to GEMINI_MODEL env var or "gemini-2.5-flash".
    """
    import numpy as np

    gem = GeminiEvaluator(model=model)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  CLOUD: {gem.name}")
        if gem.is_mock:
            print("  (mock — GOOGLE_CLOUD_PROJECT not set)")
        print(f"{'='*60}")

    rows: list[dict[str, Any]] = []
    _workers = 1 if gem.is_mock else max_workers

    for benchmark, examples in subset.items():
        if verbose:
            print(f"\n  Benchmark: {benchmark} ({len(examples)} examples)")

        def _eval_one(i_ex: tuple[int, dict]) -> dict:
            i, ex = i_ex
            prompt    = _build_prompt(benchmark, ex["context"], ex["choices"])
            result    = gem.call(prompt)
            predicted = _parse_answer(result["raw_text"], benchmark)
            correct   = _score(predicted, ex["gold"], benchmark)
            cost      = gem.cost_usd(result["prompt_tokens"], result["completion_tokens"])

            if verbose:
                mark = "~" if result["is_mock"] else ("✓" if correct else "✗")
                err_tag = f"  ERR:{result['error'][:60]}" if result["error"] else ""
                print(
                    f"    [{i+1:3d}/{len(examples)}] "
                    f"src={ex['source_idx']:5d} "
                    f"gold={ex['gold']} "
                    f"pred={str(predicted):3s} {mark}  "
                    f"{result['latency_s']:.2f}s  ${cost:.6f}{err_tag}",
                    flush=True,
                )

            return {
                "correct":   correct,
                "latency_s": result["latency_s"],
                "energy_j":  result["energy_j"],
                "emissions_g": result["emissions_g"],
                "cost_usd":  cost,
                "prompt_tokens":     result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "is_mock":           result["is_mock"],
                "error":             result["error"],
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=_workers) as pool:
            per_ex = list(pool.map(_eval_one, enumerate(examples)))

        n             = len(examples)
        n_correct     = sum(r["correct"] for r in per_ex)
        acc_val       = n_correct / n
        acc_stderr    = _wilson_stderr(n_correct, n)
        lats          = [r["latency_s"] for r in per_ex]
        total_cost    = sum(r["cost_usd"] for r in per_ex if not math.isnan(r["cost_usd"]))
        prompt_avg    = sum(r["prompt_tokens"] for r in per_ex) / n
        comp_avg      = sum(r["completion_tokens"] for r in per_ex) / n
        is_mock       = any(r["is_mock"] for r in per_ex)

        # EcoLogits energy: average per query (may be nan if model not in EcoLogits DB)
        energy_js   = [r["energy_j"] for r in per_ex if not math.isnan(r["energy_j"])]
        emissions_gs = [r["emissions_g"] for r in per_ex if not math.isnan(r["emissions_g"])]
        energy_kwh_q  = (sum(energy_js)   / len(energy_js)   / 3_600_000) if energy_js   else float("nan")
        emissions_g_q = (sum(emissions_gs) / len(emissions_gs))            if emissions_gs else float("nan")
        energy_method = "EcoLogits (est)" if energy_js else ("mock" if is_mock else "N/A")

        if verbose:
            print(
                f"  → {benchmark}: acc={acc_val:.1%}  "
                f"total_cost=${total_cost:.4f}  "
                f"lat_med={float(np.median(lats)):.2f}s"
            )

        rows.append({
            "model":              gem.name,
            "model_type":         "cloud",
            "params_b":           None,
            "benchmark":          benchmark,
            "n_examples":         n,
            "n_correct_acc":      int(n_correct),
            "n_correct_norm":     int(n_correct),   # same: no LL normalization for cloud
            "acc":                acc_val,
            "acc_norm":           acc_val,           # generative has no norm variant
            "acc_stderr":         acc_stderr,
            "latency_median_s":   float(np.median(lats)),
            "latency_p90_s":      float(np.percentile(lats, 90)),
            "energy_kwh_per_query":   energy_kwh_q,
            "emissions_g_per_query":  emissions_g_q,
            "energy_method":          energy_method,
            "cost_usd_per_query":     total_cost / n,
            "prompt_tokens_avg":      prompt_avg,
            "completion_tokens_avg":  comp_avg,
            "is_mock":            is_mock,
        })

    return rows


def _wilson_stderr(k: float, n: int) -> float:
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1 - p) / n)

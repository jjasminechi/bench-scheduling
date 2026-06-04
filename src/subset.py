"""
Build and cache the fixed benchmark subset.

Saves data/subset.json once so ALL evaluation paths (local log-likelihood and
cloud generative) score the exact same examples.  Source indices are stored so
the subset is reproducible even if the upstream HuggingFace datasets are
shuffled.

Format saved to disk:
{
  "meta": { "seed", "n_per_benchmark", "benchmarks", "scoring_note", "created" },
  "benchmarks": {
    "hellaswag": [ {"source_idx", "context", "choices", "gold"}, ... ],
    "piqa":      [ ... ]
  }
}

Context and choices are pre-formatted in lm-eval's 0-shot style so both the
local direct-LL scorer and the cloud generative path use identical text.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT   = Path(__file__).parent.parent
DATA_DIR    = REPO_ROOT / "data"
SUBSET_PATH = DATA_DIR / "subset.json"

N_PER_BENCHMARK = 200
SEED            = 42
BENCHMARKS      = ["hellaswag", "piqa"]

# Chance levels for validation
CHANCE_LEVEL = {"hellaswag": 0.25, "piqa": 0.50}


# ---------------------------------------------------------------------------
# Per-benchmark formatters  (exact lm-eval 0-shot style)
# ---------------------------------------------------------------------------

def _preprocess_hellaswag(text: str) -> str:
    """lm-eval's HellaSwag preprocessor — must match exactly for comparable numbers."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _fmt_hellaswag(ex: dict) -> dict[str, Any]:
    ctx     = ex.get("ctx", "")
    label   = ex.get("activity_label", "")
    context = f"{label}: {_preprocess_hellaswag(ctx)}" if label else _preprocess_hellaswag(ctx)
    choices = [_preprocess_hellaswag(e) for e in ex["endings"]]
    gold    = int(ex["label"])
    return {"context": context, "choices": choices, "gold": gold}


def _fmt_piqa(ex: dict) -> dict[str, Any]:
    context = f"Question: {ex['goal']}\nAnswer:"
    choices = [ex["sol1"], ex["sol2"]]
    gold    = int(ex["label"])
    return {"context": context, "choices": choices, "gold": gold}


_FORMATTERS = {
    "hellaswag": _fmt_hellaswag,
    "piqa":      _fmt_piqa,
}

_HF_PARAMS: dict[str, tuple[str, str | None, str]] = {
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "piqa":      ("ybisk/piqa",      None, "validation"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_subset(
    n_per_benchmark: int = N_PER_BENCHMARK,
    seed: int = SEED,
    force: bool = False,
) -> dict[str, list[dict]]:
    """
    Return {benchmark: [example, ...]} sampled with a fixed seed.

    Loads from disk on subsequent calls unless force=True or params changed.
    Each example: {"source_idx": int, "context": str, "choices": list[str], "gold": int}
    """
    if SUBSET_PATH.exists() and not force:
        saved = _read_json(SUBSET_PATH)
        meta  = saved.get("meta", {})
        if (meta.get("seed") == seed
                and meta.get("n_per_benchmark") == n_per_benchmark
                and set(meta.get("benchmarks", [])) == set(BENCHMARKS)):
            print(f"  Loaded cached subset from {SUBSET_PATH.relative_to(REPO_ROOT)}")
            return saved["benchmarks"]
        print("  Cached subset params differ — rebuilding...")

    from datasets import load_dataset  # type: ignore

    rng    = random.Random(seed)
    result: dict[str, list[dict]] = {}

    for bm in BENCHMARKS:
        hf_path, hf_name, split = _HF_PARAMS[bm]
        print(f"  Loading {bm} from HuggingFace ({hf_path})...")
        kwargs: dict[str, Any] = {"split": split}
        if hf_name:
            kwargs["name"] = hf_name
        if bm == "piqa":
            kwargs["trust_remote_code"] = True

        ds      = load_dataset(hf_path, **kwargs)
        n       = min(n_per_benchmark, len(ds))
        indices = sorted(rng.sample(range(len(ds)), n))

        fmt      = _FORMATTERS[bm]
        examples = []
        for src_idx in indices:
            ex = dict(ds[src_idx])
            formatted = fmt(ex)
            examples.append({"source_idx": src_idx, **formatted})

        result[bm] = examples
        print(f"    {len(examples)} examples (source range {indices[0]}–{indices[-1]})")

    DATA_DIR.mkdir(exist_ok=True)
    payload: dict[str, Any] = {
        "meta": {
            "seed":             seed,
            "n_per_benchmark":  n_per_benchmark,
            "benchmarks":       BENCHMARKS,
            "created":          datetime.now(timezone.utc).isoformat(),
            "scoring_note": (
                "Local: log-likelihood argmax (acc + acc_norm, lm-eval algorithm). "
                "Cloud: generative MCQ (unavoidable — Gemini has no forced-continuation logprobs). "
                "Contexts in lm-eval 0-shot format."
            ),
        },
        "benchmarks": result,
    }
    with open(SUBSET_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved subset → {SUBSET_PATH.relative_to(REPO_ROOT)}")
    return result


def load_subset() -> dict[str, list[dict]]:
    """Load saved subset. Raises FileNotFoundError if not yet built."""
    if not SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"No subset at {SUBSET_PATH}. Run build_subset() or:\n"
            "  python scripts/run_eval.py --build-subset"
        )
    return _read_json(SUBSET_PATH)["benchmarks"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def all_examples(subset: dict[str, list[dict]]) -> list[dict]:
    """Flatten subset into a single list, adding "benchmark" key to each example."""
    out = []
    for bm, examples in subset.items():
        for ex in examples:
            out.append({"benchmark": bm, **ex})
    return out

"""
MCQ benchmark loader for hellaswag and piqa.

Scoring approach: generative exact-match. Each example is formatted as a
multiple-choice prompt; the model outputs an answer letter (A/B/C/D or 1/2);
we compare to the gold label. This differs from lm-eval's log-likelihood
ranking by ~1-3pp — noted in README.

Subset is sampled once with a fixed seed and saved to data/benchmark_subset.json
so results are reproducible across runs.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA_DIR  = REPO_ROOT / "data"
SUBSET_PATH = DATA_DIR / "benchmark_subset.json"

N_PER_BENCHMARK = 200
SEED            = 42

BENCHMARKS = ["hellaswag", "piqa"]


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------

def _fmt_hellaswag(ex: dict) -> tuple[str, str]:
    """Returns (prompt, gold_letter)."""
    ctx = (ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip()
    if not ctx:
        ctx = ex.get("ctx", "")
    activity = ex.get("activity_label", "")
    endings  = ex["endings"]
    label    = int(ex["label"])

    header = f"{activity}: {ctx}" if activity else ctx
    choices = "\n".join(f"{chr(65+i)}: {e}" for i, e in enumerate(endings))
    prompt = (
        f"Complete the sentence by choosing the best ending.\n\n"
        f"{header}\n{choices}\n\n"
        f"Reply with only the letter of the best ending (A, B, C, or D)."
    )
    gold = chr(65 + label)   # 0→A, 1→B, 2→C, 3→D
    return prompt, gold


def _fmt_piqa(ex: dict) -> tuple[str, str]:
    """Returns (prompt, gold_letter)."""
    label = int(ex["label"])
    prompt = (
        f"Choose the better solution for the goal.\n\n"
        f"Goal: {ex['goal']}\n"
        f"1: {ex['sol1']}\n"
        f"2: {ex['sol2']}\n\n"
        f"Reply with only 1 or 2."
    )
    gold = str(label + 1)    # 0→"1", 1→"2"
    return prompt, gold


def _fmt_arc_easy(ex: dict) -> tuple[str, str]:
    """Returns (prompt, gold_letter)."""
    choices   = ex["choices"]
    labels    = choices["label"]    # ["A","B","C","D"] or ["1","2","3","4"]
    texts     = choices["text"]
    answer    = ex["answerKey"]

    # Normalise numeric labels to letters
    _num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
    answer = _num_to_letter.get(answer, answer)
    labels = [_num_to_letter.get(l, l) for l in labels]

    choice_block = "\n".join(f"{l}: {t}" for l, t in zip(labels, texts))
    prompt = (
        f"Question: {ex['question']}\n{choice_block}\n\n"
        f"Reply with only the letter of the correct answer (A, B, C, or D)."
    )
    return prompt, answer


_FORMATTERS = {
    "hellaswag": _fmt_hellaswag,
    "piqa":      _fmt_piqa,
    "arc_easy":  _fmt_arc_easy,
}


# ---------------------------------------------------------------------------
# Answer parser
# ---------------------------------------------------------------------------

_ABCD_RE = re.compile(r"\b([A-Da-d])\b")
_12_RE   = re.compile(r"\b([12])\b")


def parse_answer(raw: str, benchmark: str) -> str | None:
    """
    Extract the model's answer from raw output.
    Returns the normalised answer string or None if unparseable.
    """
    raw = raw.strip()
    if benchmark == "piqa":
        m = _12_RE.search(raw)
        if m:
            return m.group(1)
        # Some models output A/B instead of 1/2 — accept as equivalent
        m = _ABCD_RE.search(raw)
        if m:
            return {"A": "1", "B": "2"}.get(m.group(1).upper())
        return None
    else:
        m = _ABCD_RE.search(raw)
        return m.group(1).upper() if m else None


def score_answer(predicted: str | None, gold: str) -> float:
    """1.0 if correct, 0.0 if wrong or unparseable."""
    if predicted is None:
        return 0.0
    return 1.0 if predicted.upper() == gold.upper() else 0.0


# ---------------------------------------------------------------------------
# Dataset loading + subset sampling
# ---------------------------------------------------------------------------

def _load_and_sample(benchmark: str, n: int, rng: random.Random) -> list[dict]:
    """
    Download the benchmark from HuggingFace, subsample n examples,
    and return a list of {id, prompt, gold, source_idx}.
    """
    from datasets import load_dataset  # type: ignore

    if benchmark == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
    elif benchmark == "piqa":
        ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    elif benchmark == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")

    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    fmt     = _FORMATTERS[benchmark]

    examples = []
    for rank, src_idx in enumerate(indices):
        ex               = ds[src_idx]
        prompt, gold     = fmt(ex)
        examples.append({
            "id":         f"{benchmark}_{rank+1:03d}",
            "benchmark":  benchmark,
            "prompt":     prompt,
            "gold":       gold,
            "source_idx": src_idx,
        })
    return examples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_subset(
    n_per_benchmark: int = N_PER_BENCHMARK,
    seed: int = SEED,
    force: bool = False,
) -> dict[str, list[dict]]:
    """
    Build (or load from cache) the benchmark subset.

    Returns {benchmark_name: [example, ...]} where each example has
    keys: id, benchmark, prompt, gold, source_idx.

    The subset is saved to data/benchmark_subset.json on first call and
    reloaded on subsequent calls unless force=True.
    """
    if SUBSET_PATH.exists() and not force:
        with open(SUBSET_PATH) as f:
            saved = json.load(f)
        # Validate it matches the requested parameters
        if (saved.get("seed") == seed
                and saved.get("n_per_benchmark") == n_per_benchmark):
            return {b: saved["benchmarks"][b] for b in BENCHMARKS}
        print(
            f"  [benchmarks] Cached subset has different params "
            f"(seed={saved.get('seed')}, n={saved.get('n_per_benchmark')}). "
            f"Rebuilding..."
        )

    print(f"  Building benchmark subset: {n_per_benchmark} × {len(BENCHMARKS)} benchmarks...")
    rng    = random.Random(seed)
    result = {}
    for bm in BENCHMARKS:
        print(f"    Loading {bm}...")
        result[bm] = _load_and_sample(bm, n_per_benchmark, rng)
        print(f"      {len(result[bm])} examples")

    DATA_DIR.mkdir(exist_ok=True)
    with open(SUBSET_PATH, "w") as f:
        json.dump(
            {
                "seed":             seed,
                "n_per_benchmark":  n_per_benchmark,
                "scoring":          "generative MCQ exact-match (not log-likelihood)",
                "benchmarks":       result,
            },
            f,
            indent=2,
        )
    print(f"  Saved subset → {SUBSET_PATH}")
    return result


def load_subset() -> dict[str, list[dict]]:
    """Load the saved subset. Raises if not yet built."""
    if not SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"No benchmark subset found at {SUBSET_PATH}.\n"
            "Run: python scripts/run_eval.py  (it builds the subset on first run)"
        )
    with open(SUBSET_PATH) as f:
        saved = json.load(f)
    return {b: saved["benchmarks"][b] for b in BENCHMARKS}


def all_examples(subset: dict[str, list[dict]]) -> list[dict]:
    """Flatten the subset dict into a single list."""
    return [ex for bm in BENCHMARKS for ex in subset[bm]]

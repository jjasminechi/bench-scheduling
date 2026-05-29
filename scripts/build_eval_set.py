#!/usr/bin/env python3
"""
Build the eval set from the MASSIVE dataset (AmazonScience/massive, en-US).

Filters to scenario == "calendar" (intents: calendar_set, calendar_query,
calendar_remove), parses [slot : value] span annotations from annot_utt, maps
slots to our eval schema, and writes a balanced ~80-example set to disk.

Free to run — downloads a public HuggingFace dataset, no API key required.

Usage:
    python scripts/build_eval_set.py
    python scripts/build_eval_set.py --n 80 --seed 42
    python scripts/build_eval_set.py --out data/eval_set_debug.json --n 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

_SLOT_RE = re.compile(r'\[(\w+)\s*:\s*([^\]]+)\]')
_DURATION_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(hour|hr|h|minute|min|m)\b', re.I)


def parse_annot_utt(annot_utt: str) -> dict[str, str]:
    """Return {slot_type: slot_value} from a MASSIVE annotated utterance."""
    slots: dict[str, list[str]] = {}
    for m in _SLOT_RE.finditer(annot_utt):
        slot_type = m.group(1).strip()
        slot_value = m.group(2).strip()
        slots.setdefault(slot_type, []).append(slot_value)
    return {k: " ".join(v) for k, v in slots.items()}

INTENT_MAP = {
    "calendar_set": "add_event",
    "calendar_query": "query_schedule",
    "calendar_remove": "cancel",
}


def _parse_duration_minutes(text: str) -> int | None:
    m = _DURATION_RE.search(text)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower()
    return int(n * 60) if unit in ("hour", "hr", "h") else int(n)


def _normalise_recurrence(raw: str) -> str:
    r = raw.lower()
    if any(w in r for w in ("daily", "every day", "each day")):
        return "daily"
    if any(w in r for w in ("weekly", "every week", "once a week")):
        return "weekly"
    if any(w in r for w in ("biweekly", "every two week", "every other week")):
        return "biweekly"
    if any(w in r for w in ("monthly", "every month", "once a month")):
        return "monthly"
    return "none"


def slots_to_expected(massive_intent: str, slots: dict[str, str]) -> dict[str, Any]:
    """Convert a MASSIVE slot dict to our harness 'expected' schema."""
    intent = INTENT_MAP.get(massive_intent, massive_intent)

    title = slots.get("event_name", "")
    date = slots.get("date", "")
    time_ = slots.get("time", "") or slots.get("timeofday", "")
    duration_raw = slots.get("duration", "")
    person = slots.get("person", "")
    location = slots.get("place_name", "")
    recurrence_raw = slots.get("general_frequency", "")

    if date and time_:
        date_ref = f"{date} {time_}"
    elif date:
        date_ref = date
    elif time_:
        date_ref = time_
    else:
        date_ref = ""

    return {
        "intent": intent,
        "title": title,
        "start_time": None,
        "end_time": None,
        "duration_minutes": _parse_duration_minutes(duration_raw) if duration_raw else None,
        "attendees": [person] if person else [],
        "location": location,
        "recurrence": _normalise_recurrence(recurrence_raw) if recurrence_raw else "none",
        "date_reference": date_ref,
        "priority": None,
        "notes": "",
    }

def build_eval_set(n: int = 80, seed: int = 42, verbose: bool = True) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print("Loading MASSIVE en-US test split from HuggingFace (public, free)...")
    ds = load_dataset("AmazonScience/massive", "en-US", split="test")

    scenario_feat = ds.features["scenario"]
    intent_feat = ds.features["intent"]
    cal_id = scenario_feat.str2int("calendar")
    cal_intent_ids = {intent_feat.str2int(n) for n in INTENT_MAP}

    calendar = [ex for ex in ds if ex["scenario"] == cal_id and ex["intent"] in cal_intent_ids]
    if verbose:
        print(f"  {len(calendar)} calendar examples found")

    by_intent: dict[str, list] = {}
    for ex in calendar:
        intent_str = intent_feat.int2str(ex["intent"])
        by_intent.setdefault(intent_str, []).append(ex)
    for intent, exs in sorted(by_intent.items()):
        if verbose:
            print(f"  {intent}: {len(exs)}")

    rng = random.Random(seed)
    intents_sorted = sorted(by_intent.keys())
    per_intent = n // len(intents_sorted)
    remainder = n % len(intents_sorted)

    sampled: list = []
    for i, intent in enumerate(intents_sorted):
        k = min(per_intent + (1 if i < remainder else 0), len(by_intent[intent]))
        sampled.extend(rng.sample(by_intent[intent], k))
    rng.shuffle(sampled)

    if verbose:
        print(f"\nSampled {len(sampled)} examples (seed={seed})")

    counters: dict[str, int] = {}
    records = []
    for ex in sampled:
        intent_str = intent_feat.int2str(ex["intent"])
        counters[intent_str] = counters.get(intent_str, 0) + 1
        prefix = INTENT_MAP[intent_str][:3]   # add, que, can
        example_id = f"{prefix}_{counters[intent_str]:03d}"

        slots = parse_annot_utt(ex["annot_utt"])
        expected = slots_to_expected(intent_str, slots)

        records.append({
            "id": example_id,
            "source": "massive",
            "massive_id": int(ex["id"]),
            "massive_intent": intent_str,
            "input": ex["utt"],
            "expected": expected,
        })

    return records

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build eval set from MASSIVE calendar examples (free, no API key).",
    )
    parser.add_argument("--n", type=int, default=80, help="Total examples to sample (default: 80)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: data/eval_set.json)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    out = args.out or Path(__file__).parent.parent / "data" / "eval_set.json"

    records = build_eval_set(n=args.n, seed=args.seed, verbose=not args.quiet)

    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(records)} examples → {out}")
    intent_counts: dict[str, int] = {}
    for r in records:
        intent_counts[r["expected"]["intent"]] = intent_counts.get(r["expected"]["intent"], 0) + 1
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()

"""
Lenient JSON parsing and field-level scoring.

Design goals:
- Small models emit messy output (prose before/after JSON, markdown fences,
  single-quoted keys, trailing commas). We try hard to extract a JSON object
  before penalising them.
- Scoring is field-level: each predicted field is graded independently so a
  model that gets intent + title right but mis-formats a datetime isn't
  unfairly zeroed out.
- Final score = mean of per-field scores (0.0–1.0).
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _fix_common_issues(text: str) -> str:
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"'([^']*)'", r'"\1"', text)
    return text


def extract_json(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None

    candidates: list[str] = []
    for m in _FENCE_RE.finditer(raw):
        candidates.append(m.group(1))
    m = _OBJ_RE.search(raw)
    if m:
        candidates.append(m.group(0))
    candidates.append(raw.strip())

    for candidate in candidates:
        for text in [candidate, _fix_common_issues(candidate)]:
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

    return None


# ---------------------------------------------------------------------------
# Field-level scoring
# ---------------------------------------------------------------------------

SCORED_FIELDS = [
    "intent",
    "title",
    "start_time",
    "end_time",
    "duration_minutes",
    "attendees",
    "location",
    "recurrence",
    "date_reference",
    "priority",
    "notes",
]

# Fields where null/empty is the most common expected value — weight lower
_LOW_WEIGHT_FIELDS = {"start_time", "end_time", "location", "notes"}

# Weight for each field (must sum to 1)
FIELD_WEIGHTS: dict[str, float] = {
    "intent": 0.25,
    "title": 0.15,
    "start_time": 0.05,
    "end_time": 0.05,
    "duration_minutes": 0.08,
    "attendees": 0.08,
    "location": 0.04,
    "recurrence": 0.08,
    "date_reference": 0.08,
    "priority": 0.08,
    "notes": 0.06,
}
assert abs(sum(FIELD_WEIGHTS.values()) - 1.0) < 1e-6, "weights must sum to 1"


def _normalise(val: Any) -> Any:
    """Lowercase strings, strip whitespace. Pass through non-strings."""
    if isinstance(val, str):
        v = unicodedata.normalize("NFKC", val).strip().lower()
        return v if v not in ("null", "none", "") else None
    return val


def _score_field(pred: Any, gold: Any, field: str) -> float:
    """Return 0.0–1.0 for a single field comparison."""
    pred = _normalise(pred)
    gold = _normalise(gold)

    # Both null/empty → full marks (model correctly output nothing)
    if pred is None and gold is None:
        return 1.0

    # One is null and the other isn't
    if pred is None or gold is None:
        return 0.0

    if field == "attendees":
        if not isinstance(pred, list):
            pred = [pred] if pred else []
        if not isinstance(gold, list):
            gold = [gold] if gold else []
        pred_set = {_normalise(x) for x in pred}
        gold_set = {_normalise(x) for x in gold}
        if not gold_set:
            return 1.0 if not pred_set else 0.5
        if not pred_set:
            return 0.0
        overlap = len(pred_set & gold_set)
        return overlap / max(len(gold_set), len(pred_set))

    if field == "duration_minutes":
        try:
            p, g = int(pred), int(gold)
            if p == g:
                return 1.0
            # Partial credit within 15 minutes
            diff = abs(p - g)
            if diff <= 15:
                return 0.5
            return 0.0
        except (TypeError, ValueError):
            return 0.0

    if field in ("start_time", "end_time"):
        # Exact ISO match
        if str(pred) == str(gold):
            return 1.0
        # Partial credit: same date (first 10 chars)
        if str(pred)[:10] == str(gold)[:10]:
            return 0.5
        return 0.0

    if field == "date_reference":
        if str(pred) == str(gold):
            return 1.0
        # Normalise before token overlap: remove filler words and merge
        # digit-space-unit patterns like "3 pm" → "3pm".
        _STOP = {"at", "on", "the", "a", "an", "for", "in", "of"}

        def _norm_date_tokens(s: str) -> set[str]:
            toks = [t.strip(".,") for t in str(s).lower().split() if t.strip(".,") not in _STOP]
            merged: list[str] = []
            i = 0
            while i < len(toks):
                if i + 1 < len(toks) and toks[i].isdigit() and toks[i + 1] in ("am", "pm"):
                    merged.append(toks[i] + toks[i + 1])
                    i += 2
                else:
                    merged.append(toks[i])
                    i += 1
            return set(merged)

        pred_tok = _norm_date_tokens(str(pred))
        gold_tok = _norm_date_tokens(str(gold))
        if not gold_tok:
            return 1.0
        overlap = len(pred_tok & gold_tok) / max(len(pred_tok), len(gold_tok))
        return round(min(overlap, 1.0), 4)

    # Default: exact string match
    return 1.0 if str(pred) == str(gold) else 0.0


def score_prediction(pred_dict: dict[str, Any] | None, gold_dict: dict[str, Any]) -> dict[str, float]:
    """
    Returns a dict with per-field scores and an overall weighted score.

    Keys: field names + "overall".
    """
    if pred_dict is None:
        return {f: 0.0 for f in SCORED_FIELDS} | {"overall": 0.0, "parse_ok": 0.0}

    per_field = {}
    for f in SCORED_FIELDS:
        per_field[f] = _score_field(pred_dict.get(f), gold_dict.get(f), f)

    overall = sum(FIELD_WEIGHTS[f] * per_field[f] for f in SCORED_FIELDS)
    per_field["overall"] = overall
    per_field["parse_ok"] = 1.0
    return per_field
"""
Model wrappers: Ollama (local) and Google Gemini via Vertex AI (cloud).

- Ollama client talks to a running ollama daemon (default localhost:11434).
- GeminiModel uses Application Default Credentials (gcloud auth application-default login).
  Falls back to a deterministic mock if GOOGLE_CLOUD_PROJECT is not set.
- EcoLogits is initialised at import time so it patches google.genai.models.Models.generate_content
  and adds .impacts (energy, CO2) to every real Gemini response.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# EcoLogits — patches google-genai client at import time
# ---------------------------------------------------------------------------

try:
    from ecologits import EcoLogits  # type: ignore
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EcoLogits.init(providers=["google_genai"])
except Exception:
    pass


def _range_midpoint(v: object) -> float:
    """Return the midpoint of an EcoLogits RangeValue, or the float itself."""
    if hasattr(v, "min") and hasattr(v, "max"):
        return (v.min + v.max) / 2
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    raw_text: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    co2_kg: float = 0.0       # EcoLogits midpoint (cloud) or 0 (local — CodeCarbon handles it)
    is_mock: bool = False
    error: Optional[str] = None


SYSTEM_PROMPT = """\
You are a personal scheduling assistant. Parse the user's request and return
ONLY a valid JSON object — no explanation, no markdown fences.

The JSON must conform exactly to this schema:

{
  "intent":          string,   // one of: add_event | reschedule | cancel | query_schedule | prioritize
  "title":           string,   // event or task title; empty string if not applicable
  "start_time":      string,   // ISO 8601 datetime (e.g. "2025-03-15T09:00:00"); null if unknown
  "end_time":        string,   // ISO 8601 datetime; null if unknown
  "duration_minutes": integer, // duration in minutes; null if not stated
  "attendees":       [string], // list of names or emails; empty list if none
  "location":        string,   // physical or virtual location; empty string if none
  "recurrence":      string,   // one of: none | daily | weekly | biweekly | monthly; default "none"
  "date_reference":  string,   // natural-language date phrase from user (e.g. "next Monday"); empty if none
  "priority":        string,   // one of: high | medium | low | null
  "notes":           string    // any extra details; empty string if none
}

Rules:
- Always return exactly one JSON object.
- Use null (not the string "null") for unknown datetime fields.
- Do not add fields not listed above.
- If the intent is query_schedule or cancel, title may describe what to look up or cancel.\
"""


# ---------------------------------------------------------------------------
# Ollama wrapper (local models)
# ---------------------------------------------------------------------------

class OllamaModel:
    """Wraps ollama.Client for a specific model tag."""

    def __init__(self, model_tag: str, host: Optional[str] = None):
        import ollama  # type: ignore
        self.model_tag = model_tag
        host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=host)

    def call(self, user_text: str) -> ModelResponse:
        t0 = time.perf_counter()
        try:
            resp = self.client.chat(
                model=self.model_tag,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                options={"temperature": 0},
            )
            latency = time.perf_counter() - t0
            return ModelResponse(raw_text=resp["message"]["content"], latency_s=latency)
        except Exception as exc:
            latency = time.perf_counter() - t0
            return ModelResponse(raw_text="", latency_s=latency, error=str(exc))

    @property
    def name(self) -> str:
        return self.model_tag


# ---------------------------------------------------------------------------
# Google Gemini via Vertex AI (cloud baseline)
# ---------------------------------------------------------------------------

class GeminiModel:
    """
    Cloud baseline using Google Gemini via Vertex AI.

    Auth: run `gcloud auth application-default login` before use.
    Set GOOGLE_CLOUD_PROJECT (required) and optionally GOOGLE_CLOUD_LOCATION.
    Falls back to a deterministic mock if GOOGLE_CLOUD_PROJECT is not set.

    CO2 is read from resp.impacts (added by EcoLogits patching) as the midpoint
    of the reported GWP range. Returns 0.0 if EcoLogits is unavailable or the
    model is not in its database.
    """

    COST_PER_1K_INPUT_USD = 0.00015
    COST_PER_1K_OUTPUT_USD = 0.0006

    def __init__(self, model: Optional[str] = None):
        self.model_tag = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.is_mock = not bool(project)
        if not self.is_mock:
            try:
                from google import genai  # type: ignore
                self._client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=self.location,
                )
            except Exception as exc:
                self.is_mock = True
                self._init_error = str(exc)

    def call(self, user_text: str) -> ModelResponse:
        if self.is_mock:
            mock = json.dumps({
                "intent": "add_event", "title": "MOCK GEMINI EVENT",
                "start_time": None, "end_time": None, "duration_minutes": None,
                "attendees": [], "location": "", "recurrence": "none",
                "date_reference": "", "priority": None,
                "notes": "Mock — set GOOGLE_CLOUD_PROJECT to use real Gemini.",
            })
            return ModelResponse(raw_text=mock, latency_s=0.0, is_mock=True)

        t0 = time.perf_counter()
        try:
            from google.genai import types  # type: ignore
            resp = self._client.models.generate_content(
                model=self.model_tag,
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                ),
            )
            latency = time.perf_counter() - t0
            usage = resp.usage_metadata

            # EcoLogits adds .impacts to the response when the model is recognised
            co2_kg = 0.0
            impacts = getattr(resp, "impacts", None)
            if impacts is not None and getattr(impacts, "gwp", None) is not None:
                co2_kg = _range_midpoint(impacts.gwp.value)

            return ModelResponse(
                raw_text=resp.text or "",
                latency_s=latency,
                prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                co2_kg=co2_kg,
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            return ModelResponse(raw_text="", latency_s=latency, error=str(exc))

    def estimate_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1000 * self.COST_PER_1K_INPUT_USD
            + completion_tokens / 1000 * self.COST_PER_1K_OUTPUT_USD
        )

    @property
    def name(self) -> str:
        return f"gemini/{self.model_tag}" + (" [mock]" if self.is_mock else "")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MODELS = [
    "llama3.2:1b",
    "phi3:mini",
    "mistral:7b",
]

MODEL_PARAM_B: dict[str, float] = {
    "llama3.2:1b": 1.0,
    "phi3:mini": 3.8,
    "mistral:7b": 7.0,
    "gemini/gemini-2.5-flash": 50.0,
    "gemini/gemini-2.5-pro": 100.0,
    "gemini/gemini-2.0-flash-001": 50.0,
    "gemini/gemini-2.5-flash [mock]": 50.0,
    "gemini/gemini-2.5-pro [mock]": 100.0,
    "gemini/gemini-2.0-flash-001 [mock]": 50.0,
}


def get_local_models(tags: list[str] | None = None) -> list[OllamaModel]:
    tags = tags or DEFAULT_LOCAL_MODELS
    return [OllamaModel(t) for t in tags]


def get_cloud_model() -> GeminiModel:
    return GeminiModel()

"""
Model wrappers: Ollama (local) and Google Gemini via Vertex AI (cloud).

Local models run through the Ollama daemon (default localhost:11434).

GeminiModel uses google-genai SDK with Vertex AI credentials.
EcoLogits is initialised at import time so it patches
google.genai.models.Models.generate_content and adds .impacts to each response.

Energy for local models is tracked by CodeCarbon (src/power.py).
Cloud CO2 is estimated by EcoLogits from token counts + latency.
Cloud energy/CO2 are estimates; exact $ cost from measured token counts is also reported.

Gemini 2.5 Flash pricing (fetched 2026-06-02, verify at cloud.google.com/vertex-ai/pricing):
  Input:  $0.30 / 1M tokens
  Output: $2.50 / 1M tokens
"""

from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Optional

try:
    from ecologits import EcoLogits  # type: ignore
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EcoLogits.init(providers=["google_genai"])   # correct provider name
except Exception:
    pass


def _range_midpoint(v: object) -> float:
    """Return midpoint of an EcoLogits RangeValue, or the value itself."""
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
    is_mock: bool = False
    energy_j: float = float("nan")    # kWh × 3.6e6; NaN for local (CodeCarbon handles it)
    emissions_g: float = float("nan") # grams CO2; EcoLogits for cloud, NaN for local
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

class OllamaModel:
    """Wraps ollama.Client for a specific model tag."""

    def __init__(self, model_tag: str, host: Optional[str] = None):
        import ollama  # type: ignore
        self.model_tag = model_tag
        host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=host)

    def call(self, prompt: str, system: Optional[str] = None) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        try:
            resp = self.client.chat(
                model=self.model_tag,
                messages=messages,
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
# Gemini via Vertex AI (cloud reference)
# ---------------------------------------------------------------------------

GEMINI_PRICE_INPUT_PER_M  = 0.30   # USD per 1M input tokens
GEMINI_PRICE_OUTPUT_PER_M = 2.50   # USD per 1M output tokens


class GeminiModel:
    """
    Cloud reference using Gemini via Vertex AI (google-genai SDK).

    Reads GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION from the environment.
    Auth: run `gcloud auth application-default login` before use.
    Falls back to a deterministic mock when GOOGLE_CLOUD_PROJECT is not set.

    EcoLogits patches generate_content at import time; each real response
    carries .impacts.energy and .impacts.gwp (RangeValue min/max).
    """

    def __init__(self, model: Optional[str] = None):
        self.model_tag = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        project       = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.is_mock  = not bool(project)

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

    def call(self, prompt: str, system: Optional[str] = None) -> ModelResponse:
        if self.is_mock:
            return ModelResponse(
                raw_text="A",
                latency_s=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                is_mock=True,
            )

        t0 = time.perf_counter()
        try:
            from google.genai import types  # type: ignore
            cfg = types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            resp    = self._client.models.generate_content(
                model=self.model_tag, contents=prompt, config=cfg
            )
            latency = time.perf_counter() - t0
            usage   = resp.usage_metadata

            prompt_tok   = getattr(usage, "prompt_token_count",     0) or 0
            output_tok   = getattr(usage, "candidates_token_count", 0) or 0
            thinking_tok = getattr(usage, "thoughts_token_count",   0) or 0

            # EcoLogits adds .impacts after patching; may be None if model not recognised
            energy_j   = float("nan")
            emissions_g = float("nan")
            impacts = getattr(resp, "impacts", None)
            if impacts is not None:
                if getattr(impacts, "energy", None) is not None:
                    energy_j = _range_midpoint(impacts.energy.value) * 3_600_000  # kWh → J
                if getattr(impacts, "gwp", None) is not None:
                    emissions_g = _range_midpoint(impacts.gwp.value) * 1000       # kg → g

            return ModelResponse(
                raw_text=resp.text or "",
                latency_s=latency,
                prompt_tokens=prompt_tok,
                completion_tokens=output_tok + thinking_tok,
                energy_j=energy_j,
                emissions_g=emissions_g,
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            return ModelResponse(raw_text="", latency_s=latency, error=str(exc))

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens   / 1_000_000 * GEMINI_PRICE_INPUT_PER_M
            + completion_tokens / 1_000_000 * GEMINI_PRICE_OUTPUT_PER_M
        )

    @property
    def name(self) -> str:
        return f"gemini/{self.model_tag}" + (" [mock]" if self.is_mock else "")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_PARAMS_B: dict[str, float | None] = {
    "qwen2.5:0.5b":                   0.5,
    "llama3.2:1b":                    1.0,
    "llama3.2:3b":                    3.0,
    "phi3:mini":                      3.8,
    "mistral:7b":                     7.0,
    "qwen2.5:7b":                     7.0,
    "gemini/gemini-2.5-flash":        None,
    "gemini/gemini-2.5-flash [mock]": None,
}


def get_model_params(model_tag: str) -> float | None:
    """Return parameter count from registry, or parse from tag (e.g. '7b' → 7.0)."""
    if model_tag in MODEL_PARAMS_B:
        return MODEL_PARAMS_B[model_tag]
    m = re.search(r"(\d+(?:\.\d+)?)b", model_tag.lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


DEFAULT_LOCAL_MODELS = [
    "qwen2.5:0.5b",
    "llama3.2:1b",
    "llama3.2:3b",
    "phi3:mini",
    "mistral:7b",
]

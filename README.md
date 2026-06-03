# bench-scheduling

**Edge vs. cloud tradeoff for small LLMs: how small can a local model be while staying useful?**

Benchmarks the Qwen2.5 family (0.5B → 7B) and Phi-3-mini against Gemini 2.5 Flash
across **accuracy, latency, energy/carbon (estimated), and $ cost** on two commonsense
MCQ tasks: HellaSwag and PIQA.

---

## Scoring method

### Local models — log-likelihood argmax
Each answer candidate is scored by the model's log P(candidate | context).
The prediction is argmax — same algorithm as [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Two metrics are reported:
- **acc** — argmax of raw log-likelihood
- **acc_norm** — argmax of length-normalised log-likelihood *(primary metric)*

`acc_norm` corrects for answer choices that differ in token length and is the
number that matches published lm-eval baselines.

### Cloud (Gemini) — generative MCQ
Gemini does not expose per-token log-probs over forced continuations, so it
is scored generatively: the model is asked to output an answer letter/number and
that is compared against gold.  **acc and acc_norm are the same value for cloud
rows** (no length normalisation is possible).  This means local and cloud scores
are *approximately* but not exactly comparable.

---

## Energy / carbon: ESTIMATES

| Source  | Tool          | Method                                       |
|---------|---------------|----------------------------------------------|
| Local   | CodeCarbon    | CPU + RAM TDP heuristics × grid-avg CI       |
| Cloud   | EcoLogits     | Model-architecture LCA estimate              |

Accuracy, latency, and cloud $ cost are **measured/exact**.
Energy and carbon figures carry "(est)" throughout.

---

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) set your electricity rate and Gemini credentials
cp .env.example .env   # edit ELECTRICITY_RATE_USD_PER_KWH and GOOGLE_CLOUD_PROJECT

# 3. Validate one model end-to-end before the full sweep
python scripts/run_eval.py --validate-only --no-power

# 4. Full local sweep (no cloud API calls)
python scripts/run_eval.py --no-cloud

# 5. Full sweep including cloud (asks for confirmation before paid calls)
python scripts/run_eval.py

# 6. Re-plot from an existing CSV
python scripts/run_eval.py --plots-only
```

### Common flags

| Flag | Effect |
|---|---|
| `--no-cloud` | Skip Gemini, local models only |
| `--cloud-only` | Skip local models, Gemini only |
| `--no-power` | Disable CodeCarbon (faster, no energy data) |
| `--validate-only` | Run 3B model + sanity checks, then exit |
| `--rebuild-subset` | Force fresh HuggingFace download + re-sample |
| `--models HF_ID ...` | Override model list |
| `--plots-only` | Re-plot from existing CSV without re-running |

---

## Models

Default local models (open, no login required):

| HuggingFace ID | Params |
|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |
| `Qwen/Qwen2.5-3B-Instruct` | 3.0B |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B |
| `Qwen/Qwen2.5-7B-Instruct` | 7.0B |

To add Llama 3.2 or Mistral, pass them via `--models` and run
`huggingface-cli login` first (license acceptance required).

Cloud: `gemini-2.5-flash` via Vertex AI (`GOOGLE_CLOUD_PROJECT` in `.env`).
Without credentials the cloud path runs a mock (outputs "A" for every example).

---

## Benchmarks

| Benchmark | Task | Choices | Chance | Split used |
|---|---|---|---|---|
| [HellaSwag](https://rowanzellers.com/hellaswag/) | Complete the sentence | 4 | 25% | validation |
| [PIQA](https://yonatanbisk.com/piqa/) | Physical commonsense | 2 | 50% | validation |

200 examples per benchmark (seed 42) are sampled once and saved to
`data/subset.json` with source indices so all models evaluate the identical items.

---

## Plots

| File | Description |
|---|---|
| `plot1_acc_vs_size.png` | acc_norm vs parameter count, 95% Wilson CI, per benchmark |
| `plot2_pareto.png` | acc_norm vs energy/query and vs latency, Pareto frontier |
| `plot3_co2_crossover.png` | Cumulative CO2 over 1 year at 5/20/50/100 queries/day **(estimated)** |
| `plot4_cost_accuracy.png` | acc_norm vs $ per 1k queries (local electricity vs cloud API) |

---

## Project structure

```
bench-scheduling/
├── data/
│   └── subset.json            # fixed-seed subset (built on first run)
├── src/
│   ├── subset.py              # build/load data/subset.json
│   ├── local_eval.py          # direct LL scorer + CodeCarbon wrap
│   ├── cloud_eval.py          # Gemini generative MCQ + EcoLogits
│   ├── results.py             # CSV schema + write helpers
│   └── plots.py               # four analysis plots
├── scripts/
│   └── run_eval.py            # CLI entry point
├── results/
│   └── results.csv            # output (one row per model × benchmark)
├── plots/                     # PNG output
├── requirements.txt
└── .env.example
```

---

## CSV schema

One row per `(model, benchmark)`:

| Column | Notes |
|---|---|
| `model`, `model_type`, `params_b` | identity |
| `benchmark` | hellaswag / piqa |
| `acc`, `acc_norm`, `acc_stderr` | accuracy metrics (Wilson stderr) |
| `latency_median_s`, `latency_p90_s` | per-query latency |
| `energy_kwh_per_query` | **ESTIMATE** (CodeCarbon or EcoLogits) |
| `emissions_g_per_query` | **ESTIMATE** (CO2 grams) |
| `energy_method` | "CodeCarbon (est)" / "EcoLogits (est)" / "mock" |
| `cost_usd_per_query` | local: electricity rate × energy; cloud: exact API cost |
| `is_mock` | True if Gemini credentials were absent |

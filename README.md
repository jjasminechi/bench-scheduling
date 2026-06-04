# bench-scheduling

**Edge vs. cloud tradeoff for small LLMs: how small can a local model be while staying useful?**

Benchmarks the Qwen2.5 family (0.5B → 3B) against Gemini 2.5 Flash and Gemini 2.5 Pro
across **accuracy, latency, energy/carbon (estimated), and $ cost** on two commonsense
MCQ tasks: HellaSwag and PIQA.

---

## Scoring method

Both local and cloud models use **generative MCQ**: each example is formatted
as a numbered prompt, the model generates a short response, and the first valid
digit is the prediction.  This gives a direct apples-to-apples comparison
between local Qwen models and Gemini.

`acc_norm == acc` for all rows.

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
python scripts/run_eval.py --cloud-only --cloud-models gemini-2.5-flash gemini-2.5-pro

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

Cloud: `gemini-2.5-flash` and `gemini-2.5-pro` via Vertex AI (`GOOGLE_CLOUD_PROJECT` in `.env`).

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
| `plot1_accuracy.png` | Accuracy per model grouped by benchmark, with cloud reference lines |
| `plot2_accuracy_vs_latency.png` | Accuracy vs median latency per query |
| `plot3_co2.png` | CO2 emissions per query by model **(estimated)** |
| `plot4_accuracy_vs_cost.png` | Accuracy vs cost per 1k queries |

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
| `energy_kwh_per_query` | (CodeCarbon or EcoLogits) |
| `emissions_g_per_query` | (CO2 grams) |
| `energy_method` | "CodeCarbon (est)" / "EcoLogits (est)" / "mock" |
| `cost_usd_per_query` | local: electricity rate × energy; cloud: exact API cost |
| `is_mock` | True if Gemini credentials were absent |

# bench-scheduling

**Can a small local LLM handle scheduling tasks well enough to replace a cloud model?**

This project benchmarks small local models (via [Ollama](https://ollama.com)) against
Google Gemini on personal-assistant scheduling tasks drawn from the
[MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) dataset.
It measures accuracy, latency, and estimated carbon emissions to explore the
local-vs-cloud tradeoff.

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/your-username/bench-scheduling
cd bench-scheduling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Install and start Ollama  (https://ollama.com/download)
ollama serve &
ollama pull llama3.2:1b
ollama pull phi3:mini
ollama pull mistral:7b

# 3. Build the eval set from MASSIVE (free HuggingFace download, no API key)
python scripts/build_eval_set.py   # writes data/eval_set.json (~80 examples)

# 4. (Optional) Set Gemini credentials for a real cloud baseline
cp .env.example .env
# edit .env: set GOOGLE_CLOUD_PROJECT to your GCP project
# run: gcloud auth application-default login
# Without credentials the harness runs a deterministic mock Gemini response.

# 5. Run the full benchmark
# sudo gives CodeCarbon real power readings via powermetrics (macOS)
sudo python scripts/run_eval.py          # local + mock Gemini
sudo python scripts/run_eval.py --no-cloud  # local only

# 6. Results and plots land in:
#   results/results.csv
#   plots/plot1_accuracy_vs_size.png
#   plots/plot2_energy_cost.png
#   plots/plot3_cumulative_co2.png
#   plots/plot4_latency_vs_accuracy.png
```

## Project structure

```
bench-scheduling/
├── data/
│   └── eval_set.json        # ~80 labelled examples from MASSIVE (calendar scenario)
├── src/
│   ├── models.py            # OllamaModel, GeminiModel (mock fallback when no creds)
│   ├── scoring.py           # Lenient JSON parsing + weighted field-level scoring
│   ├── harness.py           # Eval loop + CodeCarbon / EcoLogits integration
│   └── plots.py             # Four benchmark plots
├── scripts/
│   ├── build_eval_set.py    # Download + parse MASSIVE → data/eval_set.json
│   ├── run_eval.py          # Main CLI entry point
│   └── demo.py              # Interactive scheduling demo
├── results/                 # CSV output (git-ignored)
└── plots/                   # PNG output (git-ignored)
```

---

## Step-by-step guide

### Step 1 — Prompt design

Each model receives a shared system prompt instructing it to parse a natural-language
scheduling request and return a JSON object with 11 fields:

| Field | Type | Notes |
|---|---|---|
| `intent` | enum | add_event / reschedule / cancel / query_schedule / prioritize |
| `title` | string | |
| `start_time` | ISO 8601 / null | |
| `end_time` | ISO 8601 / null | |
| `duration_minutes` | int / null | |
| `attendees` | string[] | |
| `location` | string | |
| `recurrence` | enum | none / daily / weekly / biweekly / monthly |
| `date_reference` | string | natural-language phrase |
| `priority` | enum / null | high / medium / low |
| `notes` | string | |

The prompt is inlined in `src/models.py` as `SYSTEM_PROMPT`.

### Step 2 — Eval set (MASSIVE dataset)

`data/eval_set.json` is built from the
[MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) dataset
(Amazon, 2022) — a multilingual NLU corpus with real user utterances and gold
intent + slot annotations.

We filter the **en-US test split** to `scenario == "calendar"` (three intents:
`calendar_set` → `add_event`, `calendar_query` → `query_schedule`,
`calendar_remove` → `cancel`), parse `[slot : value]` spans from the `annot_utt`
field into our schema, and subsample **~80 balanced examples**.

Rebuild at any time:
```bash
python scripts/build_eval_set.py          # default: 80 examples, seed 42
python scripts/build_eval_set.py --n 120  # larger set
```

### Step 3 — Models

Local models run through Ollama. Default set:

| Model | ~Params |
|---|---|
| `llama3.2:1b` | 1B |
| `phi3:mini` | 3.8B |
| `mistral:7b` | 7B |

Add or replace models with `--models`:
```bash
python scripts/run_eval.py --models llama3.2:1b phi3:mini mistral:7b
```

To enable real Gemini calls:
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id   # or set in .env
```

### Step 4 — Harness

`src/harness.py` orchestrates the eval loop:

- Wraps each input in the scheduling system prompt.
- Records raw output, latency, parse success.
- Uses **CodeCarbon** `EmissionsTracker` around each Ollama call to measure local
  energy use (requires the machine to report power draw — see caveats below).
- Uses **EcoLogits**, which patches the `google-genai` client at startup so every
  real Gemini response carries `.impacts.gwp` (a min/max GWP range); the harness
  records the midpoint.
- Scores each prediction field-by-field (see Scoring below).
- Writes `results/results.csv`.

### Step 5 — Plots

Run `python scripts/run_eval.py` (or `--plots-only` to re-plot from an existing CSV).

| Plot | File | What it shows |
|---|---|---|
| 1 | `plot1_accuracy_vs_size.png` | Weighted accuracy vs. log-scale param count |
| 2 | `plot2_energy_cost.png` | gCO₂eq and milli-USD per query |
| 3 | `plot3_cumulative_co2.png` | Annual cumulative CO₂ at 5/20/50/100 queries/day |
| 4 | `plot4_latency_vs_accuracy.png` | Latency (mean + p95) vs. accuracy scatter |

## Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Local model inference |
| `google-genai` | Cloud baseline (Gemini via Vertex AI) |
| `datasets` | Download + parse MASSIVE eval set |
| `codecarbon` | Local energy / CO₂ tracking |
| `ecologits` | Cloud CO₂ via LCA — patches google-genai client |
| `pandas` | Results dataframe + CSV |
| `matplotlib` | Plots |
| `python-dotenv` | `.env` file loading |

Install all with `pip install -r requirements.txt`

#!/usr/bin/env python3
"""
Run the full edge-LLM tradeoff benchmark.

Usage
-----
# Full run (requires sudo for energy measurement):
sudo python scripts/run_eval.py

# Local models only, no cloud call:
sudo python scripts/run_eval.py --no-cloud

# Skip energy measurement (for testing accuracy/latency without sudo):
python scripts/run_eval.py --no-power

# Specific models only:
sudo python scripts/run_eval.py --models llama3.2:1b mistral:7b

# Re-plot from an existing CSV without re-running inference:
python scripts/run_eval.py --plots-only

# Rebuild the benchmark subset (forces new download + resample):
sudo python scripts/run_eval.py --rebuild-subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.benchmarks import build_subset
from src.harness import run_eval, RESULTS_CSV
from src.models import DEFAULT_LOCAL_MODELS
from src.power import check_power_available


# ---------------------------------------------------------------------------
# Guard: warn before any paid cloud call
# ---------------------------------------------------------------------------

def _confirm_cloud(model_name: str) -> bool:
    print(f"\n  Cloud model: {model_name}")
    print("  This will make REAL API calls to Gemini via Vertex AI.")
    print("  Cost estimate: 600 examples × ~$0.0001/call ≈ $0.06")
    ans = input("  Proceed with cloud inference? [y/N] ").strip().lower()
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge-LLM tradeoff benchmark: accuracy, latency, energy, cost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="TAG",
        help=f"Ollama model tags (default: {' '.join(DEFAULT_LOCAL_MODELS)})",
    )
    parser.add_argument(
        "--no-cloud", action="store_true",
        help="Skip the Gemini cloud reference entirely",
    )
    parser.add_argument(
        "--cloud-only", action="store_true",
        help="Run only the Gemini cloud reference (skip local models)",
    )
    parser.add_argument(
        "--no-power", action="store_true",
        help="Skip energy measurement (accuracy + latency only; no sudo required)",
    )
    parser.add_argument(
        "--rebuild-subset", action="store_true",
        help="Force a fresh benchmark subset (re-download + re-sample)",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip inference — regenerate plots from the existing CSV",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help=f"Results CSV path (default: {RESULTS_CSV})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-example output",
    )
    args = parser.parse_args()

    csv_path = args.csv or RESULTS_CSV

    # ------------------------------------------------------------------
    # Plots-only shortcut
    # ------------------------------------------------------------------
    if args.plots_only:
        if not csv_path.exists():
            print(f"ERROR: no results CSV at {csv_path}. Run without --plots-only first.")
            sys.exit(1)
        _generate_plots(csv_path)
        return

    # ------------------------------------------------------------------
    # Power check (skip if cloud-only or --no-power)
    # ------------------------------------------------------------------
    use_power = not args.no_power and not args.cloud_only
    if use_power:
        ok, msg = check_power_available()
        if not ok:
            print("\nERROR: hardware power measurement unavailable.")
            print(f"  {msg}")
            print("\nOptions:")
            print("  • Ensure CodeCarbon and its dependencies (e.g. pynvml, rapl) are installed")
            print("  • Pass --no-power to skip energy measurement (records NaN)")
            sys.exit(1)
        print(f"\n  Power source: {msg}")

    # ------------------------------------------------------------------
    # Benchmark subset (build once, reuse)
    # ------------------------------------------------------------------
    print("\nLoading benchmark subset...")
    subset = build_subset(force=args.rebuild_subset)
    total  = sum(len(v) for v in subset.values())
    print(f"  {total} examples across {len(subset)} benchmarks "
          f"({', '.join(f'{k}:{len(v)}' for k, v in subset.items())})")

    # ------------------------------------------------------------------
    # Cloud confirmation
    # ------------------------------------------------------------------
    include_cloud = not args.no_cloud
    if include_cloud:
        from src.models import GeminiModel
        cloud = GeminiModel()
        if cloud.is_mock:
            print(f"\n  Cloud: {cloud.name} (mock — GOOGLE_CLOUD_PROJECT not set)")
        else:
            if not _confirm_cloud(cloud.name):
                print("  Skipping cloud inference.")
                include_cloud = False

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    local_tags = [] if args.cloud_only else (args.models or DEFAULT_LOCAL_MODELS)
    if not args.cloud_only:
        print(f"\n  Local models: {', '.join(local_tags)}")
        print(f"  Energy measurement: {'yes (CodeCarbon)' if use_power else 'no (--no-power)'}")
    else:
        print("\n  Mode: cloud only (skipping local inference)")
    print(f"  Output CSV: {csv_path}\n")

    run_eval(
        local_tags    = local_tags,
        subset        = subset,
        include_cloud = include_cloud,
        use_power     = use_power,
        results_csv   = csv_path,
        verbose       = not args.quiet,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _generate_plots(csv_path)


def _generate_plots(csv_path: Path) -> None:
    import pandas as pd
    from src.plots import generate_all

    if not csv_path.exists():
        print(f"No CSV at {csv_path} — skipping plots.")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} rows from {csv_path}")
    print("Generating plots...")
    paths = generate_all(df)
    print("\nPlots written to:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the full benchmark evaluation.

Usage:
    python scripts/run_eval.py                    # all defaults
    python scripts/run_eval.py --models llama3.2:1b  # single local model
    python scripts/run_eval.py --no-cloud         # local only
    python scripts/run_eval.py --cloud-only       # re-run Gemini, keep local rows
    python scripts/run_eval.py --no-codecarbon    # skip energy tracking
    python scripts/run_eval.py --plots-only       # re-plot from existing CSV
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.harness import run_eval, RESULTS_DIR
from src.plots import generate_all
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local vs Gemini on scheduling tasks.")
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="TAG",
        help="Ollama model tags (default: llama3.2:1b phi3:mini mistral:7b)",
    )
    parser.add_argument(
        "--no-cloud", action="store_true",
        help="Skip the Gemini cloud baseline",
    )
    parser.add_argument(
        "--cloud-only", action="store_true",
        help="Re-run only Gemini; merge result into existing CSV (skips local inference)",
    )
    parser.add_argument(
        "--no-codecarbon", action="store_true",
        help="Disable CodeCarbon energy tracking",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip inference — regenerate plots from the existing CSV",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Results CSV path (default: results/results.csv)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-example progress output",
    )
    args = parser.parse_args()

    csv_path = args.csv or RESULTS_DIR / "results.csv"

    if args.plots_only:
        if not csv_path.exists():
            print(f"ERROR: no results CSV at {csv_path}. Run without --plots-only first.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    else:
        df = run_eval(
            local_model_tags=args.models,
            include_cloud=not args.no_cloud,
            cloud_only=args.cloud_only,
            use_codecarbon=not args.no_codecarbon,
            results_csv=csv_path,
            verbose=not args.quiet,
        )

    print("\nGenerating plots...")
    paths = generate_all(df)
    print("\nDone. Plots written to:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()

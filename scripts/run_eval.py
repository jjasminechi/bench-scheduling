#!/usr/bin/env python3
"""
Edge-LLM tradeoff benchmark: accuracy, latency, energy/carbon, and $ cost.

SCORING
-------
Both local and cloud models use generative MCQ: the model generates a digit
(1/2 for PIQA, 1–4 for HellaSwag) and that is compared against gold.
acc_norm == acc for all rows.

ENERGY / CARBON
---------------
All energy and carbon figures are ESTIMATES:
  - Local : CodeCarbon (CPU/RAM TDP heuristics + grid-average carbon intensity)
  - Cloud : EcoLogits (model-architecture estimates)
Accuracy, latency, and cloud $ cost are exact / measured.

USAGE
-----
# Validate one model end-to-end (accuracy + energy + plots):
  python scripts/run_eval.py --validate-only

# Full local sweep (no cloud):
  python scripts/run_eval.py --no-cloud

# Full local + cloud (will ask before paid calls):
  python scripts/run_eval.py

# Re-plot from existing CSV without re-running:
  python scripts/run_eval.py --plots-only

# Rebuild subset from scratch:
  python scripts/run_eval.py --rebuild-subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.subset     import build_subset
from src.results    import open_csv, fill_local_cost, RESULTS_CSV
from src.local_eval import DEFAULT_MODELS, VALIDATION_MODEL


# ---------------------------------------------------------------------------
# Cloud confirmation guard
# ---------------------------------------------------------------------------

def _confirm_cloud(model_name: str, n_examples: int) -> bool:
    input_m  = n_examples / 1_000_000
    # Rough estimate: ~150 input + 2 output tokens per example
    est_cost = input_m * 150 * 0.30 + input_m * 2 * 2.50
    print(f"\n  Cloud model : {model_name}")
    print(f"  Examples    : {n_examples}")
    ans = input("  Proceed with cloud inference? [y/N] ").strip().lower()
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Validation-only run (one model, one benchmark)
# ---------------------------------------------------------------------------

def _run_validation(subset: dict, use_power: bool, csv_path: Path) -> None:
    """
    Run VALIDATION_MODEL on ALL benchmarks, check acc_norm > chance, write CSV.
    Intended to be run once before the full sweep.
    """
    from src.local_eval import run_local_eval

    print(f"\nValidation run: {VALIDATION_MODEL} on all benchmarks")
    rows = run_local_eval(
        model_id=VALIDATION_MODEL,
        subset=subset,
        use_power=use_power,
        verbose=True,
        run_validation=True,
    )
    rows = fill_local_cost(rows)
    fh, writer = open_csv(csv_path)
    writer.writerows(rows)
    fh.flush()
    fh.close()

    print("\n  Validation passed. acc_norm for each benchmark:")
    for r in rows:
        print(f"    {r['benchmark']:12s} acc={r['acc']:.1%}  acc_norm={r['acc_norm']:.1%}  "
              f"(n={r['n_examples']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge-LLM tradeoff benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="HF_ID",
        help=f"HuggingFace model IDs (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument("--no-cloud",       action="store_true", help="Skip all cloud runs")
    parser.add_argument("--cloud-only",     action="store_true", help="Skip local models")
    parser.add_argument(
        "--cloud-models", nargs="+", default=None, metavar="TAG",
        help="Gemini model tags to run (default: $GEMINI_MODEL). "
             "Example: --cloud-models gemini-2.5-flash gemini-2.5-pro",
    )
    parser.add_argument("--no-power",       action="store_true", help="Skip CodeCarbon energy tracking")
    parser.add_argument("--rebuild-subset", action="store_true", help="Force re-download and re-sample")
    parser.add_argument("--validate-only",  action="store_true",
                        help=f"Run {VALIDATION_MODEL} + check, then exit")
    parser.add_argument("--plots-only",     action="store_true", help="Regenerate plots from existing CSV")
    parser.add_argument("--csv",            type=Path, default=None, help="Override output CSV path")
    parser.add_argument("--quiet",          action="store_true")
    parser.add_argument(
        "--queries-per-day", nargs="+", type=int, default=[5, 20, 50, 100],
        metavar="N", help="CO2 crossover plot scenarios (default: 5 20 50 100)",
    )
    parser.add_argument(
        "--electricity-rate", type=float, default=None,
        help="kWh rate for local cost (default: $ELECTRICITY_RATE_USD_PER_KWH or 0.12)",
    )
    args = parser.parse_args()

    csv_path = args.csv or RESULTS_CSV
    verbose  = not args.quiet

    # --- Plots-only shortcut ---
    if args.plots_only:
        _do_plots(csv_path, args)
        return

    # --- Build subset ---
    print("\nBuilding/loading benchmark subset...")
    subset = build_subset(force=args.rebuild_subset)
    total  = sum(len(v) for v in subset.values())
    print(f"  {total} examples across {len(subset)} benchmarks "
          f"({', '.join(f'{k}:{len(v)}' for k, v in subset.items())})")

    # --- Validate-only shortcut ---
    if args.validate_only:
        _run_validation(subset, use_power=not args.no_power, csv_path=csv_path)
        _do_plots(csv_path, args)
        return

    # --- Local models ---
    from src.local_eval import run_local_eval

    local_models = [] if args.cloud_only else (args.models or DEFAULT_MODELS)
    use_power    = not args.no_power and not args.cloud_only

    if local_models:
        print(f"\nLocal models : {', '.join(local_models)}")
        print(f"Energy track : {'CodeCarbon (ESTIMATE)' if use_power else 'disabled (--no-power)'}")
        print(f"Output CSV   : {csv_path}\n")

    fh, writer = open_csv(csv_path)
    try:
        for model_id in local_models:
            rows = run_local_eval(
                model_id=model_id,
                subset=subset,
                use_power=use_power,
                verbose=verbose,
                run_validation=(model_id == VALIDATION_MODEL),
            )
            rows = fill_local_cost(rows, electricity_rate=args.electricity_rate)
            writer.writerows(rows)
            fh.flush()

        # --- Cloud models ---
        if not args.no_cloud:
            import os
            from src.cloud_eval import GeminiEvaluator, run_cloud_eval

            default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            cloud_tags = args.cloud_models or [default_model]

            for tag in cloud_tags:
                gem = GeminiEvaluator(model=tag)
                if gem.is_mock:
                    print(f"\n  Cloud: {gem.name}  (no GOOGLE_CLOUD_PROJECT — running mock)")
                    do_cloud = True
                else:
                    do_cloud = _confirm_cloud(gem.name, total)

                if do_cloud:
                    cloud_rows = run_cloud_eval(subset, model=tag, verbose=verbose)
                    writer.writerows(cloud_rows)
                    fh.flush()
                else:
                    print(f"  Skipping {tag}.")

    finally:
        fh.close()

    print(f"\nResults written to {csv_path}")
    _do_plots(csv_path, args)


def _do_plots(csv_path: Path, args: argparse.Namespace) -> None:
    import os
    import pandas as pd
    from src.plots import generate_all

    if not csv_path.exists():
        print(f"\nNo CSV at {csv_path} — skipping plots.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("\nCSV is empty — skipping plots.")
        return

    electricity_rate = args.electricity_rate or float(
        os.getenv("ELECTRICITY_RATE_USD_PER_KWH", "0.12")
    )

    print(f"\nLoaded {len(df)} rows from {csv_path.name}")
    paths = generate_all(
        df,
        queries_per_day=args.queries_per_day,
        electricity_rate=electricity_rate,
    )
    print("\nPlots saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()

"""
Utility script to (re)build visualizations for selected methods (e.g., RL,
Random) using the precomputed experiment results.

The comprehensive runner already executed the optimizations, but the
visualization step previously failed because JSON artifacts stored numpy types.
This helper inspects the relevant result files, reloads any previously saved
placements/metrics, and recreates the visualization suite. If a JSON file is
missing data it recomputes the placements on the fly.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure repo root on sys.path so "new_reward" package can be imported
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_reward.approaches import (  # pylint: disable=wrong-import-position
    EnhancedWeightedSumReward,
    MultiplicativeHierarchicalReward,
)
from new_reward.evaluation import (  # pylint: disable=wrong-import-position
    ComprehensiveMetrics,
    create_all_visualizations,
)
from new_reward.methods import (  # pylint: disable=wrong-import-position
    rl_optimization,
    random_baseline,
)
from new_reward.regional_filters import filter_region  # pylint: disable=wrong-import-position


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    print(f"✓ Loading dataset from {csv_path}")
    return pd.read_csv(csv_path)


def try_load_result(path: Path) -> Optional[Dict]:
    """Return JSON payload if readable; otherwise None."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️  JSON decode failed for {path.name}; will recompute placements.")
        return None


def build_reward(approach: str, data: pd.DataFrame, region: str):
    if approach == "Approach1":
        return EnhancedWeightedSumReward(data, region=region)
    if approach == "Approach2":
        return MultiplicativeHierarchicalReward(data, region=region)
    raise ValueError(f"Unsupported approach for RL visualizations: {approach}")


def ensure_visualizations_for_file(
    json_path: Path,
    region_data: pd.DataFrame,
    region_name: str,
    method_type: str,
    episodes_override: Optional[int] = None,
    force_rerun: bool = False,
) -> None:
    stem = json_path.stem
    if "_k" not in stem:
        print(f"Skipping {json_path.name} (unexpected filename).")
        return

    method_name, k_part = stem.split("_k")
    k_value = int(k_part)

    approach_name = method_name.split("_")[0]

    placements: Optional[List[int]] = None
    metrics: Optional[Dict[str, float]] = None

    if not force_rerun:
        payload = try_load_result(json_path)
        if payload:
            placements_raw = payload.get("placements", [])
            placements = [int(idx) for idx in placements_raw]
            metrics = payload.get("metrics")

            if len(placements) != k_value:
                print(f"⚠️  {json_path.name}: expected {k_value} placements, "
                      f"found {len(placements)}. Will recompute solution.")
                placements = None
                metrics = None

    if placements is None:
        if method_type == "RL":
            episodes = episodes_override or min(1000, k_value * 100)
            reward_fn = build_reward(approach_name, region_data, region_name)
            print(f"→ Re-running RL for {region_name} - {method_name} (k={k_value}, "
                  f"{episodes} episodes)")
            placements = rl_optimization(
                reward_fn,
                k=k_value,
                episodes=episodes,
                verbose=True,
            )
        elif method_type == "Random":
            print(f"→ Re-running Random baseline for {region_name} - {method_name} "
                  f"(k={k_value})")
            placements = random_baseline(region_data, k_value)
        else:
            raise ValueError(f"Unsupported method type '{method_type}'")

        metrics = ComprehensiveMetrics(region_data, placements).calculate_all()
    elif metrics is None:
        print(f"→ Metrics missing for {json_path.name}; recomputing metrics.")
        metrics = ComprehensiveMetrics(region_data, placements).calculate_all()

    print(f"→ Creating visualizations for {region_name} - {method_name} (k={k_value})")
    create_all_visualizations(
        region_data,
        placements,
        metrics,
        region_name,
        method_name,
        k_value,
    )


def gather_region_data(
    full_data: pd.DataFrame, region_name: str, cache: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    if region_name in cache:
        return cache[region_name]

    if region_name.lower() == "all":
        cache[region_name] = full_data.copy()
    else:
        cache[region_name] = filter_region(full_data, region_name)
    return cache[region_name]


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild visualizations from saved experiment outputs."
    )
    default_results_dir = REPO_ROOT / "new_reward" / "results" / "region_specific"
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"Directory containing per-region JSON results (default: {default_results_dir}).",
    )
    default_data_path = REPO_ROOT / "shade_optimization_data_usc_simple_features.csv"
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path,
        help=f"Path to the master dataset CSV (default: {default_data_path}).",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        help="Optional subset of regions to process (default: auto-detect).",
    )
    parser.add_argument(
        "--k",
        nargs="*",
        type=int,
        help="Optional subset of k-values to process (default: all found).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override RL training episodes when recomputing placements.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore stored placements and recompute method solutions.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["RL"],
        help="Method names to process (e.g., RL Random). Default: RL only.",
    )

    args = parser.parse_args()
    supported_methods = {"RL", "Random"}
    invalid = [m for m in args.methods if m not in supported_methods]
    if invalid:
        raise ValueError(f"Unsupported method(s): {invalid}. "
                         f"Supported: {sorted(supported_methods)}")

    full_data = load_dataset(args.data_path)
    region_cache: Dict[str, pd.DataFrame] = {}

    region_dirs = []
    if args.regions:
        for region_name in args.regions:
            region_dir = args.results_dir / region_name
            if not region_dir.is_dir():
                raise FileNotFoundError(f"Region directory not found: {region_dir}")
            region_dirs.append(region_dir)
    else:
        region_dirs = [
            p for p in args.results_dir.iterdir() if p.is_dir()
        ]

    for region_dir in sorted(region_dirs):
        region_name = region_dir.name
        region_data = gather_region_data(full_data, region_name, region_cache)

        allowed_k = set(args.k) if args.k else None
        files_to_process = []
        for method_name in args.methods:
            files = sorted(region_dir.glob(f"Approach*_{method_name}_k*.json"))
            if allowed_k:
                files = [
                    f for f in files
                    if int(f.stem.split("_k")[1]) in allowed_k
                ]
            files_to_process.extend((f, method_name) for f in files)

        if not files_to_process:
            print(f"No matching result files found for region {region_name}.")
            continue

        print(f"\nProcessing {len(files_to_process)} result(s) for region: {region_name}")
        for json_path, method_type in files_to_process:
            ensure_visualizations_for_file(
                json_path,
                region_data,
                region_name,
                method_type=method_type,
                episodes_override=args.episodes,
                force_rerun=args.force_rerun,
            )

    print("\n✓ Visualization rebuild complete.")


if __name__ == "__main__":
    main()

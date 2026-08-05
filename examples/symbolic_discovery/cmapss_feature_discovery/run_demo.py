# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run the C-MAPSS real-world feature-discovery proof."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from .benchmark import evaluate_benchmark, evaluate_repeated_benchmark
except ImportError:  # pragma: no cover
    from benchmark import evaluate_benchmark, evaluate_repeated_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss_fd001"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cmapss_feature_discovery"))
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-field-rows", type=int, default=6000)
    parser.add_argument("--max-discovered-features", type=int, default=20)
    parser.add_argument("--max-selected-features", type=int, default=25)
    parser.add_argument("--selector", choices=["greedy", "gumbel"], default="gumbel")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--datasets", default="fd001", help="Comma-separated dataset ids, e.g. fd001,fd002")
    args = parser.parse_args()

    common = {
        "hidden": args.hidden,
        "max_field_rows": args.max_field_rows,
        "max_discovered_features": args.max_discovered_features,
        "max_selected_features": args.max_selected_features,
        "selector": args.selector,
    }
    if args.repeats > 1 or "," in args.datasets:
        results = evaluate_repeated_benchmark(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            n_repeats=args.repeats,
            seed=args.seed,
            datasets=tuple(part.strip() for part in args.datasets.split(",") if part.strip()),
            **common,
        )
    else:
        results = evaluate_benchmark(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            **common,
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

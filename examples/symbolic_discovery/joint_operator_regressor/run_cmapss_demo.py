# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run the C-MAPSS joint operator-regressor refinement demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from .cmapss_benchmark import evaluate_cmapss_joint_benchmark
except ImportError:  # pragma: no cover
    from cmapss_benchmark import evaluate_cmapss_joint_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss_fd001"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/joint_operator_regressor_cmapss"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-screened-features", type=int, default=36)
    args = parser.parse_args()
    results = evaluate_cmapss_joint_benchmark(
        args.data_dir,
        args.out_dir,
        seed=args.seed,
        max_screened_features=args.max_screened_features,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

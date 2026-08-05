# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run the joint operator-predictor learning demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from .benchmark import evaluate_benchmark
except ImportError:  # pragma: no cover
    from benchmark import evaluate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/joint_operator_regressor"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = evaluate_benchmark(args.out_dir, seed=args.seed)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

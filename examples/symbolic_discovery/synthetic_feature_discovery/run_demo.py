# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run the controlled synthetic omnibias feature-discovery benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .benchmark import evaluate_benchmark, write_artifacts
except ImportError:  # pragma: no cover
    from benchmark import evaluate_benchmark, write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/synthetic_feature_discovery"))
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = evaluate_benchmark(
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        hidden=args.hidden,
        seed=args.seed,
    )
    write_artifacts(results, args.out_dir)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

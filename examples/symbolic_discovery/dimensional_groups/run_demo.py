# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run the Buckingham-Pi dimensionless-group discovery demo."""

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
    parser.add_argument("--out-dir", type=Path, default=Path("results/dimensional_groups"))
    args = parser.parse_args()

    results = evaluate_benchmark()
    write_artifacts(results, args.out_dir)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

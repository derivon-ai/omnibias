# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run lynx-hare public-CSV discovery (synthetic + committed table)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.symbolic_discovery.public_csv_discovery.discover import evaluate_benchmark

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_PATH = REPO_ROOT / "docs" / "benchmarks" / "public_csv_discovery_smoke.json"


def _out_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    scratch = os.environ.get("OMNIBIAS_SCRATCH")
    if scratch:
        return Path(scratch) / "omnibias_runs" / "public_csv_discovery"
    return REPO_ROOT / "results" / "public_csv_discovery"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smaller field for the CI smoke.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--write-smoke",
        action="store_true",
        help="Write the committed docs/benchmarks smoke JSON.",
    )
    args = parser.parse_args()
    results = evaluate_benchmark(quick=args.quick, seed=args.seed)
    dest = _out_dir(args.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.quick or args.write_smoke:
        payload = {
            "schema": results["schema"],
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": "smoke" if args.quick else "full",
            "gates": results["gates"],
            "synthetic": {
                "gates": results["synthetic"]["gates"],
                "interpolant": results["synthetic"]["interpolant"],
                "interpolant_knobs": results["synthetic"]["interpolant_knobs"],
                "jet_vs_fd_ratio": results["synthetic"]["jet_vs_fd_ratio"],
                "rollout_vs_linear": results["synthetic"]["rollout_vs_linear"],
                "skill_vs_fd": results["synthetic"]["skill_vs_fd"],
                "xy_signs_ok": results["synthetic"]["xy_signs_ok"],
            },
            "public_csv": {
                "gates": results["public_csv"]["gates"],
                "huber_vs_ridge_rollout": results["public_csv"]["huber_vs_ridge_rollout"],
                "interpolant": results["public_csv"]["interpolant"],
                "n_rows": results["public_csv"]["n_rows"],
                "rollout_vs_linear": results["public_csv"]["rollout_vs_linear"],
                "rollout_vs_zero": results["public_csv"]["rollout_vs_zero"],
                "skill_vs_fd": results["public_csv"]["skill_vs_fd"],
                "xy_signs_ok": results["public_csv"]["xy_signs_ok"],
            },
            "honesty": results["honesty"],
        }
        SMOKE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    if not results["gates"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

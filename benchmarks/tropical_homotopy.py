# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: tropical homotopy (theory 01-08). G4 path-following is --full.

Cost vs ``n`` / ``D`` is reported with the refuse cutoff; not in CI
``all_passed``. ``beta -> inf`` is temperature collapse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

COST_NS = (4, 6, 8, 10)
COST_DIMS = (2, 3)
COST_WARMUP = 1
COST_REPEATS = 3
COST_CUTOFF_N = 10
COST_CUTOFF_D = 3


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _run_cost() -> dict[str, Any]:
    """Named leftover: wall vs n/D with the refuse cutoff recorded."""
    from omnibias.struct._core.tropical import TropicalLinear, dual_subdivision

    rng = np.random.default_rng(1)
    rows: list[dict[str, Any]] = []
    for dim in COST_DIMS:
        for n in COST_NS:
            poly = TropicalLinear(rng.normal(size=n), rng.normal(size=(n, dim)))

            def _sub(p=poly) -> None:
                dual_subdivision(p)

            wall = _median_seconds(_sub, warmup=COST_WARMUP, repeats=COST_REPEATS)
            rows.append(
                {
                    "n": int(n),
                    "dim": int(dim),
                    "wall_seconds": float(wall),
                }
            )
    refused = False
    try:
        TropicalLinear(np.zeros(COST_CUTOFF_N + 1), np.zeros((COST_CUTOFF_N + 1, 2)))
    except ValueError:
        refused = True
    d2 = [row for row in rows if row["dim"] == 2]
    log_n = [float(np.log(row["n"])) for row in d2]
    log_t = [float(np.log(max(row["wall_seconds"], 1e-18))) for row in d2]
    slope, _intercept = np.polyfit(np.asarray(log_n), np.asarray(log_t), 1)
    return {
        "name": "cost_vs_n_d",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "rows": rows,
        "time_exponent_vs_n_d2": float(slope),
        "cutoff_n": COST_CUTOFF_N,
        "cutoff_d": COST_CUTOFF_D,
        "refuses_over_cutoff": bool(refused),
        "note": (
            "Sampled dual-subdivision wall vs n at D=2 and D=3. API "
            "refuses n>10 or D>3 (subdivision is exponential in D). "
            "Previous smoke-earned stub with no timing withdrawn. G4 "
            "path-following stays --full. Not in CI all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.struct._core.tropical import (
        TropicalLinear,
        certify_tropical_gap,
        dual_subdivision,
        newton_polytope,
    )

    rng = np.random.default_rng(0)
    poly = TropicalLinear(rng.normal(size=5), rng.normal(size=(5, 2)))
    x = rng.normal(size=(300, 2))
    cert = certify_tropical_gap(poly, x, beta=3.0)
    g1 = {
        "name": "g1_gap",
        "passed": cert.is_sound,
        "bound": cert.bound,
        "measured": cert.measured,
        "in_ci_all_passed": True,
    }
    cells = dual_subdivision(poly)
    verts = newton_polytope(poly)
    g2 = {
        "name": "g2_subdivision",
        "passed": len(cells) >= 1 and len(verts) >= 2,
        "n_cells": len(cells),
        "n_vertices": len(verts),
        "in_ci_all_passed": True,
    }
    cost = _run_cost()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.tropical_homotopy.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "gates_in_scope": ["g1", "g2"],
        },
    )
    payload["gates"] = gates_block([g1, g2])
    payload["cost"] = cost
    payload["honesty"] = {
        "collapse": "beta -> inf (temperature); not delta -> 0",
        "p_vs_np": False,
        "g4_path_following": "full only",
        "cost_earned": False,
        "cost_reported": True,
        "cost_in_ci_all_passed": False,
        "enumeration_cutoff_n": COST_CUTOFF_N,
        "enumeration_cutoff_d": COST_CUTOFF_D,
        "temperature_collapse": True,
        "founding_bias_collapse": False,
    }
    if args.full:
        dest = SCRATCH / "tropical"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "tropical_homotopy.json").write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(f"wrote {write_json('tropical_homotopy_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

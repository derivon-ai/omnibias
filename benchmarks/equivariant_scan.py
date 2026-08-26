# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: equivariant scan (theory 02-08). Discrete C_L, not SO(2).

G5 anisotropic-interface is leftover-recorded (leftover #25) and stays
``--full``. Orbit wall vs ``L`` is reported, not in CI ``all_passed``.
"""

from __future__ import annotations

import argparse
import math
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

COST_LS = (4, 8, 16)
COST_WARMUP = 1
COST_REPEATS = 3
COST_BATCH = 32


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
    """Named leftover: C_L orbit wall vs L; G5 interface task stays --full."""
    import torch
    from omnibias.core.scan import BankSpec
    from omnibias.torch.scan_equivariant import EquivariantScan, OrientationBank

    offsets = BankSpec(offsets=(-0.2, 0.0, 0.2))
    x = torch.randn(COST_BATCH, 2)
    rows: list[dict[str, Any]] = []
    for n_orbit in COST_LS:
        angles = tuple(2.0 * math.pi * i / n_orbit for i in range(n_orbit))
        model = EquivariantScan(2, OrientationBank(angles=angles), offsets)

        def _fwd(m=model, pts=x) -> None:
            m(pts)

        rows.append(
            {
                "L": int(n_orbit),
                "batch": COST_BATCH,
                "n_offsets": len(offsets.offsets),
                "wall_seconds": float(
                    _median_seconds(_fwd, warmup=COST_WARMUP, repeats=COST_REPEATS)
                ),
            }
        )
    return {
        "name": "cost_orbit_vs_L",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 43,
        "leftover_tick": 87,
        "in_ci_all_passed": False,
        "rows": rows,
        "g5_anisotropic_interface": {
            "earned": False,
            "reported": True,
            "leftover_recorded": True,
            "leftover_id": 25,
            "leftover_tick": 63,
            "stays_full": True,
            "need": (
                "orientation bank beats single-direction scan in angular "
                "error, skill > 0 vs random-angle, 5 seeds"
            ),
            "reason": (
                "Leftover #25 leftover-recorded: no interface-orientation "
                "task is wired. EquivariantScan forward wall vs C_L orbit "
                "size is recorded; that is not the named 5-seed skill "
                "gate. Discrete C_L, not SO(2)."
            ),
        },
        "note": (
            "Leftover #43 leftover-recorded: EquivariantScan wall vs L "
            "(L orientations multiply cost). G5 anisotropic-interface "
            "is leftover #25 and stays --full. Previous "
            "g5_anisotropic_interface passed=True / --full-only stub "
            "with no timing withdrawn. Not in CI all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.torch.scan_equivariant import steerable_basis

    g1 = steerable_basis(1, 2, base="gaussian") is not None
    g2 = steerable_basis(1, 2, base="tanh") is None
    entries: list[dict[str, Any]] = [
        {"name": "g1_gaussian_steer", "passed": g1, "in_ci_all_passed": True},
        {"name": "g2_nongaussian_none", "passed": g2, "in_ci_all_passed": True},
    ]
    cost = _run_cost()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.equivariant_scan.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "gates_in_scope": ["g1", "g2"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["cost"] = cost
    payload["honesty"] = {
        "steering": "gaussian-family only",
        "orbit": "C_L, not SO(2)",
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "g5_anisotropic_interface_earned": False,
        "g5_reported": True,
        "g5_leftover_recorded": True,
        "g5_leftover_id": 25,
        "g5_leftover_tick": 63,
        "g5_stays_full": True,
        "cost_earned": False,
        "cost_reported": True,
        "cost_leftover_recorded": True,
        "cost_leftover_id": 43,
        "cost_leftover_tick": 87,
        "cost_in_ci_all_passed": False,
    }
    if args.full:
        dest = SCRATCH / "equivariant" / "equivariant_scan.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('equivariant_scan_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

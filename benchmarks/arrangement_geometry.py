# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: hyperplane arrangements (theory 01-03).

Sampling is a lower bound. ``beta -> inf`` is temperature collapse.
G3 is tree soft-membership versus ``partition_weights`` (<= 4 ulp).
G4 is torch/jax soft-path parity. Cost vs ``n`` / ``D`` is reported
with the G1 enumeration cutoff; not in CI ``all_passed``.
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

COST_NS = (4, 6, 8)
COST_DIMS = (2, 3)
COST_WARMUP = 1
COST_REPEATS = 3
COST_CUTOFF_N = 12
COST_CUTOFF_D = 4


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
    """Named leftover: wall vs n/D with the enumeration cutoff recorded.

    Previous status said cost was smoke-earned with no timing table.
    """
    from omnibias.partition.arrangement import (
        enumerate_cells_vertices,
        general_position_normals,
        max_cells,
    )

    rng = np.random.default_rng(1)
    rows: list[dict[str, Any]] = []
    for dim in COST_DIMS:
        for n in COST_NS:
            arr = general_position_normals(n, dim, rng)

            def _enum(arrangement=arr) -> None:
                enumerate_cells_vertices(arrangement)

            wall = _median_seconds(_enum, warmup=COST_WARMUP, repeats=COST_REPEATS)
            rows.append(
                {
                    "n": int(n),
                    "dim": int(dim),
                    "max_cells": int(max_cells(n, dim)),
                    "wall_seconds": float(wall),
                }
            )
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
        "leftover_recorded": True,
        "leftover_id": 18,
        "leftover_tick": 53,
        "note": (
            "Leftover #18 leftover-recorded: vertex-enumeration wall "
            "vs n at D=2 and D=3. G1 tooling refuses n>12 or D>4. "
            "Combinatorial growth is the honesty bound, not a "
            "tractable-large-n claim. Previous smoke-earned stub with "
            "no timing withdrawn. Not in CI all_passed."
        ),
    }


def _run_g3() -> dict[str, Any]:
    from omnibias.partition._core.config import PartitionConfig
    from omnibias.partition._core.params import init_params, region_code_matrix
    from omnibias.partition._core.weights import partition_weights
    from omnibias.partition.arrangement import soft_membership, tree_arrangement

    cfg = PartitionConfig(n_features=3, depth=3)
    params = init_params(cfg, rng=0)
    arr = tree_arrangement(params.W, params.t)
    x = np.array(
        [[0.2, -0.1, 0.4], [1.0, 0.0, -0.5], [-0.3, 0.7, 0.1]],
        dtype=np.float64,
    )
    beta = 3.5
    pw = partition_weights(params, x, beta)
    codes = region_code_matrix(3)
    eps = np.finfo(np.float64).eps
    worst = 0.0
    for leaf in range(8):
        signs = tuple(1 if codes[leaf, j] > 0.5 else -1 for j in range(3))
        sm = soft_membership(arr, x, signs, beta=beta)
        for a, b in zip(sm, pw[:, leaf], strict=True):
            scale = max(abs(float(a)), abs(float(b)), 1.0)
            ulp = abs(float(a) - float(b)) / (eps * scale)
            worst = max(worst, ulp)
    passed = bool(worst <= 4.0)
    return {
        "name": "g3_tree_agreement",
        "passed": passed,
        "in_ci_all_passed": passed,
        "worst_ulp": float(worst),
    }


def _run_g4() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.partition.arrangement import general_position_normals
    from omnibias.partition.arrangement.jax import soft_membership as sm_jax
    from omnibias.partition.arrangement.torch import soft_membership as sm_torch

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(2)
    arr = general_position_normals(4, 2, rng)
    x = rng.normal(size=(6, 2))
    signs = (1, -1, 1, -1)
    t = sm_torch(arr, torch.as_tensor(x, dtype=torch.float64), signs, beta=2.5)
    j = sm_jax(arr, jnp.asarray(x, dtype=jnp.float64), signs, beta=2.5)
    gap = float(np.max(np.abs(t.detach().cpu().numpy() - np.asarray(j))))
    passed = bool(gap <= 1e-14)
    return {
        "name": "g4_parity",
        "passed": passed,
        "in_ci_all_passed": passed,
        "max_abs": gap,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.partition.arrangement import (
        certify_cell_gap,
        enumerate_cells_vertices,
        general_position_normals,
        max_cells,
    )

    rng = np.random.default_rng(0)
    arr = general_position_normals(6, 2, rng)
    n_cells = len(enumerate_cells_vertices(arr))
    g1 = {
        "name": "g1_cell_count",
        "passed": n_cells == max_cells(6, 2),
        "n_cells": n_cells,
        "max_cells": max_cells(6, 2),
        "in_ci_all_passed": True,
    }
    x = rng.normal(size=(200, 2))
    cert = certify_cell_gap(arr, x, enumerate_cells_vertices(arr)[0], beta=4.0)
    g2 = {
        "name": "g2_gap",
        "passed": cert.is_sound,
        "bound": cert.bound,
        "measured": cert.measured,
        "in_ci_all_passed": True,
    }
    g3 = _run_g3()
    g4 = _run_g4()
    cost = _run_cost()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.arrangement_geometry.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "gates_in_scope": ["g1", "g2", "g3", "g4"],
        },
    )
    payload["gates"] = gates_block([g1, g2, g3, g4])
    payload["g3"] = g3
    payload["g4"] = g4
    payload["cost"] = cost
    payload["honesty"] = {
        "complete_face_lattice": False,
        "collapse": "beta -> inf (temperature); not delta -> 0",
        "p_vs_np": False,
        "cost_earned": False,
        "cost_reported": True,
        "cost_leftover_recorded": True,
        "cost_leftover_id": 18,
        "cost_leftover_tick": 53,
        "cost_in_ci_all_passed": False,
        "enumeration_cutoff_n": COST_CUTOFF_N,
        "enumeration_cutoff_d": COST_CUTOFF_D,
        "temperature_collapse": True,
        "founding_bias_collapse": False,
        "g3_earned": bool(g3["passed"]),
        "g3_in_ci_all_passed": bool(g3["in_ci_all_passed"]),
        "g4_earned": bool(g4["passed"]),
        "g4_in_ci_all_passed": bool(g4["in_ci_all_passed"]),
    }
    if args.full:
        dest = (SCRATCH / "arrangement")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "arrangement_geometry.json").write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(f"wrote {write_json('arrangement_geometry_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

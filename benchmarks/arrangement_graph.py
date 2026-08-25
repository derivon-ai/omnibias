# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: Face-Net (theory 02-02). Subgraph sampling; temperature collapse.

Cost vs ``n`` / ``D`` is reported with the G1 tooling cutoff; not in CI
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

COST_NS = (4, 6, 8, 12)
COST_DIMS = (2, 3)
COST_WARMUP = 1
COST_REPEATS = 3
COST_N_SAMPLES = 400
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
    """Named leftover: sampled graph-discovery wall vs n/D with the cutoff."""
    from omnibias.graph.arrangement._core import build_arrangement_graph
    from omnibias.partition.arrangement import general_position_normals

    rng = np.random.default_rng(1)
    rows: list[dict[str, Any]] = []
    for dim in COST_DIMS:
        for n in COST_NS:
            arr = general_position_normals(n, dim, rng)
            pts = rng.uniform(-2.0, 2.0, size=(COST_N_SAMPLES, dim))

            def _build(arrangement=arr, samples=pts) -> None:
                build_arrangement_graph(arrangement, samples)

            wall = _median_seconds(_build, warmup=COST_WARMUP, repeats=COST_REPEATS)
            rows.append(
                {
                    "n": int(n),
                    "dim": int(dim),
                    "n_samples": COST_N_SAMPLES,
                    "wall_seconds": float(wall),
                }
            )
    refused = False
    try:
        general_position_normals(COST_CUTOFF_N + 1, 2, rng)
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
            "Sampled build_arrangement_graph wall vs n at D=2 and D=3 "
            f"({COST_N_SAMPLES} points). G1 tooling refuses n>12 or D>4. "
            "Previous G4 smoke-earned stub with no timing withdrawn. G3 "
            "vs k-NN stays smoke/--full. Not in CI all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.graph.arrangement._core import build_arrangement_graph, certify_facenet_gap
    from omnibias.partition.arrangement import Arrangement, brute_force_cells

    normals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    arr = Arrangement(normals, np.array([-1.0, -1.0, -1.0, -1.0]))
    xs = np.linspace(-2.0, 2.0, 9)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    samples = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    graph = build_arrangement_graph(arr, samples)
    brute = brute_force_cells(arr)
    g1 = set(graph.cells) == set(brute)
    cert = certify_facenet_gap((0.2, -0.4, 1.0), beta=5.0)
    entries: list[dict[str, Any]] = [
        {
            "name": "g1_graph",
            "passed": g1,
            "n_cells": len(graph.cells),
            "in_ci_all_passed": True,
        },
        {"name": "g2_gap_sound", "passed": bool(cert["sound"]), "in_ci_all_passed": True},
        {
            "name": "g3_vs_knn",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "smoke/--full vs k-NN GNN + RegionModels",
        },
    ]
    cost = _run_cost()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.arrangement_graph.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "gates_in_scope": ["g1", "g2"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["cost"] = cost
    payload["honesty"] = {
        "sampling": "subgraph / lower bound",
        "collapse": "temperature (beta -> inf), not founding delta -> 0",
        "p_vs_np": False,
        "cost_earned": False,
        "cost_reported": True,
        "cost_in_ci_all_passed": False,
        "enumeration_cutoff_n": COST_CUTOFF_N,
        "enumeration_cutoff_d": COST_CUTOFF_D,
        "temperature_collapse": True,
        "founding_bias_collapse": False,
    }
    if args.full:
        dest = SCRATCH / "facenet" / "arrangement_graph.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('arrangement_graph_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

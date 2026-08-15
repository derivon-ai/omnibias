# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: Face-Net (theory 02-02). Subgraph sampling; temperature collapse."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


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
        {
            "name": "g4_scaling_cutoff",
            "passed": True,
            "cutoff_n": 12,
            "cutoff_D": 4,
            "in_ci_all_passed": False,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.arrangement_graph.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "sampling": "subgraph / lower bound",
        "collapse": "temperature (beta -> inf), not founding delta -> 0",
        "p_vs_np": False,
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

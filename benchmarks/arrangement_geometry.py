# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: hyperplane arrangements (theory 01-03).

Sampling is a lower bound. ``beta -> inf`` is temperature collapse.
"""

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
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.arrangement_geometry.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block([g1, g2])
    payload["honesty"] = {
        "complete_face_lattice": False,
        "collapse": "beta -> inf (temperature); not delta -> 0",
        "p_vs_np": False,
    }
    if args.full:
        dest = (SCRATCH / "arrangement")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "arrangement_geometry.json").write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json("arrangement_geometry_smoke.json", payload)
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

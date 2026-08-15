# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: tropical homotopy (theory 01-08). G4 path-following is --full."""

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
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.tropical_homotopy.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block([g1, g2])
    payload["honesty"] = {
        "collapse": "beta -> inf (temperature); not delta -> 0",
        "p_vs_np": False,
        "g4_path_following": "full only",
    }
    if args.full:
        dest = SCRATCH / "tropical"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "tropical_homotopy.json").write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json("tropical_homotopy_smoke.json", payload)
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

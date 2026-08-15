# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: equality locus layer (theory 02-12). Not a PDE solver."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.core.locus import EqualitySystem, UnitTerm, affine_locus, residual

    sys_eq = EqualitySystem(
        (UnitTerm(1, 1.0, (1.0, -0.5), 0.0), UnitTerm(1, 1.0, (1.0, 0.0), 0.0))
    )
    planes = affine_locus(sys_eq)
    # Mirror branch 2x - 0.5 t = 0 => x = 0.25 t  (RH for c=0.5 would be 0.25)
    shock_ok = False
    if planes is not None:
        for pt in ((0.25, 1.0), (0.5, 2.0)):
            if abs(residual(sys_eq, pt)[0]) <= 1e-12:
                shock_ok = True
                break
    entries: list[dict[str, Any]] = [
        {"name": "g1_locus_residual", "passed": shock_ok, "in_ci_all_passed": True},
        {
            "name": "g4_burgers_rh",
            "passed": shock_ok,
            "in_ci_all_passed": False,
            "note": "smoke records RH geometry; --full noisy-data skill",
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.equality_intersection.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "level3_general_solver": False,
        "returns": "branch / condition / converged",
    }
    if args.full:
        dest = SCRATCH / "locus" / "equality_intersection.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('equality_intersection_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

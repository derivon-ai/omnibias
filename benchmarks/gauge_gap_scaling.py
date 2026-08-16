# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent certified gaps at several spacings (evidence, not a limit).

Uses the SU(2) heat-kernel transfer. Not a Yang-Mills / continuum /
uniform-in-a claim.
"""

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

SMOKE_SPACINGS = (1.0, 0.5, 0.25)
SMOKE_COUPLINGS = (0.8, 0.4, 0.2)
FULL_SPACINGS = (1.0, 0.5, 0.25, 0.125)
FULL_COUPLINGS = (0.8, 0.4, 0.2, 0.1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    spacings = FULL_SPACINGS if args.full else SMOKE_SPACINGS
    couplings = FULL_COUPLINGS if args.full else SMOKE_COUPLINGS

    from omnibias.geometry.gauge.transfer.gap import certified_gap_scaling_table
    from omnibias.geometry.gauge.transfer.matrices import su2_heat_kernel_transfer

    report = certified_gap_scaling_table(
        su2_heat_kernel_transfer,
        spacings=spacings,
        couplings=couplings,
        max_dynkin=4,
    )
    points_ok = all(point.spectral_gap_lower > 0.0 for point in report.points)
    honesty = report.continuum_claim is False and "NOT a continuum-limit" in report.note

    entries: list[dict[str, Any]] = [
        {
            "name": "scaling_points_certified",
            "passed": bool(points_ok and len(report.points) >= 3),
            "in_ci_all_passed": True,
        },
        {
            "name": "continuum_claim_false",
            "passed": bool(honesty),
            "in_ci_all_passed": True,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.gauge_gap_scaling.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "builder": "su2_heat_kernel_transfer",
            "spacings": list(spacings),
            "couplings": list(couplings),
        },
    )
    payload["points"] = [
        {
            "lattice_spacing": point.lattice_spacing,
            "coupling": point.coupling,
            "spectral_gap_lower": point.spectral_gap_lower,
            "spectral_gap_lower_per_unit": point.spectral_gap_lower_per_unit,
            "method": point.method,
            "continuum_claim": False,
        }
        for point in report.points
    ]
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "yang_mills": False,
        "mass_gap": False,
        "continuum_claim": False,
        "fixed_matrix": True,
        "note": report.note,
    }
    if args.full:
        dest = SCRATCH / "gauge_gap" / "gauge_gap_scaling.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('gauge_gap_scaling_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

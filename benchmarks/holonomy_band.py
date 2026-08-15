# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: holonomy band (theory 02-14). No YM / mass-gap claim."""

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
    from omnibias.geometry.gauge.band._core import (
        BandRegime,
        classify_regime,
        magnus_truncation_bound,
        open_line_is_gauge_dependent,
    )
    from omnibias.geometry.gauge._core.lie_algebra import su, u1

    g1 = classify_regime(u1(), transverse_constant=False) is BandRegime.ABELIAN
    g1 = g1 and classify_regime(su(2), transverse_constant=False) is BandRegime.PRODUCT
    bound = magnus_truncation_bound(a_norm=0.4, length=1.0, order=2)
    entries: list[dict[str, Any]] = [
        {"name": "g1_regime", "passed": g1, "in_ci_all_passed": True},
        {
            "name": "g3_magnus_bound",
            "passed": bound.lo < 0.0 < bound.hi,
            "in_ci_all_passed": True,
        },
        {
            "name": "g4_open_line_flagged",
            "passed": open_line_is_gauge_dependent() is True,
            "in_ci_all_passed": True,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.holonomy_band.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "closed_form": "abelian and transverse-constant only",
        "open_lines": "gauge-dependent",
        "yang_mills": False,
        "mass_gap": False,
        "continuum_claim": False,
    }
    if args.full:
        dest = SCRATCH / "holonomy" / "holonomy_band.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('holonomy_band_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

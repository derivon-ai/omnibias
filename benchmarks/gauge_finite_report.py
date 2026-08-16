# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sealed finite-gauge report (bundle of existing engines, not a Clay claim).

Runs the locked CI pack. G1 is measured. Not a Yang-Mills / continuum /
mass-gap / Osterwalder-Seiler claim.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    from omnibias.geometry.gauge.transfer.report import (
        FiniteGaugeSpec,
        finite_gauge_report,
    )

    spec = FiniteGaugeSpec(include_torus=bool(args.full))
    report = finite_gauge_report(spec)
    honesty = (
        report.continuum_claim is False
        and report.yang_mills_claim is False
        and report.scaling.continuum_claim is False
    )
    entries: list[dict[str, Any]] = [
        {
            "name": "pack_certified",
            "passed": bool(report.certified),
            "in_ci_all_passed": True,
        },
        {
            "name": "g1_ge_generic_measured",
            "passed": bool(report.g1.ge_generic and report.g1.factor + 1e-12 >= 1.0),
            "in_ci_all_passed": True,
            "measured_factor": report.g1.factor,
            "target_factor_5x": False,
        },
        {
            "name": "continuum_claim_false",
            "passed": bool(honesty),
            "in_ci_all_passed": True,
        },
        {
            "name": "haar_identities",
            "passed": bool(report.haar.certified),
            "in_ci_all_passed": True,
        },
        {
            "name": "su3_gap_certified",
            "passed": bool(report.su3_gap.certified and report.su3_gap.spectral_gap_lower > 0.0),
            "in_ci_all_passed": True,
            "spectral_gap_lower": report.su3_gap.spectral_gap_lower,
        },
        {
            "name": "three_plaquette_gap_certified",
            "passed": bool(
                report.three_plaquette.certified
                and report.three_plaquette.spectral_gap_lower > 0.0
            ),
            "in_ci_all_passed": True,
            "spectral_gap_lower": report.three_plaquette.spectral_gap_lower,
        },
        {
            "name": "wilson_character_domain",
            "passed": bool(report.wilson_character_domain.certified),
            "in_ci_all_passed": True,
            "beta_certified": [
                int(report.wilson_character_domain.beta_certified.numerator),
                int(report.wilson_character_domain.beta_certified.denominator),
            ],
            "grid_exhausted": report.wilson_character_domain.grid_exhausted,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.gauge_finite_report.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "name": spec.name,
            "include_torus": spec.include_torus,
        },
    )
    payload["g1"] = {
        "factor": report.g1.factor,
        "ge_generic": report.g1.ge_generic,
        "generic_gap": report.g1.generic_gap,
        "holonomy_gap": report.g1.holonomy_gap,
        "official_generic": report.g1.official_generic,
        "official_holonomy": report.g1.official_holonomy,
        "target_factor_5x": False,
        "note": report.g1.note,
    }
    payload["domain"] = {
        "beta_certified": [
            int(report.polymer_domain.beta_certified.numerator),
            int(report.polymer_domain.beta_certified.denominator),
        ],
        "beta_outside": [
            int(report.polymer_domain.beta_outside.numerator),
            int(report.polymer_domain.beta_outside.denominator),
        ],
        "counting": report.polymer_domain.counting,
    }
    payload["su3_gap"] = {
        "certified": report.su3_gap.certified,
        "spectral_gap_lower": report.su3_gap.spectral_gap_lower,
        "dimension": report.su3_gap.dimension,
        "n_cells": spec.su3_n_cells,
        "method": report.su3_gap.method,
    }
    payload["three_plaquette"] = {
        "certified": report.three_plaquette.certified,
        "spectral_gap_lower": report.three_plaquette.spectral_gap_lower,
        "dimension": report.three_plaquette.dimension,
        "method": report.three_plaquette.method,
    }
    payload["wilson_domain"] = {
        "beta_certified": [
            int(report.wilson_character_domain.beta_certified.numerator),
            int(report.wilson_character_domain.beta_certified.denominator),
        ],
        "beta_outside": None,
        "grid_exhausted": report.wilson_character_domain.grid_exhausted,
        "quarter_certified": report.wilson_character_domain.quarter_certified,
    }
    payload["honesty"] = {
        "yang_mills": False,
        "mass_gap": False,
        "continuum_claim": False,
        "clay_existence": False,
        "note": report.note,
    }
    payload["gates"] = gates_block(entries)
    if args.full:
        dest = SCRATCH / "gauge_gap" / "gauge_finite_report.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('gauge_finite_report_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

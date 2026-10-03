#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact compactification and finite Dulac-model nonoscillation gates.

The benchmark verifies three declared finite local models and preserves the
open saddle-node-at-infinity obstruction. It does not prove physical
return-map membership, a uniform remainder theorem, finite graphic cyclicity,
or Hilbert's sixteenth problem.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof.certificate import verify_certificate_digest  # noqa: E402
from omnibias.dynamics.compactify import (  # noqa: E402
    verify_poincare_compactification,
    verify_poincare_compactification_formally,
)
from omnibias.dynamics.dulac import (  # noqa: E402
    RationalInterval,
    enclose_dulac_term,
    verify_dulac_cyclicity_formally,
    verify_dulac_uniform_cover_formally,
)
from omnibias.dynamics.graphic import (  # noqa: E402
    certify_graphic_cyclicity,
    named_irrational_hyperbolic_graphic,
    named_open_saddle_node_infinity_graphic,
    named_rational_hyperbolic_graphic,
    named_resonant_homoclinic_graphic,
    verify_graphic_cyclicity,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _q(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _term_payload(term: Any) -> dict[str, object]:
    return {
        "exponent": term.exponent.to_payload(),
        "log_power": term.log_power,
        "coefficient": term.coefficient.to_payload(),
    }


def run(*, run_lean: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    rational = certify_graphic_cyclicity(named_rational_hyperbolic_graphic())
    resonant = certify_graphic_cyclicity(named_resonant_homoclinic_graphic())
    irrational = certify_graphic_cyclicity(named_irrational_hyperbolic_graphic())
    open_case = certify_graphic_cyclicity(
        named_open_saddle_node_infinity_graphic()
    )

    compact_formal = (
        verify_poincare_compactification_formally(rational.compactification)
        if run_lean
        else None
    )
    rational_formal = (
        verify_dulac_cyclicity_formally(rational.rational)
        if run_lean and rational.rational is not None
        else None
    )
    resonant_formal = (
        verify_dulac_cyclicity_formally(resonant.rational)
        if run_lean and resonant.rational is not None
        else None
    )
    irrational_formal = (
        verify_dulac_uniform_cover_formally(irrational.uniform)
        if run_lean and irrational.uniform is not None
        else None
    )

    gd1 = {
        "name": "gd1_exact_poincare_compactification",
        "passed": (
            verify_poincare_compactification(rational.compactification)
            and all(
                chart.equator_invariant
                for chart in rational.compactification.charts[:2]
            )
            and (
                compact_formal.theorem_prover_verified
                if compact_formal is not None
                else True
            )
        ),
        "chart_count": len(rational.compactification.charts),
        "equator_singularity_count_with_chart_overlap": len(
            rational.compactification.singularities
        ),
        "lean_requested": run_lean,
        "lean_verified": bool(
            compact_formal and compact_formal.theorem_prover_verified
        ),
        "scope": "Exact chart coefficients and invariant equator; no graphic capture.",
    }
    gd2 = {
        "name": "gd2_rational_hyperbolic_dulac_model",
        "passed": (
            rational.status == "PROVED_MODEL"
            and rational.upper_bound == 2
            and len(rational.leading_terms) == 3
            and verify_graphic_cyclicity(rational)
        ),
        "ratio": _q(rational.target.saddle.ratio.lo),  # type: ignore[union-attr]
        "upper_bound": rational.upper_bound,
        "leading_terms": [_term_payload(term) for term in rational.leading_terms],
    }
    transformed = (
        () if resonant.rational is None else resonant.rational.transformed.terms
    )
    confluent = any(len(coefficients) > 1 for _, coefficients in transformed)
    gd3 = {
        "name": "gd3_resonant_logarithmic_model",
        "passed": (
            resonant.status == "PROVED_MODEL"
            and resonant.upper_bound == 2
            and confluent
            and verify_graphic_cyclicity(resonant)
        ),
        "ratio": _q(resonant.target.saddle.ratio.lo),  # type: ignore[union-attr]
        "upper_bound": resonant.upper_bound,
        "confluent_polynomial_factor": confluent,
        "leading_terms": [_term_payload(term) for term in resonant.leading_terms],
    }
    compensator_term = next(
        term for term in irrational.leading_terms if term.log_power
    )
    irrational_term_enclosure = enclose_dulac_term(
        compensator_term,
        RationalInterval.create((Fraction(1, 5), Fraction(1, 4))),
    )
    gd4 = {
        "name": "gd4_irrational_ratio_interval_cover",
        "passed": (
            irrational.status == "PROVED_MODEL"
            and irrational.upper_bound == 0
            and irrational.uniform is not None
            and irrational.uniform.status == "PROVED"
            and irrational_term_enclosure.lo > 0.0
            and verify_graphic_cyclicity(irrational)
        ),
        "ratio_enclosure": (
            irrational.target.saddle.ratio.to_payload()  # type: ignore[union-attr]
        ),
        "uniform_bound": irrational.upper_bound,
        "compensator_term_enclosure": [
            irrational_term_enclosure.lo.hex(),
            irrational_term_enclosure.hi.hex(),
        ],
        "cover_leaves": (
            0
            if irrational.uniform is None
            else len(irrational.uniform.tree.leaves())
        ),
        "leading_terms": [_term_payload(term) for term in irrational.leading_terms],
    }
    gd5 = {
        "name": "gd5_finite_lean_replay",
        "passed": (
            not run_lean
            or bool(
                compact_formal
                and compact_formal.theorem_prover_verified
                and rational_formal
                and rational_formal.theorem_prover_verified
                and resonant_formal
                and resonant_formal.theorem_prover_verified
                and irrational_formal
                and irrational_formal.theorem_prover_verified
            )
        ),
        "lean_requested": run_lean,
        "compactification": bool(
            compact_formal and compact_formal.theorem_prover_verified
        ),
        "rational_derivation_chain": bool(
            rational_formal and rational_formal.theorem_prover_verified
        ),
        "resonant_derivation_chain": bool(
            resonant_formal and resonant_formal.theorem_prover_verified
        ),
        "irrational_box_cover": bool(
            irrational_formal and irrational_formal.theorem_prover_verified
        ),
        "formal_scope": (
            "finite rational coefficient identities, signs, and tiling only"
        ),
    }
    honesty = open_case.seal["honesty"]
    gd6 = {
        "name": "gd6_open_case_refusal",
        "passed": (
            open_case.status == "BLOCKED"
            and open_case.upper_bound is None
            and len(open_case.obstruction) == 3
            and verify_graphic_cyclicity(open_case)
            and honesty.get("graphic_finite_cyclicity_proved") is False
            and honesty.get("drr_case_closed") is False
            and honesty.get("full_hilbert16_solved") is False
            and verify_certificate_digest(open_case.seal)
        ),
        "status": open_case.status,
        "obstructions": list(open_case.obstruction),
        "graphic_finite_cyclicity_proved": False,
        "drr_case_closed": False,
        "full_hilbert16_solved": False,
    }
    entries = [gd1, gd2, gd3, gd4, gd5, gd6]
    return {
        **provenance(
            schema="omnibias.benchmarks.dulac_cyclicity.v1",
            config={"lean": run_lean},
        ),
        "gates": dict(gates_block(entries)),
        "physical_return_membership_proved": False,
        "uniform_remainder_proved": False,
        "graphic_finite_cyclicity_proved": False,
        "drr_case_closed": False,
        "full_hilbert16_solved": False,
        "disclaimer": (
            "Exact compactification and zero bounds for declared finite Dulac "
            "models. No physical return-map membership, uniform remainder, "
            "graphic cyclicity, DRR-case closure, or Hilbert-XVI theorem."
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(run_lean=bool(args.lean))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            __import__("json").dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        path = args.output
    else:
        path = write_json("dulac_cyclicity_smoke.json", payload)
    for entry in payload["gates"]["entries"]:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

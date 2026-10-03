#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified instance and finite-family Abelian-integral counts on a cubic.

The gates retain the historical forced-factor replay, then certify a genuine
mixed one-form and a rational coefficient box with one uniform winding bound.
They do not bound the Hilbert number or solve Hilbert's sixteenth problem.
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
from omnibias.core.verified.dfinite_continuation import (  # noqa: E402
    continue_dfinite_path,
)
from omnibias.core.verified.interval import Interval  # noqa: E402
from omnibias.dynamics.abelian import (  # noqa: E402
    AbelianCoefficientBox,
    AbelianProblem,
    build_abelian_evaluator,
    build_abelian_period_evaluator,
    certify_abelian_uniform_cover,
    certify_abelian_zero_count,
    named_cubic_abelian_problem,
    named_genuine_cubic_abelian_problem,
    verify_abelian_uniform_cover,
    verify_abelian_uniform_cover_formally,
    verify_abelian_zero_count,
    verify_abelian_zero_count_formally,
)
from omnibias.holonomic import certify_picard_fuchs, diff_algebra  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _iv_payload(value: Interval | None) -> list[str] | None:
    return None if value is None else [value.lo.hex(), value.hi.hex()]


def _overlap(left: Interval, right: Interval) -> bool:
    return max(left.lo, right.lo) <= min(left.hi, right.hi)


def _settings(full: bool) -> dict[str, int]:
    return {
        "quadrature_panels": 2048 if full else 64,
        "continuation_steps": 32 if full else 6,
        "continuation_order": 20 if full else 12,
        "segments": 32 if full else 8,
        "max_segments": 128 if full else 16,
    }


def _ga1(problem: AbelianProblem) -> dict[str, Any]:
    exact = certify_picard_fuchs(
        problem.p,
        problem.q,
        energy_factor=problem.energy_factor,
    )
    bad = diff_algebra().operator(
        (
            (16,),
            (0, 216),
            (-16, 0, 108),
        )
    )
    rejected = certify_picard_fuchs(
        problem.p,
        problem.q,
        energy_factor=problem.energy_factor,
        candidate=bad,
    )
    passed = exact.verified and exact.seal is not None and not rejected.verified
    return {
        "name": "ga1_exact_picard_fuchs",
        "passed": passed,
        "operator": [
            [[value.numerator, value.denominator] for value in polynomial]
            for polynomial in exact.period_operator.coeffs
        ],
        "matrix_shape": [
            len(exact.determining_matrix),
            len(exact.determining_matrix[0]),
        ],
        "rank_kernel_dimension": len(exact.rank_kernel),
        "perturbed_operator_rejected": not rejected.verified,
    }


def _independent_problem(
    base_h: Fraction,
    roots: tuple[Fraction, Fraction, Fraction],
    settings: dict[str, int],
) -> AbelianProblem:
    return AbelianProblem.create(
        p=-1,
        energy_factor=(Fraction(-1, 64), 0, 1),
        base_h=base_h,
        base_roots=roots,
        quadrature_panels=settings["quadrature_panels"],
        continuation_steps=settings["continuation_steps"],
        continuation_order=settings["continuation_order"],
    )


def _ga2(problem: AbelianProblem, settings: dict[str, int]) -> dict[str, Any]:
    evaluator = build_abelian_evaluator(problem)
    loop = continue_dfinite_path(
        evaluator.picard_fuchs.area_operator.coeffs,
        float(problem.base_h),
        evaluator.initial_data.area_jet,
        (0.04j, 0.04 + 0.04j, 0.04, 0.0),
        steps_per_segment=max(4, settings["continuation_steps"] // 2),
        order=settings["continuation_order"],
    )
    returned = loop[-1].jet
    loop_contains_start = all(
        final.re.lo <= initial.re.lo <= initial.re.hi <= final.re.hi
        and final.im.contains_zero()
        for initial, final in zip(
            evaluator.initial_data.area_jet,
            returned,
            strict=True,
        )
    )

    crosschecks: list[dict[str, object]] = []
    for base_h, roots in (
        (
            Fraction(-120, 343),
            (Fraction(-8, 7), Fraction(3, 7), Fraction(5, 7)),
        ),
        (
            Fraction(120, 343),
            (Fraction(-5, 7), Fraction(-3, 7), Fraction(8, 7)),
        ),
    ):
        independent = build_abelian_evaluator(
            _independent_problem(base_h, roots, settings)
        )
        continued = evaluator.evaluate_real(Interval.point(float(base_h)), 0)
        direct = independent.initial_data.integral_jet[0].re
        crosschecks.append(
            {
                "h": [base_h.numerator, base_h.denominator],
                "continued": _iv_payload(continued),
                "independent_quadrature": _iv_payload(direct),
                "overlap": _overlap(continued, direct),
            }
        )

    passed = loop_contains_start and all(bool(row["overlap"]) for row in crosschecks)
    return {
        "name": "ga2_validated_continuation",
        "passed": passed,
        "closed_loop_contains_initial_jet": loop_contains_start,
        "independent_real_energy_crosschecks": crosschecks,
        "bbjt_overlap": "not_applicable_to_phase_one_depressed_cubic_instance",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--lean", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    full, run_lean = bool(args.full), bool(args.lean)
    settings = _settings(full)
    problem = named_cubic_abelian_problem(
        quadrature_panels=settings["quadrature_panels"],
        continuation_steps=settings["continuation_steps"],
        continuation_order=settings["continuation_order"],
    )
    started = time.perf_counter()
    ga1 = _ga1(problem)
    ga2 = _ga2(problem, settings)
    certificate = certify_abelian_zero_count(
        problem,
        half_width=0.2,
        half_height=0.04,
        segments=settings["segments"],
        max_segments=settings["max_segments"],
    )
    blocked = certify_abelian_zero_count(
        problem,
        half_width=0.125,
        half_height=0.04,
        segments=settings["segments"],
        max_segments=settings["segments"],
    )
    ga3 = {
        "name": "ga3_argument_principle_upper_count",
        "passed": (
            certificate.action_zero_free_domain.verified
            and certificate.upper_status == "PROVED"
            and certificate.upper_count == 2
            and blocked.upper_status == "BLOCKED"
        ),
        "action_zero_free_domain": (
            certificate.action_zero_free_domain.to_payload()
        ),
        "upper_status": certificate.upper_status,
        "upper_count": certificate.upper_count,
        "winding_enclosure": _iv_payload(certificate.winding_enclosure),
        "zero_on_contour_status": blocked.upper_status,
    }
    ga4 = {
        "name": "ga4_krawczyk_lower_count",
        "passed": (
            len(certificate.lower_zeros) == 2
            and all(zero.kappa < 1.0 for zero in certificate.lower_zeros)
        ),
        "lower_count": len(certificate.lower_zeros),
        "boxes": [
            [[lo.hex(), hi.hex()] for lo, hi in zero.enclosure]
            for zero in certificate.lower_zeros
        ],
        "kappa": [zero.kappa for zero in certificate.lower_zeros],
    }
    ga5 = {
        "name": "ga5_exact_count_sandwich",
        "passed": certificate.exact_count == 2,
        "exact_count": certificate.exact_count,
        "scope": (
            "one certified forced-factor two-zero instance in Petrov's "
            "elliptic setting; not Petrov's parameter-uniform sharp theorem"
        ),
    }
    pf_seal = certificate.picard_fuchs.seal
    digest_ok = (
        certificate.seal is not None
        and pf_seal is not None
        and verify_certificate_digest(certificate.seal)
        and verify_certificate_digest(pf_seal)
        and verify_abelian_zero_count(certificate)
    )
    formal = verify_abelian_zero_count_formally(certificate) if run_lean else None
    lean_ok = bool(formal and formal.theorem_prover_verified)
    ga6 = {
        "name": "ga6_seal_and_finite_lean",
        "passed": digest_ok and (lean_ok if run_lean else True),
        "digest_replay": digest_ok,
        "lean_requested": run_lean,
        "picard_fuchs_lean_verified": bool(
            formal and formal.picard_fuchs.verified
        ),
        "winding_integer_lean_verified": bool(
            formal and formal.winding_integer.verified
        ),
        "formal_scope": "finite rational syzygy and winding integer isolation only",
    }
    genuine_problem = named_genuine_cubic_abelian_problem(
        quadrature_panels=settings["quadrature_panels"],
        continuation_steps=settings["continuation_steps"],
        continuation_order=settings["continuation_order"],
    )
    genuine = certify_abelian_zero_count(
        genuine_problem,
        half_width=0.2,
        half_height=0.04,
        root_guesses=(Fraction(-123, 1000), Fraction(123, 1000)),
        root_radius=1e-3,
        segments=settings["segments"],
        max_segments=settings["max_segments"],
    )
    genuine_formal = (
        verify_abelian_zero_count_formally(genuine) if run_lean else None
    )
    genuine_honesty = (
        {} if genuine.seal is None else genuine.seal["honesty"]
    )
    period_evaluator = build_abelian_period_evaluator(genuine_problem)
    area, moment, j0, j1 = period_evaluator.evaluate_state(0.0)
    rank_two_reduction = (
        _overlap(
            area.re,
            Interval.from_rational(Fraction(2, 5)) * j1.re,
        )
        and _overlap(
            moment.re,
            Interval.from_rational(Fraction(2, 21)) * j0.re,
        )
    )
    ga7 = {
        "name": "ga7_genuine_mixed_one_form",
        "passed": (
            genuine.exact_count == 2
            and genuine.seal is not None
            and verify_abelian_zero_count(genuine)
            and bool(genuine_honesty.get("genuine_mixed_one_form"))
            and bool(genuine_honesty.get("declared_factor_roots_excluded"))
            and genuine_honesty.get("forced_factor_instance") is False
            and rank_two_reduction
            and (
                bool(genuine_formal and genuine_formal.theorem_prover_verified)
                if run_lean
                else True
            )
        ),
        "beta": [
            [value.numerator, value.denominator]
            for value in genuine_problem.moment_factor
        ],
        "exact_count": genuine.exact_count,
        "boxes": [
            [[lo.hex(), hi.hex()] for lo, hi in zero.enclosure]
            for zero in genuine.lower_zeros
        ],
        "declared_factor_roots_excluded": genuine_honesty.get(
            "declared_factor_roots_excluded",
            False,
        ),
        "forced_factor_instance": genuine_honesty.get(
            "forced_factor_instance",
            True,
        ),
        "irrational_zeros_verified": False,
        "gauss_manin_differential_rank": 2,
        "period_view_components": 4,
        "exact_period_reduction_replayed": rank_two_reduction,
        "reason": (
            "The certified boxes exclude the two declared planted roots. "
            "Finite-width boxes do not prove irrationality because each contains rationals."
        ),
    }
    coefficient_box = AbelianCoefficientBox.create(
        alpha=(Fraction(-1, 64), 0, 1),
        beta=((Fraction(9, 10_000), Fraction(11, 10_000)),),
    )
    uniform = certify_abelian_uniform_cover(
        genuine_problem,
        coefficient_box,
        half_width=0.2,
        half_height=0.04,
        expected_count=2,
        segments=settings["segments"],
        max_segments=settings["max_segments"],
        max_depth=4,
    )
    uniform_formal = (
        verify_abelian_uniform_cover_formally(uniform)
        if run_lean and uniform.seal is not None
        else None
    )
    ga8 = {
        "name": "ga8_parameter_uniform_box_cover",
        "passed": (
            uniform.status == "PROVED"
            and uniform.uniform_bound == 2
            and uniform.seal is not None
            and verify_abelian_uniform_cover(uniform)
            and (
                bool(
                    uniform_formal
                    and uniform_formal.theorem_prover_verified
                )
                if run_lean
                else True
            )
        ),
        "coefficient_box": coefficient_box.to_payload(),
        "status": uniform.status,
        "uniform_bound": uniform.uniform_bound,
        "leaves": len(uniform.tree.leaves()),
        "box_cover_tiling_lean_verified": bool(
            uniform_formal and uniform_formal.theorem_prover_verified
        ),
        "leaf_winding_lean_verified": bool(
            uniform_formal
            and uniform_formal.leaf_windings
            and all(result.verified for result in uniform_formal.leaf_windings)
        ),
        "scope": (
            "finite rational beta interval for one cubic Hamiltonian and one "
            "regular contour; not a degree-uniform Petrov or H(2) theorem"
        ),
    }
    entries = [ga1, ga2, ga3, ga4, ga5, ga6, ga7, ga8]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")

    refinement_sweep: list[dict[str, object]] = []
    if full:
        by_segments = {settings["segments"]: certificate}
        for segment_count in (16, 32, 64):
            refined = by_segments.get(segment_count)
            if refined is None:
                refined = certify_abelian_zero_count(
                    problem,
                    half_width=0.2,
                    half_height=0.04,
                    segments=segment_count,
                    max_segments=segment_count,
                )
            refinement_sweep.append(
                {
                    "segments": segment_count,
                    "status": refined.upper_status,
                    "count": refined.upper_count,
                    "winding_enclosure": _iv_payload(refined.winding_enclosure),
                    "passed": refined.upper_status == "PROVED"
                    and refined.upper_count == 2,
                }
            )
        if not all(bool(row["passed"]) for row in refinement_sweep):
            raise AssertionError(f"full contour refinement failed: {refinement_sweep}")

    payload = {
        **provenance(
            schema="omnibias.benchmarks.abelian_zero_count.v2",
            config={
                **settings,
                "full": full,
                "lean": run_lean,
                "problem": problem.to_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "certificate_digest": (
            None if certificate.seal is None else certificate.seal["digest"]
        ),
        "picard_fuchs_digest": None if pf_seal is None else pf_seal["digest"],
        "genuine_certificate_digest": (
            None if genuine.seal is None else genuine.seal["digest"]
        ),
        "uniform_cover_digest": (
            None if uniform.seal is None else uniform.seal["digest"]
        ),
        "contour_refinement_sweep": refinement_sweep,
        "disclaimer": (
            "Instance-level and finite rational coefficient-box infinitesimal "
            "Hilbert-XVI counts. Not Petrov's degree-uniform theorem; no H(n), "
            "H(2), singular-passage, or full Hilbert-XVI claim."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    artifact = "abelian_zero_count.json" if full else "abelian_zero_count_smoke.json"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            __import__("json").dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        path = args.output
    elif full:
        destination = SCRATCH / "abelian_zero_count"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified forced-factor Abelian-integral evaluation and zero count."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.abelian import (
    AbelianCoefficientBox,
    AbelianProblem,
    build_abelian_evaluator,
    build_abelian_period_evaluator,
    certify_abelian_uniform_cover,
    certify_abelian_zero_count,
    certify_action_zero_free_rectangle,
    named_cubic_abelian_problem,
    named_genuine_cubic_abelian_problem,
    verify_abelian_uniform_cover,
    verify_abelian_uniform_cover_formally,
    verify_abelian_zero_count,
)


def _midpoint_integral(
    function: Callable[[float], float], panels: int = 100_000
) -> float:
    width = math.pi / panels
    return width * sum(
        function((index + 0.5) * width) for index in range(panels)
    )


def test_turning_point_quadrature_contains_dense_numerical_reference() -> None:
    problem = named_cubic_abelian_problem(
        quadrature_panels=128,
        continuation_steps=6,
        continuation_order=12,
    )
    initial = build_abelian_evaluator(problem).initial_data

    def x(theta: float) -> float:
        return 0.5 + 0.5 * math.cos(theta)

    area = _midpoint_integral(
        lambda theta: 0.5 * math.sin(theta) ** 2 * math.sqrt(x(theta) + 1.0)
    )
    moment = _midpoint_integral(
        lambda theta: (
            0.5
            * x(theta)
            * math.sin(theta) ** 2
            * math.sqrt(x(theta) + 1.0)
        )
    )
    j0 = _midpoint_integral(lambda theta: 2.0 / math.sqrt(x(theta) + 1.0))
    j1 = _midpoint_integral(
        lambda theta: 2.0 * x(theta) / math.sqrt(x(theta) + 1.0)
    )

    assert initial.area.contains(area)
    assert initial.moment.contains(moment)
    assert initial.j0.contains(j0)
    assert initial.j1.contains(j1)


def test_four_component_view_obeys_exact_rank_two_period_reduction() -> None:
    problem = AbelianProblem.create(
        p=-1,
        q=0,
        energy_factor=(1,),
        moment_factor=(1,),
        base_h=0,
        base_roots=(-1, 0, 1),
        quadrature_panels=64,
        continuation_steps=6,
        continuation_order=12,
    )
    evaluator = build_abelian_period_evaluator(problem)
    area, moment, j0, j1 = evaluator.evaluate_state(0.0)
    initial = evaluator.initial_data
    area.re.intersect(initial.area)
    moment.re.intersect(initial.moment)
    assert j0.re.contains(initial.j0.mid)
    assert j1.re.contains(initial.j1.mid)
    area.re.intersect((Interval.from_rational(2) * j1.re) / Interval.from_rational(5))
    moment.re.intersect(Interval.from_rational(Fraction(2, 21)) * j0.re)

    value = evaluator.evaluate_real(Interval.point(0.0), derivative=0)
    derivative = evaluator.evaluate_real(Interval.point(0.0), derivative=1)
    assert value.contains(initial.area.mid + initial.moment.mid)
    assert derivative.contains((initial.j0.mid + initial.j1.mid) / 2)

    target = Interval.from_rational(Fraction(1, 20))
    area, moment, j0, j1 = evaluator.evaluate_state(target)
    area.re.intersect(
        (
            Interval.from_rational(Fraction(3, 20)) * j0.re
            + Interval.from_rational(2) * j1.re
        )
        / Interval.from_rational(5)
    )
    moment.re.intersect(
        Interval.from_rational(Fraction(2, 21)) * j0.re
        + Interval.from_rational(Fraction(3, 140)) * j1.re
    )


def test_rank_two_period_continuation_refuses_a_critical_energy_path() -> None:
    problem = AbelianProblem.create(
        p=-1,
        q=0,
        energy_factor=(1,),
        moment_factor=(1,),
        base_h=0,
        base_roots=(-1, 0, 1),
        quadrature_panels=64,
        continuation_steps=6,
        continuation_order=12,
    )
    evaluator = build_abelian_period_evaluator(problem)
    critical = 2.0 / (3.0 * math.sqrt(3.0))
    with pytest.raises(ZeroDivisionError, match="critical energy"):
        evaluator.evaluate(critical)


def test_named_cubic_has_certified_exact_two_zero_count() -> None:
    problem = named_cubic_abelian_problem(
        quadrature_panels=64,
        continuation_steps=6,
        continuation_order=12,
    )
    certificate = certify_abelian_zero_count(
        problem,
        half_width=0.2,
        half_height=0.04,
        segments=8,
        max_segments=16,
    )

    domain = certify_action_zero_free_rectangle(
        problem,
        contour_center=0j,
        half_width=0.2,
    )
    assert domain.verified
    assert certificate.upper_status == "PROVED"
    assert certificate.action_zero_free_domain == domain
    assert certificate.upper_count == 2
    assert len(certificate.lower_zeros) == 2
    assert certificate.exact_count == 2
    assert certificate.seal is not None
    assert verify_certificate_digest(certificate.seal)
    assert verify_abelian_zero_count(certificate)


def test_genuine_mixed_one_form_has_two_unplanted_certified_zeros() -> None:
    problem = named_genuine_cubic_abelian_problem(
        quadrature_panels=64,
        continuation_steps=6,
        continuation_order=12,
    )
    certificate = certify_abelian_zero_count(
        problem,
        half_width=0.2,
        half_height=0.04,
        root_guesses=(Fraction(-123, 1000), Fraction(123, 1000)),
        root_radius=1e-3,
        segments=8,
        max_segments=16,
    )
    assert problem.moment_factor == (Fraction(1, 1000),)
    assert certificate.upper_status == "PROVED"
    assert certificate.upper_count == certificate.exact_count == 2
    assert certificate.seal is not None
    honesty = certificate.seal["honesty"]
    assert honesty["genuine_mixed_one_form"]
    assert honesty["declared_factor_roots_excluded"]
    assert honesty["forced_factor_instance"] is False
    assert honesty["cubic_gauss_manin_differential_rank_two"]
    assert honesty["four_component_period_view_overcomplete"]
    # Finite-width boxes cannot establish irrationality: every such interval
    # contains rationals. The earned claim is non-factor planting.
    assert honesty["irrational_zeros_verified"] is False
    assert verify_abelian_zero_count(certificate)


def test_interval_coefficients_have_a_uniform_two_zero_upper_bound() -> None:
    problem = named_genuine_cubic_abelian_problem(
        quadrature_panels=64,
        continuation_steps=6,
        continuation_order=12,
    )
    coefficient_box = AbelianCoefficientBox.create(
        alpha=(Fraction(-1, 64), 0, 1),
        beta=((Fraction(9, 10_000), Fraction(11, 10_000)),),
    )
    certificate = certify_abelian_uniform_cover(
        problem,
        coefficient_box,
        segments=8,
        max_segments=16,
        max_depth=2,
    )
    assert certificate.status == "PROVED"
    assert certificate.uniform_bound == 2
    assert len(certificate.tree.leaves()) == 1
    leaf = certificate.tree.leaves()[0]
    assert leaf.winding_seal is not None
    assert certificate.seal is not None
    assert certificate.seal["honesty"]["parameter_uniform_abelian_bound"]
    assert certificate.seal["honesty"]["cubic_gauss_manin_differential_rank_two"]
    assert certificate.seal["honesty"]["four_component_period_view_overcomplete"]
    assert verify_abelian_uniform_cover(certificate)
    formal = verify_abelian_uniform_cover_formally(certificate)
    if formal.box_cover.available:
        assert formal.theorem_prover_verified
        assert len(formal.leaf_windings) == 1
        assert formal.leaf_windings[0].verified
    else:
        assert not formal.theorem_prover_verified
    assert not verify_abelian_uniform_cover(
        replace(
            certificate,
            tree=replace(
                leaf,
                winding_seal={
                    **leaf.winding_seal,
                    "payload": {
                        **leaf.winding_seal["payload"],
                        "integer": 3,
                    },
                },
            ),
        )
    )


def test_blocked_interval_winding_is_bisected_per_leaf() -> None:
    problem = named_genuine_cubic_abelian_problem(
        quadrature_panels=32,
        continuation_steps=4,
        continuation_order=10,
    )
    coefficient_box = AbelianCoefficientBox.create(
        alpha=(Fraction(-1, 64), 0, 1),
        beta=((Fraction(1, 10_000), Fraction(1, 200)),),
    )
    certificate = certify_abelian_uniform_cover(
        problem,
        coefficient_box,
        segments=4,
        max_segments=8,
        max_depth=1,
    )
    assert certificate.status == "BLOCKED"
    assert certificate.uniform_bound is None
    assert not certificate.tree.is_leaf
    assert len(certificate.tree.leaves()) == 2
    assert certificate.seal is None
    with pytest.raises(ValueError, match="beta != 0"):
        AbelianCoefficientBox.create(
            alpha=(1,),
            beta=((Fraction(-1, 10), Fraction(1, 10)),),
        )


def test_contour_through_factor_root_is_blocked() -> None:
    problem = named_cubic_abelian_problem(
        quadrature_panels=64,
        continuation_steps=6,
        continuation_order=12,
    )
    certificate = certify_abelian_zero_count(
        problem,
        half_width=0.125,
        half_height=0.04,
        segments=8,
        max_segments=8,
    )

    assert certificate.upper_status == "BLOCKED"
    assert certificate.upper_count is None
    assert certificate.exact_count is None
    assert certificate.seal is None


def test_rectangle_outside_action_zero_free_strip_is_rejected() -> None:
    problem = named_cubic_abelian_problem(
        quadrature_panels=64,
        continuation_steps=6,
        continuation_order=12,
    )

    with pytest.raises(ValueError, match="zero-free critical strip"):
        certify_abelian_zero_count(
            problem,
            half_width=0.4,
            half_height=0.04,
            segments=8,
            max_segments=8,
        )

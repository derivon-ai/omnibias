# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained-expression algebra: the switching identity and its two gates.

Everything the hard-condition cages rely on reduces to one identity,
``C_k[phi_i] = delta_ki``, plus the two preconditions that make it legitimate:
an invertible support matrix, and (across axes, tested at the cage level)
compatible corner data. These tests pin the identity and both gates in pure
Python, before any tensor is involved.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.pinn._core.constrained import (
    AxisConstraints,
    AxisPlan,
    HardCondition,
    LinearConstraint,
    apply_constraint,
    certify_support_matrix,
    compatibility_sample,
    corner_pairs,
    derivative_at,
    dirichlet,
    face_point,
    group_hard_conditions,
    is_relative,
    neumann,
    periodic,
    projection_cost,
    robin,
    support_matrix,
    support_matrix_condition,
    switching_derivative_coeffs,
)


def _phi(axis: AxisConstraints, i: int, x: float, order: int) -> float:
    """``d^order phi_i / dx^order`` at ``x``, the way a backend evaluates it."""
    coeffs = switching_derivative_coeffs(axis.switching[i], order, axis.length)
    xi = axis.normalize(x)
    return sum(c * xi**k for k, c in enumerate(coeffs))


def _mixed() -> AxisConstraints:
    """Dirichlet at lo, Neumann at hi, Robin at hi -- three kinds on one axis."""
    return AxisConstraints(
        axis=0,
        lo=0.0,
        hi=1.0,
        constraints=(
            dirichlet(0.0),
            neumann(1.0, outward=1.0),
            robin(1.0, alpha=2.0, beta=0.5, outward=1.0),
        ),
    )


# --------------------------------------------------------------------------- #
# The defining identity.
# --------------------------------------------------------------------------- #
def test_switching_functions_are_one_at_their_own_condition_and_zero_at_the_others() -> None:
    axis = _mixed()
    for k, c in enumerate(axis.constraints):
        for i in range(axis.n_constraints):
            got = apply_constraint(c, lambda p, o, _i=i: _phi(axis, _i, p, o))
            assert got == pytest.approx(1.0 if k == i else 0.0, abs=1e-12)


def test_the_constrained_expression_hits_every_target_for_an_arbitrary_free_function() -> None:
    axis = _mixed()
    targets = (0.7, -1.3, 2.9)

    def g(x: float, order: int) -> float:
        # an arbitrary analytic free function, unrelated to the targets
        return math.sin(3.1 * x + 0.4) * 3.1**order * math.cos(order * math.pi / 2) - (
            math.cos(3.1 * x + 0.4) * 3.1**order * math.sin(order * math.pi / 2)
        )

    def u(x: float, order: int) -> float:
        out = g(x, order)
        for i, c in enumerate(axis.constraints):
            rho = targets[i] - apply_constraint(c, g)
            out += _phi(axis, i, x, order) * rho
        return out

    for k, c in enumerate(axis.constraints):
        assert apply_constraint(c, u) == pytest.approx(targets[k], abs=1e-12)


def test_a_second_free_function_gives_the_same_exactness() -> None:
    """Exactness is structural: it must not depend on which free function is used."""
    axis = _mixed()
    targets = (0.7, -1.3, 2.9)

    def g(x: float, order: int) -> float:
        return math.exp(0.6 * x) * 0.6**order

    def u(x: float, order: int) -> float:
        out = g(x, order)
        for i, c in enumerate(axis.constraints):
            out += _phi(axis, i, x, order) * (targets[i] - apply_constraint(c, g))
        return out

    for k, c in enumerate(axis.constraints):
        assert apply_constraint(c, u) == pytest.approx(targets[k], abs=1e-12)


def test_the_identity_survives_a_domain_that_is_not_the_unit_interval() -> None:
    axis = AxisConstraints(
        axis=1,
        lo=-3.0,
        hi=5.0,
        constraints=(dirichlet(-3.0), neumann(5.0), derivative_at(-3.0, 1)),
    )
    for k, c in enumerate(axis.constraints):
        for i in range(axis.n_constraints):
            got = apply_constraint(c, lambda p, o, _i=i: _phi(axis, _i, p, o))
            assert got == pytest.approx(1.0 if k == i else 0.0, abs=1e-10)


def test_a_wave_style_axis_carries_value_and_velocity_at_the_same_point() -> None:
    axis = AxisConstraints(
        axis=0, lo=0.0, hi=2.0, constraints=(dirichlet(0.0), derivative_at(0.0, 1))
    )
    for k, c in enumerate(axis.constraints):
        for i in range(2):
            got = apply_constraint(c, lambda p, o, _i=i: _phi(axis, _i, p, o))
            assert got == pytest.approx(1.0 if k == i else 0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Sign conventions.
# --------------------------------------------------------------------------- #
def test_the_outward_normal_sign_is_folded_into_the_lo_face_coefficient() -> None:
    lo_face = neumann(0.0, outward=-1.0)
    hi_face = neumann(1.0, outward=1.0)
    assert lo_face.terms[0].coef == -1.0
    assert hi_face.terms[0].coef == 1.0
    # A lo-face Neumann therefore constrains -du/dx, which is du/dn there.
    assert apply_constraint(lo_face, lambda p, o: 1.0 if o == 1 else 0.0) == -1.0


def test_robin_needs_a_nonzero_coefficient() -> None:
    with pytest.raises(ValueError, match="alpha or beta nonzero"):
        robin(0.0, alpha=0.0, beta=0.0)


def test_a_constraint_needs_terms_and_a_nonnegative_order() -> None:
    with pytest.raises(ValueError, match="at least one term"):
        LinearConstraint(())
    with pytest.raises(ValueError, match="order must be >= 0"):
        derivative_at(0.0, -1)


def test_axis_bounds_must_be_ordered_and_nonempty() -> None:
    with pytest.raises(ValueError, match="lo < hi"):
        AxisConstraints(axis=0, lo=1.0, hi=1.0, constraints=(dirichlet(1.0),))
    with pytest.raises(ValueError, match="no constraints"):
        AxisConstraints(axis=0, lo=0.0, hi=1.0, constraints=())


# --------------------------------------------------------------------------- #
# Gate 1: the support matrix must be certified invertible.
# --------------------------------------------------------------------------- #
def test_a_well_posed_condition_set_seals_a_verifiable_certificate() -> None:
    axis = _mixed()
    cert = certify_support_matrix(axis)
    assert verify_certificate_digest(cert)
    assert cert["payload"]["type"] == "constrained_expression_support"
    assert cert["payload"]["n_constraints"] == 3
    lo = float.fromhex(cert["payload"]["gram_lambda_min"]["lo"])
    assert lo > 0.0, "invertibility must be certified, not assumed"
    assert cert["honesty"]["unproven_claim"] is False


def test_a_linearly_dependent_condition_set_is_refused() -> None:
    """u(0), u(1) and their sum: no support family can interpolate all three."""
    axis_ok = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), dirichlet(1.0))
    )
    dependent = LinearConstraint(
        (
            *dirichlet(0.0).terms,
            *dirichlet(1.0).terms,
        ),
        label="u(0) + u(1)",
    )
    with pytest.raises(ValueError, match="singular support matrix"):
        AxisConstraints(
            axis=0,
            lo=0.0,
            hi=1.0,
            constraints=(*axis_ok.constraints, dependent),
        )


def test_the_condition_number_separates_a_good_set_from_a_near_dependent_one() -> None:
    good = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), neumann(1.0))
    )
    assert support_matrix_condition(good).hi < 10.0

    # Two Dirichlet points a hair apart: formally independent, numerically not.
    # Squaring into the Gram costs half the digits, so this reports unbounded.
    near = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), dirichlet(1e-9))
    )
    assert support_matrix_condition(near).hi == math.inf

    # A milder separation stays certifiable, and the reported kappa tracks 2/d.
    mild = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), dirichlet(1e-3))
    )
    kappa = support_matrix_condition(mild)
    assert 1e2 < kappa.lo <= kappa.hi < 1e5


def test_a_near_dependent_set_is_refused() -> None:
    near = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), dirichlet(1e-9))
    )
    with pytest.raises(ValueError, match="not certified invertible"):
        certify_support_matrix(near)


def test_the_condition_limit_is_a_second_independent_gate() -> None:
    """A set can be certifiably invertible and still too ill-conditioned to use."""
    mild = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), dirichlet(1e-3))
    )
    certify_support_matrix(mild)  # passes at the default limit
    with pytest.raises(ValueError, match="condition number"):
        certify_support_matrix(mild, condition_limit=10.0)


def test_the_support_matrix_is_the_constraint_operator_applied_to_each_monomial() -> None:
    axis = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), neumann(1.0))
    )
    # s_0 = 1, s_1 = xi. C_0 = u(0) -> [1, 0]; C_1 = u'(1) -> [0, 1].
    assert support_matrix(axis) == [[1.0, 0.0], [0.0, 1.0]]


def test_a_pure_neumann_set_shifts_the_support_family_up_a_degree() -> None:
    """Pure Neumann is well posed, but ``{1, x}`` cannot see it: every constraint
    annihilates the constant, so the family has to start at ``x``."""
    axis = AxisConstraints(
        axis=0,
        lo=0.0,
        hi=1.0,
        constraints=(neumann(0.0, outward=-1.0), neumann(1.0, outward=1.0)),
    )
    assert axis.degrees == (1, 2)
    for k, c in enumerate(axis.constraints):
        for i in range(2):
            got = apply_constraint(c, lambda p, o, _i=i: _phi(axis, _i, p, o))
            assert got == pytest.approx(1.0 if k == i else 0.0, abs=1e-12)
    certify_support_matrix(axis)


def test_a_lone_derivative_condition_skips_the_constant_entirely() -> None:
    axis = AxisConstraints(
        axis=0, lo=0.0, hi=2.0, constraints=(derivative_at(1.0, 2),)
    )
    assert axis.degrees == (2,)
    assert apply_constraint(axis.constraints[0], lambda p, o: _phi(axis, 0, p, o)) == (
        pytest.approx(1.0, abs=1e-12)
    )


def test_projection_points_are_deduplicated() -> None:
    """A face carrying both a value and a slope costs one base evaluation, not two."""
    axis = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), derivative_at(0.0, 1))
    )
    assert axis.projection_points() == (0.0,)


def test_switching_derivatives_terminate_at_the_polynomial_degree() -> None:
    axis = AxisConstraints(
        axis=0, lo=0.0, hi=1.0, constraints=(dirichlet(0.0), neumann(1.0))
    )
    # phi_i are degree <= 1, so the second derivative is identically zero.
    for i in range(2):
        assert switching_derivative_coeffs(axis.switching[i], 2, axis.length) == ()


# --------------------------------------------------------------------------- #
# Relative constraints: two points, neither of them pinned.
# --------------------------------------------------------------------------- #
def test_a_periodic_seam_satisfies_the_same_switching_identity() -> None:
    axis = AxisConstraints(
        axis=0,
        lo=0.0,
        hi=2.0,
        constraints=(periodic(0.0, 2.0, order=0), periodic(0.0, 2.0, order=1)),
    )
    # The constant is invisible to a difference, so the family starts at x.
    assert axis.degrees == (1, 2)
    for k, c in enumerate(axis.constraints):
        for i in range(2):
            got = apply_constraint(c, lambda p, o, _i=i: _phi(axis, _i, p, o))
            assert got == pytest.approx(1.0 if k == i else 0.0, abs=1e-12)
    certify_support_matrix(axis)


def test_a_seam_on_top_of_both_dirichlet_ends_is_refused() -> None:
    """``u(1)-u(0)`` is the difference of two conditions already present."""
    with pytest.raises(ValueError, match="singular support matrix"):
        AxisConstraints(
            axis=0,
            lo=0.0,
            hi=1.0,
            constraints=(dirichlet(0.0), dirichlet(1.0), periodic(0.0, 1.0)),
        )


def test_a_seam_needs_two_distinct_points() -> None:
    with pytest.raises(ValueError, match="two distinct points"):
        periodic(0.5, 0.5)


def test_face_point_is_none_exactly_for_a_relative_constraint() -> None:
    assert face_point(dirichlet(0.25)) == 0.25
    assert face_point(robin(1.0, alpha=1.0, beta=2.0)) == 1.0  # one point, two terms
    assert face_point(periodic(0.0, 1.0)) is None
    assert not is_relative(dirichlet(0.0))
    assert is_relative(periodic(0.0, 1.0))


# --------------------------------------------------------------------------- #
# The pairwise corner enumeration, which is what makes the n-D gate complete.
# --------------------------------------------------------------------------- #
def _plans() -> dict[str, tuple[AxisPlan, ...]]:
    return group_hard_conditions(
        [
            HardCondition("u", 0, dirichlet(0.0), 1.0),
            HardCondition("u", 0, derivative_at(0.0, 1), 2.0),
            HardCondition("u", 1, dirichlet(0.0), 3.0),
            HardCondition("u", 2, dirichlet(0.0), 4.0),
            HardCondition("u", 2, neumann(1.0), 5.0),
        ],
        ((0.0, 1.0),) * 3,
    )


def test_every_pair_spanning_two_axes_is_enumerated_and_no_others() -> None:
    pairs = corner_pairs(_plans())
    # 2 conditions on axis 0, 1 on axis 1, 2 on axis 2 -> 2*1 + 2*2 + 1*2
    assert len(pairs) == 8
    assert all(p.axis_a != p.axis_b for p in pairs)
    assert {(p.axis_a, p.axis_b) for p in pairs} == {(0, 1), (0, 2), (1, 2)}


def test_conditions_sharing_an_axis_are_not_a_corner_pair() -> None:
    """They are embedded together by one support matrix; that is the certificate's job."""
    plans = group_hard_conditions(
        [
            HardCondition("u", 0, dirichlet(0.0), 1.0),
            HardCondition("u", 0, dirichlet(1.0), 2.0),
        ],
        ((0.0, 1.0),) * 2,
    )
    assert corner_pairs(plans) == ()


def test_the_pair_label_names_both_conditions_and_both_axes() -> None:
    pair = corner_pairs(_plans())[0]
    assert "axis 0" in pair.label
    assert "axis 1" in pair.label


def test_components_do_not_pair_across_each_other() -> None:
    plans = group_hard_conditions(
        [
            HardCondition("u", 0, dirichlet(0.0), 1.0),
            HardCondition("v", 1, dirichlet(0.0), 2.0),
        ],
        ((0.0, 1.0),) * 2,
    )
    assert corner_pairs(plans) == ()


# --------------------------------------------------------------------------- #
# The shared sample: identical points on both backends, by construction.
# --------------------------------------------------------------------------- #
def test_the_compatibility_sample_lies_inside_the_bounds_and_is_deterministic() -> None:
    bounds = ((-1.0, 2.0), (0.0, 0.5))
    sample = compatibility_sample(bounds, 12)
    assert len(sample) == 12
    assert sample == compatibility_sample(bounds, 12)
    for row in sample:
        for value, (lo, hi) in zip(row, bounds, strict=True):
            assert lo <= value < hi


def test_the_sample_does_not_collapse_onto_a_diagonal() -> None:
    """A lattice with equal multipliers would test one line through the box."""
    sample = compatibility_sample(((0.0, 1.0), (0.0, 1.0)), 16)
    assert max(abs(a - b) for a, b in sample) > 0.2


def test_the_sample_handles_more_axes_than_the_multiplier_table() -> None:
    sample = compatibility_sample(((0.0, 1.0),) * 15, 4)
    assert len(sample[0]) == 15
    assert len(set(sample[0])) == 15, "every axis must get a distinct offset"


def test_an_empty_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        compatibility_sample(((0.0, 1.0),), 0)


# --------------------------------------------------------------------------- #
# The cost, which is a product rather than a sum -- stated in the docs, so
# measured here rather than trusted.
# --------------------------------------------------------------------------- #
def test_two_conditions_on_one_face_share_a_projection() -> None:
    plans = group_hard_conditions(
        [
            HardCondition("u", 0, dirichlet(0.0), 1.0),
            HardCondition("u", 0, derivative_at(0.0, 1), 2.0),
        ],
        ((0.0, 1.0),),
    )
    assert projection_cost(plans) == 2  # the batch itself, plus one pinned face


def test_a_second_axis_multiplies_rather_than_adds() -> None:
    plans = group_hard_conditions(
        [
            HardCondition("u", 0, dirichlet(0.0), 1.0),
            HardCondition("u", 1, dirichlet(0.0), 2.0),
            HardCondition("u", 1, neumann(1.0), 3.0),
        ],
        ((0.0, 1.0),) * 2,
    )
    # axis 0 has one face, axis 1 has two: 2 * 3, not 2 + 3.
    assert projection_cost(plans) == 6


def test_a_relative_constraint_costs_both_of_its_faces() -> None:
    plans = group_hard_conditions(
        [HardCondition("u", 0, periodic(0.0, 1.0), 0.0)], ((0.0, 1.0),)
    )
    assert projection_cost(plans) == 3


def test_components_needing_the_same_pins_pay_once() -> None:
    """The projected states are cached per pin tuple, not per component."""
    shared = group_hard_conditions(
        [
            HardCondition("u", 0, dirichlet(0.0), 1.0),
            HardCondition("v", 0, dirichlet(0.0), 2.0),
        ],
        ((0.0, 1.0),),
    )
    alone = group_hard_conditions(
        [HardCondition("u", 0, dirichlet(0.0), 1.0)], ((0.0, 1.0),)
    )
    assert projection_cost(shared) == projection_cost(alone)

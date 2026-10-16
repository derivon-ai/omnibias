# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact nonlinear PDE systems, relation discovery, and independent holdouts."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.exp_polynomial import ExpPolynomial
from omnibias.core.rational_field import RationalField, discover_rational_relations
from omnibias.core.verified.exp_polynomial import evaluate_rational_field
from omnibias.core.verified.interval import Interval


def _heat_sum(modes, nu=Fraction(1)):
    dimension = len(modes[0][0]) + 1
    tau = ExpPolynomial.constant(0, dimension)
    for direction, coefficient in modes:
        rates = (nu * sum(Fraction(k)**2 for k in direction), *direction)
        tau = tau + ExpPolynomial.exponential(rates, coefficient)
    return tau


def _velocity(tau, nu=Fraction(1)):
    return tuple(RationalField(tau.differentiate(i), tau).scale(-2 * nu)
                 for i in range(1, tau.dimension))


def _features(velocity, component):
    u, v = velocity
    w = velocity[component]
    one = RationalField.from_polynomial(ExpPolynomial.constant(1, w.dimension))
    return (w.differentiate(0), u * w.differentiate(1), v * w.differentiate(2),
            w.differentiate(1, 2), w.differentiate(2, 2), u, v, one)


def _combine(features, coefficients):
    result = features[0].scale(0)
    for feature, coefficient in zip(features, coefficients, strict=True):
        result = result + feature.scale(coefficient)
    return result


TRAINING_MODES = (((0, 0), 1), ((1, 0), 1), ((0, 2), 1), ((1, 1), 2))
HOLDOUT_MODES = (((0, 0), 1), ((2, -1), 2), ((-1, 1), 3), ((1, 3), 1))


@pytest.mark.parametrize("nu", [Fraction(1), Fraction(3, 2)])
def test_coupled_burgers_identity_with_nonzero_cross_advection(nu) -> None:
    tau = _heat_sum(TRAINING_MODES, nu)
    heat = tau.differentiate(0) - (tau.differentiate(1, 2) + tau.differentiate(2, 2)).scale(nu)
    assert heat.is_zero()
    assert tau.known_global_sign() == 1
    velocity = _velocity(tau, nu)
    for component in range(2):
        features = _features(velocity, component)
        residual = _combine(features, (1, 1, 1, -nu, -nu, 0, 0, 0))
        assert residual.is_zero()
        assert residual.nonzero_conditions
        assert all(guard.known_global_sign() in (-1, 1) for guard in residual.nonzero_conditions)
        assert not features[2 if component == 0 else 1].is_zero()
        # Omitting a coupled advection term genuinely changes this equation.
        assert not (residual - features[2 if component == 0 else 1]).is_zero()
    assert (velocity[0].differentiate(2) - velocity[1].differentiate(1)).is_zero()


def test_exact_discovery_recovers_coupled_pde_and_transfers_without_refitting() -> None:
    training = _velocity(_heat_sum(TRAINING_MODES))
    holdout = _velocity(_heat_sum(HOLDOUT_MODES))
    expected = (Fraction(1), Fraction(1), Fraction(1), Fraction(-1), Fraction(-1),
                Fraction(0), Fraction(0), Fraction(0))
    for component in range(2):
        result = discover_rational_relations((_features(training, component),))
        assert result.candidate_count == 8
        assert result.rank == 7
        assert len(result.relations) == 1
        relation = result.relations[0]
        normalized = tuple(value / relation[0] for value in relation)
        assert normalized == expected
        # These exact coefficients are applied to different modes and weights.
        assert _combine(_features(holdout, component), relation).is_zero()
        assert all(guard.known_global_sign() in (-1, 1)
                   for guards in result.nonzero_conditions for guard in guards)


def test_multirate_scalar_burgers_discovery() -> None:
    tau = _heat_sum((((0,), 1), ((1,), 1), ((2,), 1)))
    u, = _velocity(tau)
    one = RationalField.from_polynomial(ExpPolynomial.constant(1, 2))
    features = (u.differentiate(0), u * u.differentiate(1), u.differentiate(1, 2), u, one)
    result = discover_rational_relations((features,))
    assert result.rank == 4
    assert len(result.relations) == 1
    relation = result.relations[0]
    assert tuple(x / relation[0] for x in relation) == (1, 1, -1, 0, 0)
    assert not (u.differentiate(0) + u * u.differentiate(1) - u.differentiate(1, 2).scale(2)).is_zero()


def test_interval_derivatives_contain_independent_high_precision_coupled_solution() -> None:
    mp = pytest.importorskip("mpmath")
    tau = _heat_sum(TRAINING_MODES)
    velocity = _velocity(tau)
    with mp.workdps(80):
        def direct(component, t, x, y):
            modes = [(mp.mpf(c) * mp.exp((kx*kx + ky*ky)*t + kx*x + ky*y), kx, ky)
                     for (kx, ky), c in TRAINING_MODES]
            return -2 * sum(term * (kx if component == 0 else ky) for term, kx, ky in modes) / sum(
                term for term, _, _ in modes)

        for point in ((0.0, 0.0, 0.0), (0.2, -0.3, 0.1), (-0.1, 0.4, -0.2)):
            box = tuple(Interval.point(x) for x in point)
            exact_point = tuple(mp.mpf(x) for x in point)
            for component, field in enumerate(velocity):
                orders = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 2, 0), (0, 0, 2))
                for multi in orders:
                    derivative = field
                    for axis, order in enumerate(multi):
                        derivative = derivative.differentiate(axis, order)
                    interval = evaluate_rational_field(derivative, box)
                    expected = mp.diff(lambda t, x, y, c=component: direct(c, t, x, y), exact_point, multi)
                    assert mp.mpf(interval.lo) <= expected <= mp.mpf(interval.hi)


def test_cancelled_pde_identity_has_zero_interval_after_global_guard_discharge() -> None:
    velocity = _velocity(_heat_sum(TRAINING_MODES))
    residual = _combine(_features(velocity, 0), (1, 1, 1, -1, -1, 0, 0, 0))
    # No numerical exponential evaluation is needed for this zero identity:
    # positivity of every retained guard is established from exact coefficients.
    assert evaluate_rational_field(residual, (Interval(-1000.0, 1000.0),) * 3) == Interval.point(0.0)


@pytest.mark.parametrize("parameterization", ["amplitude", "bias"])
def test_coupled_identity_and_discovery_for_symbolic_parameters(parameterization) -> None:
    # Coordinates are (t,x,y,a,b,c); PDE differentiation holds amplitudes fixed.
    # Thus coefficient-zero proves the equation for every parameter value on
    # its regular domain, rather than only for selected numerical amplitudes.
    tau = ExpPolynomial.constant(1, 6)
    for parameter, physical_frequency in enumerate(((1, 1, 0), (4, 0, 2), (2, 1, 1)), start=3):
        powers = tuple(int(axis == parameter) for axis in range(6))
        amplitude = (ExpPolynomial.monomial(powers) if parameterization == "amplitude"
                     else ExpPolynomial.exponential(powers))
        tau = tau + amplitude * ExpPolynomial.exponential((*physical_frequency, 0, 0, 0))
    velocity = tuple(RationalField(tau.differentiate(axis), tau).scale(-2) for axis in (1, 2))
    expected = (1, 1, 1, -1, -1, 0, 0, 0)
    for component in range(2):
        features = _features(velocity, component)
        residual = _combine(features, expected)
        assert residual.is_zero()
        assert residual.nonzero_conditions
        result = discover_rational_relations((features,))
        assert result.rank == 7 and len(result.relations) == 1
        relation = result.relations[0]
        assert tuple(value / relation[0] for value in relation) == expected
    # Signed amplitudes can give poles, so the engine must not invent a global
    # positivity claim on the entire six-dimensional parameter space.
    assert tau.known_global_sign() == (None if parameterization == "amplitude" else 1)

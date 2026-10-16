# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""From an exact nonlinear PDE family to a certified global inverse decision."""

from fractions import Fraction

import pytest
from omnibias.core.exp_polynomial import ExpPolynomial
from omnibias.core.rational_field import RationalField
from omnibias.core.verified.exp_polynomial import evaluate_rational_field
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_mean_map import inverse_mean_map
from omnibias.core.verified.linalg import interval_solve
from omnibias.core.verified.mean_map import MeanMap

MODES = ((0, 0), (1, 0), (0, 2), (1, 1))


def _symbolic_velocity(nu):
    # Coordinates (t,x,y,b0,b1,b2,b3) keep every bias free in the identity.
    tau = ExpPolynomial.constant(0, 7)
    for j, (kx, ky) in enumerate(MODES):
        frequency = (nu * (kx*kx + ky*ky), kx, ky,
                     *(int(i == j) for i in range(4)))
        tau += ExpPolynomial.exponential(frequency)
    assert tau.known_global_sign() == 1
    heat = tau.differentiate(0) - (
        tau.differentiate(1, 2) + tau.differentiate(2, 2)).scale(nu)
    assert heat.is_zero()
    velocity = tuple(RationalField(tau.differentiate(axis), tau).scale(-2 * nu)
                     for axis in (1, 2))
    for component in velocity:
        residual = (component.differentiate(0)
                    + velocity[0] * component.differentiate(1)
                    + velocity[1] * component.differentiate(2)
                    - (component.differentiate(1, 2)
                       + component.differentiate(2, 2)).scale(nu))
        assert residual.is_zero()
    return velocity


def test_certified_inverse_selects_unique_input_of_exact_coupled_pde_family():
    pytest.importorskip("mpmath")
    nu, time = Fraction(3, 2), Fraction(1, 4)
    expected = (Fraction(3, 4), Fraction(-1, 2))
    target = (Fraction(1, 2), Fraction(3, 4))
    biases = tuple(-kx*expected[0] - ky*expected[1] - nu*(kx*kx + ky*ky)*time
                   for kx, ky in MODES)
    effective = tuple(b + nu*(kx*kx + ky*ky)*time
                      for b, (kx, ky) in zip(biases, MODES, strict=True))
    model = MeanMap(MODES, effective, target)
    result = inverse_mean_map(model)
    assert result.status == "certified"
    certificate = result.certificate
    assert certificate is not None and certificate.affine_rank == 2
    assert all(iv.contains(float(x)) for iv, x in zip(certificate.enclosure, expected, strict=True))

    # The same exact model gives U=-2 nu mean, with a PDE identity that holds
    # for all inputs and biases, not merely at this certified inverse point.
    velocity = _symbolic_velocity(nu)
    box = (Interval.from_rational(time), *certificate.enclosure,
           *(Interval.from_rational(b) for b in biases))
    for field, mean in zip(velocity, target, strict=True):
        enclosed = evaluate_rational_field(field, box)
        assert enclosed.contains(float(-2 * nu * mean))


def test_nontrivial_inverse_and_sensitivity_match_independent_high_precision():
    mp = pytest.importorskip("mpmath")
    # This q is an exact interior witness; it need not itself be in the
    # two-dimensional exponential family to prove the target is attainable.
    weights = (Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(4, 10))
    target = tuple(sum(q*k[axis] for q, k in zip(weights, MODES, strict=True))
                   for axis in range(2))
    biases = (Fraction(1, 3), Fraction(-2, 5), Fraction(3, 7), Fraction(1, 9))
    model = MeanMap(MODES, biases, target)
    assert model.target_interior(weights)
    result = inverse_mean_map(model)
    assert result.status == "certified" and result.certificate is not None
    enclosure = result.certificate.enclosure
    covariance = model.jacobian(enclosure)
    inverse_columns = [interval_solve(covariance, [int(i == j) for i in range(2)])
                       for j in range(2)]
    lower = model.dual_bound(weights)

    with mp.workdps(80):
        def exact(q):
            return mp.mpf(q.numerator) / q.denominator

        def mean(x, y):
            terms = [mp.exp(kx*x + ky*y + exact(b))
                     for (kx, ky), b in zip(MODES, biases, strict=True)]
            return tuple(sum(w*k[axis] for w, k in zip(terms, MODES, strict=True))/sum(terms)
                         for axis in range(2))

        root = mp.findroot(lambda x, y: tuple(v - exact(m) for v, m in
                                            zip(mean(x, y), target, strict=True)), (0, 0))
        for interval, value in zip(enclosure, root, strict=True):
            assert mp.mpf(interval.lo) <= value <= mp.mpf(interval.hi)
        exact_jacobian = mp.matrix([[mp.diff(lambda x, y, a=axis: mean(x, y)[a],
                                            tuple(root), tuple(int(i == j) for i in range(2)))
                                    for j in range(2)] for axis in range(2)])
        exact_inverse = exact_jacobian**-1
        for j, column in enumerate(inverse_columns):
            for i, interval in enumerate(column):
                # D_target(input*) = covariance(input*)^-1 by implicit differentiation.
                assert mp.mpf(interval.lo) <= exact_inverse[i, j] <= mp.mpf(interval.hi)
        minimum = mp.log(sum(mp.exp(kx*root[0] + ky*root[1] + exact(b))
                             for (kx, ky), b in zip(MODES, biases, strict=True))) - sum(
                                 exact(m)*x for m, x in zip(target, root, strict=True))
        assert mp.mpf(lower.lo) <= minimum
        objective = model.objective(enclosure)
        assert mp.mpf(objective.lo) <= minimum <= mp.mpf(objective.hi)


def test_boundary_and_exterior_pde_targets_never_gain_a_finite_inverse_claim():
    pytest.importorskip("mpmath")
    # Scalar velocity lies strictly between -2 and 0 for these two modes.
    for target in (0, 1, 2, -1):
        result = inverse_mean_map(MeanMap(((0,), (1,)), (0, 0), (target,)), max_iterations=24)
        assert result.status == "not_certified"
        assert result.certificate is None


def test_unequal_probabilities_give_tight_primal_dual_global_value_enclosure():
    mp = pytest.importorskip("mpmath")
    target = (Fraction(2, 3), Fraction(2, 3))
    q = (Fraction(1, 6), Fraction(1, 3), Fraction(1, 6), Fraction(1, 3))
    model = MeanMap(MODES, (0, 0, 0, 0), target)
    result = inverse_mean_map(model, radii=(1e-8,))
    assert result.status == "certified" and result.certificate is not None
    lower = model.dual_bound(q)
    upper = model.objective(tuple(Interval.point(x) for x in result.point))
    # q is precisely the softmax distribution at (log(2),0); hence no
    # optimization duality gap remains apart from numerical enclosure widths.
    with mp.workdps(80):
        expected = (mp.log(2), mp.mpf(0))
        for interval, value in zip(result.certificate.enclosure, expected, strict=True):
            assert mp.mpf(interval.lo) <= value <= mp.mpf(interval.hi)
        minimum = mp.log(6) - mp.mpf(2)/3 * mp.log(2)
        assert mp.mpf(lower.lo) <= minimum <= mp.mpf(upper.hi)
    assert (Interval.point(upper.hi) - Interval.point(lower.lo)).hi < 1e-10


def test_affine_gauge_prevents_unique_input_claim_despite_unique_output():
    # Dependence on x+y alone leaves an unconstrained input direction.
    model = MeanMap(((0, 0), (1, 1), (2, 2)), (0, 0, 0), (1, 1))
    result = inverse_mean_map(model)
    assert model.affine_rank == 1
    assert result.status == "rank_deficient" and result.certificate is None

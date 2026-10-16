# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Gaussian heat PDE lift, sharp threshold, and certified decisions."""

from fractions import Fraction

import pytest
from omnibias.core.exp_polynomial import ExpPolynomial
from omnibias.core.rational_field import RationalField
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import inverse_gaussian_heat


@pytest.mark.parametrize("centers", [((-1,), (0,), (2,)), ((0, 0), (1, 0), (0, 2), (1, 1))])
def test_exact_gaussian_heat_lift_solves_coupled_burgers(centers):
    nu = Fraction(3, 2)
    dimension = len(centers[0])
    # Lift coordinates (y_1,...,y_d,z)=(x/s,1/s). Positive integer
    # amplitudes are sufficient for this exact identity replay.
    tau = ExpPolynomial.constant(0, dimension+1)
    for index, center in enumerate(centers):
        rates = tuple(Fraction(value)/(2*nu) for value in center) + (-sum(Fraction(a)**2 for a in center)/(4*nu),)
        tau += ExpPolynomial.exponential(rates, index+1)
    heat_residual = tau.differentiate(dimension)
    for axis in range(dimension):
        heat_residual += tau.differentiate(axis, 2).scale(nu)
    assert heat_residual.is_zero() and tau.known_global_sign() == 1
    coordinates = tuple(RationalField.from_polynomial(ExpPolynomial.monomial(
        tuple(int(j == axis) for j in range(dimension+1)))) for axis in range(dimension+1))
    ys, z = coordinates[:-1], coordinates[-1]
    velocity = tuple(y-z*RationalField(tau.differentiate(axis), tau).scale(2*nu)
                     for axis, y in enumerate(ys))

    def dx(field, axis):
        return z*field.differentiate(axis)

    def dt(field):
        result = -(z*z)*field.differentiate(dimension)
        for axis, y in enumerate(ys):
            result -= z*y*field.differentiate(axis)
        return result

    for component in velocity:
        residual = dt(component)
        for axis, advector in enumerate(velocity):
            residual += advector*dx(component, axis)-dx(dx(component, axis), axis).scale(nu)
        assert residual.is_zero()
        assert residual.nonzero_conditions
        assert all(guard.known_global_sign() == 1 for guard in residual.nonzero_conditions)
        # Coordinate dependence cannot be ignored when applying time derivatives.
        assert not component.differentiate(dimension).is_zero()
        assert (dt(dx(component, 0))-dx(dt(component), 0)).is_zero()


def test_diameter_threshold_is_sharp_for_uniform_smooth_invertibility():
    mp = pytest.importorskip("mpmath")
    before = GaussianHeatMap(((-1,), (1,)), (0, 0), time=Fraction(1, 4))
    after = GaussianHeatMap(((-1,), (1,)), (0, 0), time=1)
    assert before.contraction_bound == 2 and after.contraction_bound == Fraction(1, 2)
    assert before.scaled_jacobian((Interval.point(0.0),))[0][0].hi < 0
    assert after.scaled_jacobian((Interval.point(0.0),))[0][0].lo > 0
    with mp.workdps(80):
        positive = mp.findroot(lambda x: x-mp.tanh(2*x), mp.mpf("0.9"))
        assert 0 < positive < 1
        for root in (-positive, mp.mpf(0), positive):
            assert abs(root-mp.tanh(2*root)) < mp.mpf("1e-75")
    refused = inverse_gaussian_heat(before, (0,))
    assert refused.status == "outside_contraction_regime"
    certified = inverse_gaussian_heat(after, (0,))
    assert certified.status == "certified" and certified.tolerance_met


def test_coupled_inverse_encloses_independent_gaussian_root():
    mp = pytest.importorskip("mpmath")
    centers = ((0, 0), (1, 0), (0, 2), (1, 1))
    model = GaussianHeatMap(centers, (0, 1, -1, 0), viscosity=Fraction(3, 2), time=2)
    target = (Fraction(1, 3), Fraction(-1, 4))
    result = inverse_gaussian_heat(model, target, absolute_tolerance=1e-10)
    assert result.tolerance_met and result.certificate is not None
    with mp.workdps(80):
        def residual(x, y):
            weights = [mp.exp(b-((x-a[0])**2+(y-a[1])**2)/12)
                       for a, b in zip(centers, (0, 1, -1, 0), strict=True)]
            mean = [sum(p*a[j] for p, a in zip(weights, centers, strict=True))/sum(weights) for j in range(2)]
            return ((x-mean[0])/2-mp.mpf(1)/3, (y-mean[1])/2+mp.mpf(1)/4)
        root = mp.findroot(residual, (1, 0))
        for interval, exact in zip(result.certificate.enclosure, root, strict=True):
            assert mp.mpf(interval.lo) <= exact <= mp.mpf(interval.hi)

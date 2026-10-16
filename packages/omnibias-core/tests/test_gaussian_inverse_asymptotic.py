# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent checks of uniform physical Gaussian inverse predictor bounds.

For q=D²/(8*nu*s)<1, let p_infinity be proportional to
exp(b+a.v/(2*nu)), with mean mu and third trace T. These tests audit
||X-(s*v+mu)||<=D*q/2 and ||X-(s*v+mu-T/(4*nu*s))||<=D*q².
They are validation of an analytic argument, not an interval certificate API.
"""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_heat import GaussianHeatMap

FAMILIES = (
    (((F(-1, 2),), (F(1, 4),), (F(3, 2),)), (F(1, 5), F(-2, 7), F(3, 11)),
     F(3, 2), ((F(0),), (F(1, 3),), (F(-2, 5),), (F(10),), (F(-13),))),
    (((F(-1), F(0)), (F(1), F(1)), (F(0), F(2)), (F(2), F(-1))),
     (F(1, 5), F(-2, 7), F(3, 11), F(-1, 4)), F(2),
     ((F(0), F(0)), (F(1, 3), F(-2, 5)), (F(-3, 7), F(2, 9)), (F(10), F(-13)), (F(-9), F(7)))),
)


def _mp(mp, value):
    exact = F(value)
    return mp.mpf(exact.numerator)/exact.denominator


def _diameter_squared(centers):
    return max(sum((x-y)**2 for x, y in zip(a, b, strict=True)) for a in centers for b in centers)


def _mean(mp, centers, logits):
    weights = [mp.exp(value-max(logits)) for value in logits]
    probabilities = [weight/sum(weights) for weight in weights]
    mean = tuple(sum(p*a[axis] for p, a in zip(probabilities, centers, strict=True))
                 for axis in range(len(centers[0])))
    return probabilities, mean


def _prediction(mp, model, target):
    centers = [tuple(_mp(mp, value) for value in row) for row in model.centers]
    v = tuple(_mp(mp, value) for value in target)
    nu, age = _mp(mp, model.viscosity), _mp(mp, model.time)
    logits = [_mp(mp, b)+sum(a*x for a, x in zip(row, v, strict=True))/(2*nu)
              for b, row in zip(model.biases, centers, strict=True)]
    probabilities, mean = _mean(mp, centers, logits)
    deviations = [tuple(a-mu for a, mu in zip(row, mean, strict=True)) for row in centers]
    third = tuple(sum(p*row[axis]*sum(x*x for x in row) for p, row in zip(probabilities, deviations, strict=True))
                  for axis in range(model.dimension))
    zeroth = tuple(age*x+mu for x, mu in zip(v, mean, strict=True))
    first = tuple(x-t/(4*nu*age) for x, t in zip(zeroth, third, strict=True))
    return centers, v, logits, mean, third, zeroth, first


def _physical_root(mp, model, target, initial):
    centers = [tuple(_mp(mp, value) for value in row) for row in model.centers]
    nu, age = _mp(mp, model.viscosity), _mp(mp, model.time)
    v = tuple(_mp(mp, value) for value in target)

    def residual(*point):
        # Direct original Gaussian mixture, independently of the predictor's
        # limiting exponential weights and all implementation proposal solvers.
        logits = [_mp(mp, b)-sum((x-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
                  for b, row in zip(model.biases, centers, strict=True)]
        _, mean = _mean(mp, centers, logits)
        return tuple((x-mu)/age-wanted for x, mu, wanted in zip(point, mean, v, strict=True))

    root = tuple(mp.findroot(residual, initial, tol=mp.mpf("1e-80"), maxsteps=80))
    assert mp.norm(residual(*root)) < mp.mpf("1e-75")
    return root


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("q", [F(1, 100), F(1, 10), F(1, 2), F(99, 100)])
def test_uniform_predictor_bounds_for_asymmetric_scalar_and_coupled_families(family, q):
    mp = pytest.importorskip("mpmath")
    centers, biases, viscosity, targets = family
    squared = _diameter_squared(centers)
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=squared/(8*viscosity*q))
    assert model.contraction_bound == q
    with mp.workdps(90):
        diameter = mp.sqrt(_mp(mp, squared))
        zeroth_bound, first_bound = diameter*_mp(mp, q)/2, diameter*_mp(mp, q*q)
        for target in targets:
            _, _, _, _, _, zeroth, first = _prediction(mp, model, target)
            root = _physical_root(mp, model, target, zeroth)
            assert mp.norm(tuple(x-y for x, y in zip(root, zeroth, strict=True))) <= zeroth_bound
            assert mp.norm(tuple(x-y for x, y in zip(root, first, strict=True))) <= first_bound


def test_frozen_hull_tilt_derivatives_obey_the_proof_constants():
    mp = pytest.importorskip("mpmath")
    centers, biases, viscosity, targets = FAMILIES[1]
    squared = _diameter_squared(centers)
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=squared/(8*viscosity*F(4, 5)))
    with mp.workdps(90):
        centers_mp, target, logits, _, _, zeroth, _ = _prediction(mp, model, targets[1])
        root = _physical_root(mp, model, targets[1], zeroth)
        age, nu = _mp(mp, model.time), _mp(mp, viscosity)
        frozen_mean = tuple(x-age*v for x, v in zip(root, target, strict=True))
        distances = [sum((a-mu)**2 for a, mu in zip(row, frozen_mean, strict=True)) for row in centers_mp]
        diameter = mp.sqrt(_mp(mp, squared))
        assert all(0 <= distance <= diameter**2 for distance in distances)

        def tilt_mean(theta):
            return _mean(mp, centers_mp, [value-theta*distance/(4*nu*age)
                                          for value, distance in zip(logits, distances, strict=True)])[1]

        for theta in (mp.mpf(0), mp.mpf("0.5"), mp.mpf(1)):
            first = tuple(mp.diff(lambda t, axis=axis: tilt_mean(t)[axis], theta) for axis in range(model.dimension))
            second = tuple(mp.diff(lambda t, axis=axis: tilt_mean(t)[axis], theta, 2) for axis in range(model.dimension))
            assert mp.norm(first) <= diameter**3/(16*nu*age)
            assert mp.norm(second) <= diameter**5/(64*nu**2*age**2)


def test_large_time_correction_sign_and_physical_scaling_are_essential():
    mp = pytest.importorskip("mpmath")
    q = F(1, 10000)
    model = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), viscosity=F(3, 2), time=1/(3*q))
    with mp.workdps(90):
        _, _, _, mean, third, zeroth, first = _prediction(mp, model, (F(2, 5),))
        root = _physical_root(mp, model, (F(2, 5),), zeroth)
        assert abs(third[0]) > mp.mpf("0.01")
        signed_coefficient = (root[0]-zeroth[0])*_mp(mp, model.time)
        expected_coefficient = -third[0]/(4*_mp(mp, model.viscosity))
        assert abs(signed_coefficient-expected_coefficient) < abs(expected_coefficient)/1000
        assert abs(root[0]-first[0]) <= 2*_mp(mp, q*q)
        # Returning mu_infinity alone would approximate a posterior mean;
        # the inverse of physical U needs the growing affine s*v term.
        assert abs(root[0]-mean[0]) > 1000


def test_identical_centers_make_both_predictors_exact():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((2, -3), (2, -3)), (F(1, 3), F(-5, 7)), viscosity=2, time=F(7, 3))
    with mp.workdps(90):
        _, _, _, _, third, zeroth, first = _prediction(mp, model, (F(3, 5), F(-2, 7)))
        expected = (_mp(mp, F(17, 5)), _mp(mp, F(-11, 3)))
        assert mp.norm(third) < mp.mpf("1e-85")
        assert mp.norm(tuple(x-y for x, y in zip(zeroth, expected, strict=True))) < mp.mpf("1e-85")
        assert mp.norm(tuple(x-y for x, y in zip(first, expected, strict=True))) < mp.mpf("1e-85")

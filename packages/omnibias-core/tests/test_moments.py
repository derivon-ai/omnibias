# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact moment <-> cumulant identities and the analytic delta method."""

from __future__ import annotations

import random
from fractions import Fraction

from omnibias.core.moments import (
    central_moments_from_cumulants,
    central_to_raw_moments,
    cumulants_from_raw_moments,
    delta_method_central_moments,
    gaussian_central_moments,
    raw_moments_from_cumulants,
    raw_to_central_moments,
    second_order_delta,
)


def test_standard_normal_raw_moments() -> None:
    # cumulants of N(0,1): kappa_2 = 1, all others 0.
    kappa = [0, 1, 0, 0, 0, 0]
    assert raw_moments_from_cumulants(kappa) == [0, 1, 0, 3, 0, 15]


def test_gaussian_central_moments_match_double_factorial() -> None:
    var = Fraction(7, 3)
    central = central_moments_from_cumulants([Fraction(5), var, 0, 0, 0, 0])
    expected = gaussian_central_moments(var, 6)
    assert central == expected
    assert central == [0, var, 0, 3 * var**2, 0, 15 * var**3]


def test_moment_cumulant_roundtrip_exact() -> None:
    rng = random.Random(0)
    for _ in range(50):
        kappa = [Fraction(rng.randint(-5, 5), rng.randint(1, 4)) for _ in range(6)]
        raw = raw_moments_from_cumulants(kappa)
        assert cumulants_from_raw_moments(raw) == kappa


def test_raw_central_roundtrip_exact() -> None:
    rng = random.Random(1)
    for _ in range(50):
        raw = [Fraction(rng.randint(-5, 5), rng.randint(1, 4)) for _ in range(6)]
        mean = raw[0]
        central = raw_to_central_moments(raw, mean)
        assert central[0] == 0  # first central moment vanishes
        assert central_to_raw_moments(central, mean) == raw


def test_delta_method_linear_is_exact() -> None:
    # f(x) = a x + b: output moments are an exact affine image of the input.
    a, b, mu, var = Fraction(3), Fraction(-2), Fraction(5, 2), Fraction(4, 3)
    derivs = [a * mu + b, a, Fraction(0), Fraction(0), Fraction(0)]
    central_in = gaussian_central_moments(var, 4)
    out = delta_method_central_moments(derivs, central_in, order=4)
    assert out["mean"] == a * mu + b
    assert out["variance"] == a**2 * var
    # nu_p(Y) = a^p mu_p(X)
    assert out["central"] == [a**p * central_in[p - 1] for p in (2, 3, 4)]


def test_delta_method_square_is_exact() -> None:
    # f(x) = x^2 with Gaussian input: E[Y] = mu^2 + var, Var[Y] = 2 var^2 + 4 mu^2 var.
    mu, var = Fraction(3, 2), Fraction(5, 4)
    derivs = [mu**2, 2 * mu, Fraction(2), Fraction(0), Fraction(0)]
    central_in = gaussian_central_moments(var, 4)
    out = delta_method_central_moments(derivs, central_in, order=4)
    assert out["mean"] == mu**2 + var
    assert out["variance"] == 2 * var**2 + 4 * mu**2 * var


def test_second_order_delta_linear_reference() -> None:
    # Linear map: mean correction is zero, covariance is J^T Sigma J exactly.
    value = [Fraction(0), Fraction(0)]
    jac = [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]]  # jac[i][c]
    cov = [[Fraction(2), Fraction(0)], [Fraction(0), Fraction(5)]]
    out_mean, out_cov = second_order_delta(value, jac, None, cov)
    assert out_mean == value
    # Cov_cd = sum_ij jac[i][c] cov[i][j] jac[j][d]
    assert out_cov[0][0] == 1 * 2 * 1 + 3 * 5 * 3
    assert out_cov[0][1] == 1 * 2 * 2 + 3 * 5 * 4
    assert out_cov[1][1] == 2 * 2 * 2 + 4 * 5 * 4

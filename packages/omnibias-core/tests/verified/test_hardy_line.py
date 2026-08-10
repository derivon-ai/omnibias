# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified Cauchy-Hardy line Hilbert pair (generalized Poisson).

Soundness: every enclosure contains a dense deterministic grid and a random
sample of true values; the Hilbert rotation is cross-checked against
independent high-precision PV quadrature.
"""

from __future__ import annotations

import math
import random

import mpmath as mp
import pytest
from omnibias.core.verified import (
    conjugate_poisson,
    conjugate_poisson_deriv,
    hardy_even,
    hardy_even_deriv,
    hardy_even_profile,
    hardy_odd,
    hardy_odd_deriv,
    hardy_pair,
    hardy_tail_constant,
    hilbert_hardy_even_profile,
    hilbert_of_hardy_even,
    hilbert_of_hardy_odd,
    poisson_kernel,
    poisson_kernel_deriv,
)


def _true_p(y: float, a: float, alpha: float) -> float:
    r = math.hypot(a, y)
    phi = math.atan(y / a)
    return (r ** (-alpha)) * math.cos(alpha * phi)


def _true_q(y: float, a: float, alpha: float) -> float:
    r = math.hypot(a, y)
    phi = math.atan(y / a)
    return (r ** (-alpha)) * math.sin(alpha * phi)


def _mp_p(t: object, a: float, alpha: float) -> object:
    r = mp.sqrt(mp.mpf(a) ** 2 + t * t)
    phi = mp.atan(t / mp.mpf(a))
    return (r ** (-mp.mpf(alpha))) * mp.cos(mp.mpf(alpha) * phi)


def _mp_q(t: object, a: float, alpha: float) -> object:
    r = mp.sqrt(mp.mpf(a) ** 2 + t * t)
    phi = mp.atan(t / mp.mpf(a))
    return (r ** (-mp.mpf(alpha))) * mp.sin(mp.mpf(alpha) * phi)


def _pv_hilbert_hardy(f: object, x0: float, a: float, alpha: float) -> float:
    assert callable(f)
    with mp.workdps(40):
        fx0 = f(mp.mpf(x0), a, alpha)

        def integrand(t: object) -> object:
            return (f(t, a, alpha) - fx0) / (mp.mpf(x0) - t)

        val = mp.quad(integrand, [mp.ninf, x0, mp.inf]) / mp.pi
        return float(val)


_GRID_Y = [-5.0, -2.0, -0.7, -0.1, 0.0, 0.15, 0.9, 1.5, 4.0, 12.0]
_GRID_A = [0.4, 1.0, 2.5]
_GRID_ALPHA = [0.55, 0.623, 0.75, 1.0]


@pytest.mark.parametrize("alpha", _GRID_ALPHA)
@pytest.mark.parametrize("a", _GRID_A)
@pytest.mark.parametrize("y", _GRID_Y)
def test_hardy_pair_encloses_truth_dense_grid(y: float, a: float, alpha: float) -> None:
    p, q = hardy_pair(y, a, alpha)
    assert p.contains(_true_p(y, a, alpha))
    assert q.contains(_true_q(y, a, alpha))
    assert hardy_even_deriv(y, a, alpha).contains(
        -alpha * _true_q(y, a, alpha + 1.0)
    )
    assert hardy_odd_deriv(y, a, alpha).contains(
        alpha * _true_p(y, a, alpha + 1.0)
    )


def test_hardy_pair_encloses_random_sample() -> None:
    rng = random.Random(17)
    for _ in range(40):
        y = rng.uniform(-8.0, 8.0)
        a = rng.uniform(0.2, 3.0)
        alpha = rng.uniform(0.51, 1.2)
        p, q = hardy_pair(y, a, alpha)
        assert p.contains(_true_p(y, a, alpha))
        assert q.contains(_true_q(y, a, alpha))


def test_alpha_one_recovers_poisson() -> None:
    for a in (0.5, 1.3, 2.0):
        for y in (-2.0, 0.0, 0.7, 5.0):
            assert hardy_even(y, a, 1.0).contains(float(poisson_kernel(y, a).mid))
            assert hardy_odd(y, a, 1.0).contains(float(conjugate_poisson(y, a).mid))
            # Derivatives match Poisson closed forms at alpha=1.
            assert hardy_even_deriv(y, a, 1.0).contains(
                float(poisson_kernel_deriv(y, a).mid)
            )
            assert hardy_odd_deriv(y, a, 1.0).contains(
                float(conjugate_poisson_deriv(y, a).mid)
            )


@pytest.mark.parametrize("alpha", [0.623, 1.0])
@pytest.mark.parametrize("a", [0.8, 1.5])
@pytest.mark.parametrize("x0", [-2.0, 0.4, 3.0])
def test_hilbert_rotation_matches_pv(x0: float, a: float, alpha: float) -> None:
    h_p = _pv_hilbert_hardy(_mp_p, x0, a, alpha)
    assert hilbert_of_hardy_even(x0, a, alpha).contains(h_p)
    assert abs(h_p - _true_q(x0, a, alpha)) < 1e-9
    h_q = _pv_hilbert_hardy(_mp_q, x0, a, alpha)
    assert hilbert_of_hardy_odd(x0, a, alpha).contains(h_q)
    assert abs(h_q + _true_p(x0, a, alpha)) < 1e-9


def test_hardy_profile_and_tail() -> None:
    coeffs = [1.0, -0.3, 0.1]
    scales = [0.6, 1.2, 2.0]
    alpha = 0.623
    y = 0.5
    th = hardy_even_profile(y, coeffs, scales, alpha)
    truth = sum(c * _true_p(y, a, alpha) for c, a in zip(coeffs, scales, strict=True))
    assert th.contains(truth)
    hth = hilbert_hardy_even_profile(y, coeffs, scales, alpha)
    h_truth = sum(c * _true_q(y, a, alpha) for c, a in zip(coeffs, scales, strict=True))
    assert hth.contains(h_truth)
    c, p = hardy_tail_constant(coeffs, scales, alpha)
    assert p == alpha
    assert c >= sum(abs(x) for x in coeffs) - 1e-15


def test_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError):
        hardy_even(0.0, 0.0, 0.6)
    with pytest.raises(ValueError):
        hardy_even(0.0, -1.0, 0.6)

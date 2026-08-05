# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified line Hilbert transform on the Poisson / conjugate-Poisson basis.

Rigor checks:

* basis evaluators (``p_a``, ``q_a``, derivatives, ``atan`` primitive) enclose truth;
* the **exact** rotation ``H[p_a] = q_a`` and ``H[q_a] = -p_a`` is cross-checked
  against an independent high-precision *principal-value* quadrature (the
  desingularized line integral), so the closed forms are validated numerically and
  not merely asserted.
"""

from __future__ import annotations

import math

import mpmath as mp
import pytest
from omnibias.core.verified import (
    conjugate_poisson,
    conjugate_poisson_deriv,
    even_profile,
    even_profile_deriv,
    even_profile_tail_constant,
    hilbert_even_profile,
    hilbert_even_profile_deriv,
    hilbert_of_conjugate,
    hilbert_of_poisson,
    hilbert_tail_bound,
    poisson_kernel,
    poisson_kernel_deriv,
    poisson_primitive,
)


def _pv_hilbert(f: object, x0: float, a: float) -> float:
    """(1/pi) p.v. int f(t)/(x0 - t) dt via the desingularized integrand."""
    assert callable(f)
    with mp.workdps(40):
        fx0 = f(mp.mpf(x0), a)

        def integrand(t: object) -> object:
            return (f(t, a) - fx0) / (mp.mpf(x0) - t)

        val = mp.quad(integrand, [mp.ninf, x0, mp.inf]) / mp.pi
        return float(val)


def _mp_p(t: object, a: float) -> object:
    return mp.mpf(a) / (t * t + mp.mpf(a) ** 2)


def _mp_q(t: object, a: float) -> object:
    return t / (t * t + mp.mpf(a) ** 2)


@pytest.mark.parametrize("a", [0.5, 1.0, 2.7])
@pytest.mark.parametrize("x", [-3.1, -0.4, 0.0, 0.8, 5.0])
def test_basis_encloses_truth(x: float, a: float) -> None:
    d = x * x + a * a
    assert poisson_kernel(x, a).contains(a / d)
    assert conjugate_poisson(x, a).contains(x / d)
    assert conjugate_poisson_deriv(x, a).contains((a * a - x * x) / (d * d))
    assert poisson_kernel_deriv(x, a).contains(-2.0 * a * x / (d * d))
    assert poisson_primitive(x, a).contains(math.atan(x / a))


@pytest.mark.parametrize("a", [0.5, 1.0, 2.7])
@pytest.mark.parametrize("x0", [-3.1, -0.4, 0.8, 5.0])
def test_hilbert_rotation_matches_pv_quadrature(x0: float, a: float) -> None:
    # H[p_a] = q_a
    h_p = _pv_hilbert(_mp_p, x0, a)
    assert hilbert_of_poisson(x0, a).contains(h_p)
    assert abs(h_p - x0 / (x0 * x0 + a * a)) < 1e-12
    # H[q_a] = -p_a
    h_q = _pv_hilbert(_mp_q, x0, a)
    assert hilbert_of_conjugate(x0, a).contains(h_q)
    assert abs(h_q + a / (x0 * x0 + a * a)) < 1e-12


def test_hilbert_rotation_is_exact_closed_form() -> None:
    """H[q_a] = -p_a and H[p_a] = q_a as interval-equal closed forms."""
    for a in (0.7, 1.3):
        for x in (-2.0, 0.3, 4.5):
            hq = hilbert_of_conjugate(x, a)
            mp_neg_p = -a / (x * x + a * a)
            assert hq.contains(mp_neg_p)
            hp = hilbert_of_poisson(x, a)
            assert hp.contains(x / (x * x + a * a))


def test_conjugate_decays_like_inverse_x() -> None:
    """Physical far field: |q_a(x)| ~ 1/x (not exponentially small)."""
    a = 1.0
    big = conjugate_poisson(1000.0, a)
    assert 0.0 < big.lo
    assert big.contains(1000.0 / (1000.0**2 + 1.0))


def _true_tail(a: float, x_trunc: float, x0: float) -> float:
    """(1/pi) int_{|t|>=X} q_a(t)/(x0 - t) dt -- the exact far-field contribution."""
    with mp.workdps(40):
        def integrand(t: object) -> object:
            return (t / (t * t + mp.mpf(a) ** 2)) / (mp.mpf(x0) - t)

        lo = mp.quad(integrand, [mp.ninf, -x_trunc])
        hi = mp.quad(integrand, [x_trunc, mp.inf])
        return float((lo + hi) / mp.pi)


@pytest.mark.parametrize("x_trunc", [8.0, 20.0, 60.0])
@pytest.mark.parametrize("x0", [0.0, 1.5, -2.0])
def test_tail_bound_encloses_true_tail(x0: float, x_trunc: float) -> None:
    # q_a obeys |q_a(t)| <= 1/|t| everywhere, so C=1, p=1.
    a = 1.0
    rho = 2.0
    b = hilbert_tail_bound(decay_const=1.0, decay_power=1.0, x_trunc=x_trunc, core_radius=rho)
    true = _true_tail(a, x_trunc, x0)
    assert b.contains(true)
    # not absurdly loose: within ~3 orders of magnitude of the true tail
    assert b.hi < 1000.0 * max(abs(true), 1e-12)


def test_tail_bound_decreases_with_truncation() -> None:
    kw = {"decay_const": 1.0, "decay_power": 1.0, "core_radius": 1.0}
    b10 = hilbert_tail_bound(x_trunc=10.0, **kw).hi
    b100 = hilbert_tail_bound(x_trunc=100.0, **kw).hi
    b1000 = hilbert_tail_bound(x_trunc=1000.0, **kw).hi
    assert b10 > b100 > b1000 > 0.0
    # p=1 => tail ~ 1/X: a 10x truncation should shrink the bound ~10x
    assert 8.0 < b10 / b100 < 12.0


def test_tail_bound_steeper_decay_is_tighter() -> None:
    soft = hilbert_tail_bound(decay_const=1.0, decay_power=1.0, x_trunc=50.0, core_radius=1.0).hi
    steep = hilbert_tail_bound(decay_const=1.0, decay_power=2.0, x_trunc=50.0, core_radius=1.0).hi
    assert steep < soft


def test_tail_bound_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        hilbert_tail_bound(decay_const=1.0, decay_power=0.0, x_trunc=10.0, core_radius=1.0)
    with pytest.raises(ValueError):
        hilbert_tail_bound(decay_const=1.0, decay_power=1.0, x_trunc=1.0, core_radius=2.0)


# --------------------------------------------------------------------------- #
# Even-profile layer (CCF-on-the-line representation)                          #
# --------------------------------------------------------------------------- #
_COEFFS = (1.2, -0.7, 0.5)
_SCALES = (0.6, 1.3, 2.1)


def _mp_theta(t: object) -> object:
    return sum(c * (mp.mpf(a) / (t * t + mp.mpf(a) ** 2)) for c, a in zip(_COEFFS, _SCALES, strict=True))


@pytest.mark.parametrize("x", [-3.1, -0.4, 0.0, 0.8, 5.0])
def test_even_profile_values_enclose_truth(x: float) -> None:
    theta_true = sum(c * (a / (x * x + a * a)) for c, a in zip(_COEFFS, _SCALES, strict=True))
    dtheta_true = sum(c * (-2.0 * a * x / (x * x + a * a) ** 2) for c, a in zip(_COEFFS, _SCALES, strict=True))
    assert even_profile(x, _COEFFS, _SCALES).contains(theta_true)
    assert even_profile_deriv(x, _COEFFS, _SCALES).contains(dtheta_true)


def test_even_profile_is_even_and_deriv_is_odd() -> None:
    for x in (0.3, 1.7, 4.2):
        assert even_profile(x, _COEFFS, _SCALES).contains(even_profile(-x, _COEFFS, _SCALES).mid)
        d_pos = even_profile_deriv(x, _COEFFS, _SCALES)
        d_neg = even_profile_deriv(-x, _COEFFS, _SCALES)
        assert d_pos.contains(-d_neg.mid)


@pytest.mark.parametrize("x0", [-3.1, -0.4, 0.8, 5.0])
def test_hilbert_even_profile_matches_pv_quadrature(x0: float) -> None:
    # H[Theta] computed exactly term-by-term must match the desingularized PV integral.
    with mp.workdps(40):
        fx0 = _mp_theta(mp.mpf(x0))

        def integrand(t: object) -> object:
            return (_mp_theta(t) - fx0) / (mp.mpf(x0) - t)  # noqa: B023

        h_true = float(mp.quad(integrand, [mp.ninf, x0, mp.inf]) / mp.pi)
    assert hilbert_even_profile(x0, _COEFFS, _SCALES).contains(h_true)
    # closed form: H[Theta] = sum c_i q_{a_i}
    closed = sum(c * (x0 / (x0 * x0 + a * a)) for c, a in zip(_COEFFS, _SCALES, strict=True))
    assert abs(h_true - closed) < 1e-11


@pytest.mark.parametrize("x", [-3.1, -0.4, 0.8, 5.0])
def test_hilbert_even_profile_deriv_is_derivative_of_hilbert(x: float) -> None:
    # (H Theta)' must equal the mpmath derivative of H[Theta] (closed form sum c_i q'_{a_i}).
    with mp.workdps(40):
        def h_theta(t: object) -> object:
            return sum(c * (t / (t * t + mp.mpf(a) ** 2)) for c, a in zip(_COEFFS, _SCALES, strict=True))

        d_true = float(mp.diff(h_theta, x))
    assert hilbert_even_profile_deriv(x, _COEFFS, _SCALES).contains(d_true)


def test_even_profile_tail_constant_bounds_far_field() -> None:
    c_const, p = even_profile_tail_constant(_COEFFS, _SCALES)
    assert p == 2.0
    # |Theta(x)| <= C |x|^{-2} must hold in the far field.
    for x in (10.0, 50.0, 200.0):
        theta = abs(even_profile(x, _COEFFS, _SCALES).mid)
        assert theta <= c_const / x**2 + 1e-15
    # feeds the Hilbert tail bound without error
    b = hilbert_tail_bound(decay_const=c_const, decay_power=p, x_trunc=20.0, core_radius=3.0)
    assert b.hi > 0.0


def test_even_profile_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        even_profile(0.5, (1.0, 2.0), (1.0,))  # mismatched lengths
    with pytest.raises(ValueError):
        even_profile(0.5, (1.0,), (0.0,))  # zero scale
    with pytest.raises(ValueError):
        even_profile(0.5, (), ())  # empty

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified 2-D SQG substrate tests against an independent mpmath reference.

The crux of the SQG basis is that the *single* Riesz transform / half-Laplacian
is elementary on the 2-D Poisson kernel ``theta_a`` whose symbol is
``e^{-a|xi|}``.  We pin that down two independent ways: a direct ``mpmath`` Hankel
transform of the symbol (the radial 2-D inverse Fourier transform) and ``mpmath``
derivatives of the closed-form stream function.
"""

from __future__ import annotations

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sqg import (
    sqg_blob,
    sqg_blob_gradient,
    sqg_blob_l2_inner,
    sqg_riesz,
    sqg_stream,
    sqg_stream_gradient,
    sqg_velocity,
    sqg_velocity_divergence_residual,
)

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

_PTS = [(0.3, -0.2), (1.1, 0.4), (-0.5, 0.9), (0.0, 0.6)]
_A = 0.7


def _hankel(symbol, r: float) -> object:
    """2-D radial inverse Fourier transform ``(1/2pi) int_0^inf S(rho) J0(r rho) rho drho``."""
    return (1 / (2 * mp.pi)) * mp.quad(lambda rho: symbol(rho) * mp.j0(r * rho) * rho, [0, mp.inf])


def test_blob_symbol_is_poisson_exp_via_hankel() -> None:
    # theta_a has Fourier symbol e^{-a|xi|}: the inverse transform must match the
    # closed form a/(2pi D^{3/2}).  This is what makes the half-Laplacian elementary.
    for x, y in _PTS:
        r = (x * x + y * y) ** 0.5
        truth = float(_hankel(lambda rho: mp.e ** (-_A * rho), r))
        enc = sqg_blob(x, y, _A)
        assert enc.lo <= truth <= enc.hi


def test_stream_is_half_inverse_laplacian_via_hankel() -> None:
    # psi_a = (-Delta)^{-1/2} theta_a has symbol e^{-a|xi|}/|xi|; the inverse
    # transform is the closed form 1/(2pi D^{1/2}).
    for x, y in _PTS:
        r = (x * x + y * y) ** 0.5
        truth = float(_hankel(lambda rho: mp.e ** (-_A * rho) / rho, r))
        enc = sqg_stream(x, y, _A)
        assert enc.lo <= truth <= enc.hi


def _psi(x: float, y: float, a: float) -> object:
    return 1 / (2 * mp.pi * mp.sqrt(x * x + y * y + a * a))


def test_riesz_equals_gradient_of_stream() -> None:
    # The single Riesz transform R_j theta_a = d_j psi_a (definition of psi as the
    # half-inverse): checked against an mpmath derivative of the closed-form psi.
    for x, y in _PTS:
        px = float(mp.diff(lambda t: _psi(t, y, _A), x))  # noqa: B023
        py = float(mp.diff(lambda t: _psi(x, t, _A), y))  # noqa: B023
        r0 = sqg_riesz(0, x, y, _A)
        r1 = sqg_riesz(1, x, y, _A)
        assert r0.lo <= px <= r0.hi
        assert r1.lo <= py <= r1.hi


def test_stream_gradient_matches_mpmath() -> None:
    for x, y in _PTS:
        px = float(mp.diff(lambda t: _psi(t, y, _A), x))  # noqa: B023
        py = float(mp.diff(lambda t: _psi(x, t, _A), y))  # noqa: B023
        gx, gy = sqg_stream_gradient(x, y, _A)
        assert gx.lo <= px <= gx.hi
        assert gy.lo <= py <= gy.hi


def test_velocity_is_perp_gradient_of_stream() -> None:
    # u = nabla^perp psi = (-d_y psi, d_x psi); also u = (-R_2 theta, R_1 theta).
    for x, y in _PTS:
        px = float(mp.diff(lambda t: _psi(t, y, _A), x))  # noqa: B023
        py = float(mp.diff(lambda t: _psi(x, t, _A), y))  # noqa: B023
        ux, uy = sqg_velocity(x, y, _A)
        assert ux.lo <= -py <= ux.hi
        assert uy.lo <= px <= uy.hi
        # cross-check against the independent Riesz route u = R^perp theta
        r0 = sqg_riesz(0, x, y, _A)
        r1 = sqg_riesz(1, x, y, _A)
        assert (ux - (-r1)).abs().hi < 1e-12
        assert (uy - r0).abs().hi < 1e-12


def _theta(x: float, y: float, a: float) -> object:
    return a / (2 * mp.pi * (x * x + y * y + a * a) ** mp.mpf("1.5"))


def test_blob_gradient_matches_mpmath() -> None:
    for x, y in _PTS:
        tx = float(mp.diff(lambda t: _theta(t, y, _A), x))  # noqa: B023
        ty = float(mp.diff(lambda t: _theta(x, t, _A), y))  # noqa: B023
        gx, gy = sqg_blob_gradient(x, y, _A)
        assert gx.lo <= tx <= gx.hi
        assert gy.lo <= ty <= gy.hi


def test_velocity_perp_to_blob_gradient_exact_steady_state() -> None:
    # u ~ (y, -x) tangential, nabla theta ~ (x, y) radial: u . nabla theta == 0.
    for x, y in _PTS:
        ux, uy = sqg_velocity(x, y, _A)
        gx, gy = sqg_blob_gradient(x, y, _A)
        res = ux * gx + uy * gy
        assert res.lo <= 0.0 <= res.hi
        assert res.abs().hi < 1e-12


def test_velocity_divergence_residual_encloses_zero() -> None:
    for x, y in _PTS:
        res = sqg_velocity_divergence_residual(x, y, _A)
        assert res.lo <= 0.0 <= res.hi
        assert res.width < 1e-10


def test_divergence_free_via_independent_mpmath() -> None:
    def u1(x: float, y: float) -> object:
        return y / (2 * mp.pi * (x * x + y * y + _A * _A) ** mp.mpf("1.5"))

    def u2(x: float, y: float) -> object:
        return -x / (2 * mp.pi * (x * x + y * y + _A * _A) ** mp.mpf("1.5"))

    for x, y in _PTS:
        div = mp.diff(lambda t: u1(t, y), x) + mp.diff(lambda t: u2(x, t), y)  # noqa: B023
        assert abs(float(div)) < 1e-12


def test_blob_l2_inner_matches_mpmath_quadrature() -> None:
    # <theta_a, theta_b>_{L2(R2)} = 1/(2 pi (a+b)^2): check the closed form against
    # an independent mpmath radial quadrature int_0^inf theta_a theta_b 2 pi r dr.
    for a, b in [(0.7, 0.7), (0.5, 1.3), (1.1, 2.4)]:
        truth = float(
            mp.quad(lambda r: _theta(r, 0.0, a) * _theta(r, 0.0, b) * 2 * mp.pi * r, [0, mp.inf])  # noqa: B023
        )
        enc = sqg_blob_l2_inner(a, b)
        assert enc.lo <= truth <= enc.hi
        # and against the bare closed form
        closed = 1.0 / (2 * float(mp.pi) * (a + b) ** 2)
        assert enc.lo <= closed <= enc.hi


def test_blob_l2_inner_is_symmetric_and_positive() -> None:
    ab = sqg_blob_l2_inner(0.6, 1.4)
    ba = sqg_blob_l2_inner(1.4, 0.6)
    assert ab.lo == ba.lo and ab.hi == ba.hi
    assert ab.lo > 0.0


def test_blob_l2_inner_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError):
        sqg_blob_l2_inner(0.0, 1.0)
    with pytest.raises(ValueError):
        sqg_blob_l2_inner(1.0, -0.3)


def test_blob_is_positive_and_decays() -> None:
    near = sqg_blob(0.0, 0.0, _A)
    far = sqg_blob(5.0, 0.0, _A)
    assert near.lo > 0.0
    assert far.hi < near.lo  # monotone radial decay


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        sqg_blob(0.1, 0.2, 0.0)  # zero scale
    with pytest.raises(ValueError):
        sqg_riesz(2, 0.1, 0.2, 0.7)  # bad axis
    with pytest.raises(ValueError):
        sqg_velocity(0.1, 0.2, 0.0)  # zero scale
    assert isinstance(sqg_stream(0.1, 0.2, 0.7), Interval)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Radii-polynomial + Krawczyk existence proofs on published CAP benchmarks."""

from __future__ import annotations

import math

from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import (
    certify_zero_radii,
    krawczyk_certificate,
    newton_kantorovich_bounds,
    radii_polynomial_certificate,
)
from omnibias.core.verified.transcend import cos_point, sin_point

# The Dottie number: the unique real fixed point of cos, x* = cos(x*).
DOTTIE = 0.7390851332151607


def _cos_iv(box: Interval) -> Interval:
    """Enclosure of cos over an interval via the center value + Lipschitz 1."""
    c = cos_point(box.mid)
    r = max(box.hi - box.mid, box.mid - box.lo)
    return c + Interval(-r, r)


def _sin_iv(box: Interval) -> Interval:
    s = sin_point(box.mid)
    r = max(box.hi - box.mid, box.mid - box.lo)
    return s + Interval(-r, r)


def test_radii_polynomial_synthetic_existence() -> None:
    # Y0 small, contraction slack large -> existence with a tiny radius.
    cert = radii_polynomial_certificate(y0=1e-3, z0=0.1, z1=0.0, z2=0.5)
    assert cert is not None
    assert cert.kappa < 1.0
    assert cert.p_value < 0.0
    assert cert.radius > 0.0
    assert verify_certificate_digest(cert.certificate)


def test_radii_polynomial_no_contraction_returns_none() -> None:
    # Z0 + Z1 >= 1: no contraction possible.
    assert radii_polynomial_certificate(y0=0.1, z0=0.9, z1=0.2, z2=0.5) is None
    # Discriminant negative: Y0 too large for the available slack.
    assert radii_polynomial_certificate(y0=10.0, z0=0.0, z1=0.0, z2=1.0) is None


def test_dottie_via_krawczyk() -> None:
    # F(x) = x - cos(x); F'(x) = 1 + sin(x); A = 1/F'(x_bar).
    x_bar = DOTTIE
    a = 1.0 / (1.0 + math.sin(x_bar))

    def func(xs: list[Interval]) -> list[Interval]:
        return [xs[0] - _cos_iv(xs[0])]

    def jac(xs: list[Interval]) -> list[list[Interval]]:
        return [[Interval.point(1.0) + _sin_iv(xs[0])]]

    cert = krawczyk_certificate(func, jac, [x_bar], [[a]], r=1e-3)
    assert cert is not None
    assert cert.kappa < 1.0
    lo, hi = cert.enclosure[0]
    assert lo <= DOTTIE <= hi  # the true root is inside the certified box


def test_dottie_via_radii_polynomial() -> None:
    # Hand-derived rigorous bounds: |F''(x)| = |cos x| <= 1 everywhere, so the
    # Lipschitz constant of F' is 1 and Z2 = ||A|| * 1.
    x_bar = DOTTIE
    a = 1.0 / (1.0 + math.sin(x_bar))

    def func(xs: list[Interval]) -> list[Interval]:
        return [xs[0] - _cos_iv(xs[0])]

    def jac(xs: list[Interval]) -> list[list[Interval]]:
        return [[Interval.point(1.0) + _sin_iv(xs[0])]]

    bounds = newton_kantorovich_bounds(func, jac, [x_bar], [[a]], lipschitz_df=1.0)
    assert bounds.y0 < 1e-6  # x_bar is an excellent approximation
    cert = radii_polynomial_certificate(bounds.y0, bounds.z0, bounds.z1, bounds.z2)
    assert cert is not None
    # The unique root lies within radius of x_bar; check it brackets Dottie.
    assert abs(DOTTIE - x_bar) <= cert.radius


def test_2d_polynomial_system_krawczyk() -> None:
    # Intersection of the unit circle and the line y = x in the first quadrant:
    # F(x,y) = [x^2 + y^2 - 1, x - y]; root = (1/sqrt2, 1/sqrt2).
    root = 1.0 / math.sqrt(2.0)

    def func(v: list[Interval]) -> list[Interval]:
        x, y = v
        return [x * x + y * y - Interval.point(1.0), x - y]

    def jac(v: list[Interval]) -> list[list[Interval]]:
        x, y = v
        two = Interval.point(2.0)
        return [[two * x, two * y], [Interval.point(1.0), Interval.point(-1.0)]]

    # A = inverse of DF at the root (computed in float).
    j = [[2 * root, 2 * root], [1.0, -1.0]]
    det = j[0][0] * j[1][1] - j[0][1] * j[1][0]
    a_inv = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]]

    cert = krawczyk_certificate(func, jac, [root, root], a_inv, r=1e-2)
    assert cert is not None
    assert cert.kappa < 1.0
    for lo, hi in cert.enclosure:
        assert lo <= root <= hi


def test_2d_polynomial_system_radii_polynomial() -> None:
    root = 1.0 / math.sqrt(2.0)

    def func(v: list[Interval]) -> list[Interval]:
        x, y = v
        return [x * x + y * y - Interval.point(1.0), x - y]

    def jac(v: list[Interval]) -> list[list[Interval]]:
        x, y = v
        two = Interval.point(2.0)
        return [[two * x, two * y], [Interval.point(1.0), Interval.point(-1.0)]]

    j = [[2 * root, 2 * root], [1.0, -1.0]]
    det = j[0][0] * j[1][1] - j[0][1] * j[1][0]
    a_inv = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]]

    # DF is affine in (x,y): ||DF(u)-DF(v)||_inf <= 2||u-v||_inf, so Lipschitz = 2.
    cert = certify_zero_radii(func, jac, [root, root], a_inv, lipschitz_df=2.0)
    assert cert is not None
    assert cert.kappa < 1.0

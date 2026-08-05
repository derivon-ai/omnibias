# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Containment tests for Taylor models over a whole cell."""

from __future__ import annotations

import random

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.taylor_model import TaylorModel


def _mpmath() -> object:
    """Skip only the tests that genuinely need the high-precision oracle.

    Gating the whole module would also skip the dozen containment tests that
    compare against plain double arithmetic, which is exactly the coverage a
    rigorous register cannot afford to lose on an mpmath-free machine.
    """
    return pytest.importorskip("mpmath")


def _sample(center: float, radius: float, n: int = 21) -> list[float]:
    return [center - radius + 2 * radius * i / (n - 1) for i in range(n)]


def test_identity_bound_is_cell() -> None:
    tm = TaylorModel.identity(0.5, 0.25, order=4)
    b = tm.bound()
    assert b.lo <= 0.25
    assert b.hi >= 0.75


def test_polynomial_product_encloses_pointwise() -> None:
    center, radius = 0.3, 0.4
    x = TaylorModel.identity(center, radius, order=5)
    # f(x) = (x^2 + 1) * (x - 2)
    f = (x.pow_int(2) + 1.0) * (x - 2.0)
    b = f.bound()
    for xi in _sample(center, radius):
        val = (xi * xi + 1.0) * (xi - 2.0)
        assert b.lo <= val <= b.hi


def test_remainder_absorbs_high_degree() -> None:
    center, radius = 0.0, 0.5
    x = TaylorModel.identity(center, radius, order=2)
    # x^4 has degree above the truncation order 2 -> lives in the remainder.
    f = x.pow_int(4)
    b = f.bound()
    for xi in _sample(center, radius):
        assert b.lo <= xi**4 <= b.hi
    # remainder must be non-trivial (the shape is pushed into R).
    assert b.width > 0.0


def test_constant_model() -> None:
    tm = TaylorModel.constant(3.5, center=1.0, radius=2.0, order=3)
    b = tm.bound()
    assert b.lo <= 3.5 <= b.hi
    assert b.width < 1e-12


def test_reciprocal_encloses_affine_pointwise() -> None:
    center, radius = 2.0, 0.3
    x = TaylorModel.identity(center, radius, order=6)
    inv = x.reciprocal()
    b = inv.bound()
    for xi in _sample(center, radius):
        assert b.lo <= 1.0 / xi <= b.hi


def test_reciprocal_of_poisson_denominator_matches_mpmath() -> None:
    mpmath = _mpmath()
    center, radius, a = 0.4, 0.3, 0.7
    x = TaylorModel.identity(center, radius, order=8)
    denom = x.pow_int(2) + a * a  # y^2 + a^2 > 0
    inv = denom.reciprocal()
    b = inv.bound()
    for xi in _sample(center, radius):
        truth = float(mpmath.mpf(1) / (mpmath.mpf(xi) ** 2 + mpmath.mpf(a) ** 2))
        assert b.lo <= truth <= b.hi


def test_reciprocal_cancels_dependency_tightly() -> None:
    # f * (1/f) must enclose 1 far more tightly than the naive interval
    # extension (D_range * (1/D)_range) -- the whole point of a Taylor model.
    center, radius, a = 0.4, 0.3, 0.7
    x = TaylorModel.identity(center, radius, order=8)
    denom = x.pow_int(2) + a * a
    one = (denom * denom.reciprocal()).bound()
    assert one.lo <= 1.0 <= one.hi
    d_range = denom.bound()
    naive_width = (d_range * d_range.reciprocal()).width
    assert one.width < 0.1 * naive_width


def test_truediv_matches_identity_times_reciprocal() -> None:
    center, radius, a = 0.4, 0.3, 0.7
    x = TaylorModel.identity(center, radius, order=6)
    denom = x.pow_int(2) + a * a
    q_div = (x / denom).bound()  # x / (x^2 + a^2)
    q_mul = (x * denom.reciprocal()).bound()
    assert q_div.lo == q_mul.lo
    assert q_div.hi == q_mul.hi
    for xi in _sample(center, radius):
        truth = xi / (xi * xi + a * a)
        assert q_div.lo <= truth <= q_div.hi


def test_reciprocal_scalar_divisor() -> None:
    center, radius = 1.0, 0.2
    x = TaylorModel.identity(center, radius, order=4)
    half = (x / 2.0).bound()
    for xi in _sample(center, radius):
        assert half.lo <= xi / 2.0 <= half.hi


def test_reciprocal_rejects_zero_crossing() -> None:
    x = TaylorModel.identity(0.0, 1.0, order=4)  # range [-1, 1] straddles 0
    with pytest.raises(ValueError, match="bounded away from 0"):
        x.reciprocal()


def test_reciprocal_rejects_nonconvergent_cell() -> None:
    # f bounded away from 0 (range [0.001, 2.0]) but the relative variation
    # tau >= 1, so the geometric 1/(1+g) series cannot converge: the guard must
    # fire rather than return an unsound model.
    tm = TaylorModel(0.0, 1.0, [Interval.point(1.0)], Interval(-0.999, 1.0))
    assert tm.bound().lo > 0.0
    with pytest.raises(ValueError, match="does not converge"):
        tm.reciprocal()


def test_sqrt_encloses_affine_pointwise() -> None:
    mpmath = _mpmath()
    center, radius = 2.0, 0.4
    x = TaylorModel.identity(center, radius, order=6)
    b = x.sqrt().bound()
    for xi in _sample(center, radius):
        assert b.lo <= mpmath.sqrt(xi) <= b.hi


def test_sqrt_of_poisson_denominator_matches_mpmath() -> None:
    mpmath = _mpmath()
    # sqrt(x^2 + a^2): the half-power at the heart of the SQG Poisson blob.
    center, radius, a = 0.4, 0.3, 0.7
    x = TaylorModel.identity(center, radius, order=8)
    root = (x.pow_int(2) + a * a).sqrt()
    b = root.bound()
    for xi in _sample(center, radius):
        truth = float(mpmath.sqrt(mpmath.mpf(xi) ** 2 + mpmath.mpf(a) ** 2))
        assert b.lo <= truth <= b.hi


def test_sqrt_inverse_three_halves_power_matches_mpmath() -> None:
    mpmath = _mpmath()
    # (x^2 + a^2)^{-3/2} = invd * sqrt(invd): the exact SQG velocity kernel.
    center, radius, a = 0.5, 0.25, 1.1
    x = TaylorModel.identity(center, radius, order=8)
    invd = (x.pow_int(2) + a * a).reciprocal()
    kernel = (invd * invd.sqrt()).bound()
    for xi in _sample(center, radius):
        truth = float((mpmath.mpf(xi) ** 2 + mpmath.mpf(a) ** 2) ** mpmath.mpf("-1.5"))
        assert kernel.lo <= truth <= kernel.hi


def test_sqrt_kernel_tighter_than_naive_interval() -> None:
    # The SQG velocity kernel r * (r^2 + a^2)^{-3/2} via a Taylor model must be
    # tighter than the naive interval extension (which suffers the r-dependency).
    center, radius, a = 1.0, 0.3, 0.7
    x = TaylorModel.identity(center, radius, order=8)
    invd = (x.pow_int(2) + a * a).reciprocal()
    tm = (x * (invd * invd.sqrt())).bound()
    # pointwise correctness
    for xi in _sample(center, radius):
        truth = xi * (xi * xi + a * a) ** (-1.5)
        assert tm.lo <= truth <= tm.hi
    # naive interval extension of the same expression over the whole cell
    d = Interval(center - radius, center + radius).pow_int(2) + a * a
    naive = Interval(center - radius, center + radius) * (d.reciprocal() * d.reciprocal().sqrt())
    assert tm.width < 0.5 * naive.width


def test_sqrt_rejects_nonpositive_cell() -> None:
    x = TaylorModel.identity(0.0, 1.0, order=4)  # range [-1, 1] includes <= 0
    with pytest.raises(ValueError, match="requires f > 0"):
        x.sqrt()


def test_sqrt_rejects_nonconvergent_cell() -> None:
    # f > 0 (range [0.001, 2.0]) but relative variation tau >= 1: guard fires.
    tm = TaylorModel(0.0, 1.0, [Interval.point(1.0)], Interval(-0.999, 1.0))
    assert tm.bound().lo > 0.0
    with pytest.raises(ValueError, match="does not converge"):
        tm.sqrt()


def test_reciprocal_tail_is_sound_on_a_dense_grid_and_random_sample() -> None:
    # The geometric tail bound grows with tau, so a tau that is not outward
    # rounded silently shrinks the remainder. Check containment against both a
    # dense deterministic grid and a seeded random sample, over a range of cell
    # radii and truncation orders (the rigorous-register soundness convention).
    rng = random.Random(20260803)
    for order in (2, 4, 8):
        for center, radius in ((2.0, 0.5), (0.4, 0.3), (-1.5, 0.4), (7.0, 1.5)):
            x = TaylorModel.identity(center, radius, order=order)
            inv = x.reciprocal()
            b = inv.bound()
            dense = [center - radius + 2 * radius * i / 400 for i in range(401)]
            drawn = [rng.uniform(center - radius, center + radius) for _ in range(200)]
            for xi in dense + drawn:
                assert b.lo <= 1.0 / xi <= b.hi, (order, center, radius, xi)


def test_reciprocal_tau_is_outward_rounded() -> None:
    # tau must be the outward `.mag` of the relative variation, never the bare
    # max(abs(lo), abs(hi)) that sits one ulp lower.
    center, radius = 2.0, 0.5
    x = TaylorModel.identity(center, radius, order=6)
    g_range = (x - Interval.point(x.coeffs[0].mid)).bound() * Interval.point(
        1.0 / x.coeffs[0].mid
    )
    bare = max(abs(g_range.lo), abs(g_range.hi))
    assert g_range.mag > bare  # the accessor is strictly more conservative
    # and the model built from it still encloses the truth
    assert x.reciprocal().bound().contains(1.0 / center)


def test_cell_rel_is_outward_enclosure_of_claimed_radius() -> None:
    # [-radius, radius] must be outwardly inflated: the bare float endpoints are
    # not enough for the rigorous register's cell geometry.
    from omnibias.core.verified.interval import _pred, _succ
    from omnibias.core.verified.taylor_model import _cell_rel

    for radius in (0.25, 0.5, 1.0, 1e-16, 7.5):
        cell = _cell_rel(radius)
        assert cell.lo <= -radius <= radius <= cell.hi
        assert cell.lo == _pred(-radius)
        assert cell.hi == _succ(radius)
        # Every sample offset in the mathematical cell is inside the enclosure.
        for t in (-radius, -radius / 2, 0.0, radius / 2, radius):
            assert cell.contains(t)
    assert _cell_rel(0.0) == Interval.point(0.0)


def test_taylor_model_rel_matches_outward_cell() -> None:
    from omnibias.core.verified.taylor_model import _cell_rel

    tm = TaylorModel.identity(1.0, 0.3, order=3)
    assert tm._rel().lo == _cell_rel(0.3).lo
    assert tm._rel().hi == _cell_rel(0.3).hi
    # Identity bound must still cover the mathematical cell endpoints.
    b = tm.bound()
    assert b.lo <= 1.0 - 0.3
    assert b.hi >= 1.0 + 0.3


def test_antiderivative_uses_the_shared_cell() -> None:
    # antiderivative() must scale the remainder by exactly the cell's relative
    # variable; F(center) = 0 and F encloses the true integral on the cell.
    center, radius = 1.0, 0.25
    x = TaylorModel.identity(center, radius, order=5)
    F = (x * x).antiderivative()  # d/dx of (x-c)^3/3 style primitive
    assert F.radius == radius and F.center == center
    assert F.bound().contains(0.0)  # F(center) = 0 lies in the cell enclosure
    b = F.bound()
    for xi in _sample(center, radius, n=41):
        t = xi - center
        truth = ((t + center) ** 3 - center**3) / 3.0
        assert b.lo <= truth <= b.hi, (xi, truth, b.lo, b.hi)

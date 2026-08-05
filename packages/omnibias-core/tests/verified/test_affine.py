# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soundness and tightness of rigorous affine arithmetic (zonotopes)."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest
from omnibias.core.verified.affine import AffineForm, new_noise_symbol
from omnibias.core.verified.interval import Interval


def _realize(form: AffineForm, assignment: dict[int, float]) -> Interval:
    """The enclosure the form yields when the *named* symbols are fixed.

    The anonymous error term still ranges over ``[-1, 1]``, so the returned
    interval is what a sound result must contain at this assignment.
    """
    acc = Interval.point(form.center)
    for sid, coeff in form.deviations.items():
        acc = acc + Interval.point(coeff) * Interval.point(assignment.get(sid, 0.0))
    return acc + Interval(-form.error, form.error)


def _exact_value(form: AffineForm, assignment: dict[int, float]) -> Fraction:
    """Exact rational value of a *pure* (error-free) affine form at ``a``.

    Computed with :class:`~fractions.Fraction` so the ground truth carries no
    floating rounding of its own -- the enclosure is rigorous to the last ulp.
    """
    assert form.error == 0.0
    acc = Fraction(form.center)
    for sid, coeff in form.deviations.items():
        acc += Fraction(coeff) * Fraction(assignment[sid])
    return acc


def test_constant_is_tight() -> None:
    a = AffineForm.constant(2.5)
    iv = a.to_interval()
    assert iv.lo <= 2.5 <= iv.hi
    assert iv.width < 1e-15


def test_subtraction_cancels_shared_symbol_exactly() -> None:
    x = AffineForm.symbol(3.0, 2.0)
    z = x - x
    iv = z.to_interval()
    assert iv.lo <= 0.0 <= iv.hi
    assert iv.width < 1e-12  # dependency seen: x - x = 0, not [-4, 4]


def test_interval_loses_the_correlation() -> None:
    # x*(1 - x) on x in [0, 1]: true range [0, 1/4]; naive interval gives [0, 1].
    x = AffineForm.symbol(0.5, 0.5)
    affine = (x * (1.0 - x)).to_interval()
    naive = Interval(0.0, 1.0) * (1.0 - Interval(0.0, 1.0))
    assert affine.width < naive.width
    assert affine.lo <= 0.0 and affine.hi >= 0.25  # still encloses the truth


def test_addition_soundness_random() -> None:
    rng = random.Random(0)
    s1, s2 = new_noise_symbol(), new_noise_symbol()
    for _ in range(200):
        x = AffineForm(rng.uniform(-3, 3), {s1: rng.uniform(-2, 2), s2: rng.uniform(-2, 2)})
        y = AffineForm(rng.uniform(-3, 3), {s1: rng.uniform(-2, 2), s2: rng.uniform(-2, 2)})
        z = x + y
        for _ in range(8):
            a = {s1: rng.uniform(-1, 1), s2: rng.uniform(-1, 1)}
            truth = _exact_value(x, a) + _exact_value(y, a)
            enc = _realize(z, a)
            assert enc.lo <= truth <= enc.hi


def test_multiplication_soundness_random() -> None:
    rng = random.Random(1)
    s1, s2 = new_noise_symbol(), new_noise_symbol()
    for _ in range(200):
        x = AffineForm(rng.uniform(-3, 3), {s1: rng.uniform(-2, 2), s2: rng.uniform(-1, 1)})
        y = AffineForm(rng.uniform(-3, 3), {s1: rng.uniform(-1, 1), s2: rng.uniform(-2, 2)})
        z = x * y
        for _ in range(8):
            a = {s1: rng.uniform(-1, 1), s2: rng.uniform(-1, 1)}
            truth = _exact_value(x, a) * _exact_value(y, a)
            enc = _realize(z, a)
            assert enc.lo <= truth <= enc.hi


def test_reciprocal_soundness_random() -> None:
    rng = random.Random(2)
    sid = new_noise_symbol()
    for _ in range(200):
        x = AffineForm(rng.uniform(3.0, 6.0), {sid: rng.uniform(-1.0, 1.0)})
        z = x.reciprocal()
        for _ in range(8):
            a = {sid: rng.uniform(-1, 1)}
            truth = 1 / _exact_value(x, a)
            enc = _realize(z, a)
            assert enc.lo <= truth <= enc.hi


def test_sqrt_and_pow_soundness_random() -> None:
    rng = random.Random(3)
    sid = new_noise_symbol()
    for _ in range(200):
        x = AffineForm(rng.uniform(3.0, 6.0), {sid: rng.uniform(-1.0, 1.0)})
        zr = x.sqrt()
        zp = x**3
        for _ in range(8):
            a = {sid: rng.uniform(-1, 1)}
            xa = _exact_value(x, a)
            er, ep = _realize(zr, a), _realize(zp, a)
            assert er.lo <= math.sqrt(xa) <= er.hi
            assert ep.lo <= xa**3 <= ep.hi


def test_negative_radius_rejected() -> None:
    with pytest.raises(ValueError):
        AffineForm.symbol(0.0, -1.0)
    with pytest.raises(ValueError):
        AffineForm(0.0, {}, -1.0)


def test_reciprocal_through_zero_rejected() -> None:
    x = AffineForm.symbol(0.0, 1.0)  # encloses [-1, 1]
    with pytest.raises(ZeroDivisionError):
        x.reciprocal()


def test_property_based_multiplication_soundness() -> None:
    hypothesis = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")
    given = hypothesis.given
    settings = hypothesis.settings

    finite = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    eps = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)

    @settings(max_examples=300, deadline=None)
    @given(x0=finite, xc=finite, y0=finite, yc=finite, a=eps)
    def check(x0: float, xc: float, y0: float, yc: float, a: float) -> None:
        sid = 7
        x = AffineForm(x0, {sid: xc})
        y = AffineForm(y0, {sid: yc})
        z = x * y
        truth = (Fraction(x0) + Fraction(xc) * Fraction(a)) * (
            Fraction(y0) + Fraction(yc) * Fraction(a)
        )
        enc = _realize(z, {sid: a})
        assert enc.lo <= truth <= enc.hi

    check()

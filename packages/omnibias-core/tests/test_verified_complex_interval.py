# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for rigorous complex interval arithmetic (``ComplexInterval``)."""

from __future__ import annotations

import random

import pytest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval


def test_point_arithmetic_contains_true_value() -> None:
    rng = random.Random(0)
    for _ in range(3000):
        a = complex(rng.uniform(-4, 4), rng.uniform(-4, 4))
        b = complex(rng.uniform(-4, 4), rng.uniform(-4, 4))
        ca, cb = ComplexInterval.point(a), ComplexInterval.point(b)
        assert (ca + cb).contains(a + b)
        assert (ca - cb).contains(a - b)
        assert (ca * cb).contains(a * b)
        if abs(b) > 1e-6:
            assert (ca / cb).contains(a / b)


def test_wide_rectangle_product_contains_samples() -> None:
    a_box = ComplexInterval(Interval(0.5, 1.5), Interval(-0.5, 0.5))
    b_box = ComplexInterval(Interval(-2.0, -1.0), Interval(1.0, 2.0))
    prod = a_box * b_box
    rng = random.Random(1)
    for _ in range(2000):
        a = complex(rng.uniform(0.5, 1.5), rng.uniform(-0.5, 0.5))
        b = complex(rng.uniform(-2.0, -1.0), rng.uniform(1.0, 2.0))
        assert prod.contains(a * b)


def test_magnitude_is_upper_bound_and_modulus_encloses() -> None:
    box = ComplexInterval(Interval(-1.0, 2.0), Interval(0.5, 3.0))
    rng = random.Random(2)
    mod = box.modulus()
    for _ in range(2000):
        a = complex(rng.uniform(-1.0, 2.0), rng.uniform(0.5, 3.0))
        assert abs(a) <= box.mag + 1e-12
        assert mod.lo - 1e-12 <= abs(a) <= mod.hi + 1e-12


def test_modulus_lower_bound_zero_when_straddling_origin() -> None:
    box = ComplexInterval(Interval(-1.0, 1.0), Interval(-1.0, 1.0))
    assert box.modulus().lo == 0.0


def test_imag_unit_and_conj() -> None:
    i = ComplexInterval.imag_unit()
    sq = i * i
    assert sq.contains(-1 + 0j)
    z = ComplexInterval.point(3 - 4j)
    assert z.conj().contains(3 + 4j)
    # |z|^2 = z * conj(z) is real and ~25
    prod = z * z.conj()
    assert prod.contains(25 + 0j)


def test_from_value_promotions() -> None:
    assert ComplexInterval.from_value(2).contains(2 + 0j)
    assert ComplexInterval.from_value(1.5).contains(1.5 + 0j)
    assert ComplexInterval.from_value(2 + 3j).contains(2 + 3j)
    assert ComplexInterval.from_value(Interval.point(7.0)).contains(7 + 0j)
    ci = ComplexInterval.point(1j)
    assert ComplexInterval.from_value(ci) is ci


def test_division_by_zero_straddling_raises() -> None:
    z = ComplexInterval.point(1 + 1j)
    zero_denom = ComplexInterval(Interval(-1.0, 1.0), Interval(-1.0, 1.0))
    with pytest.raises(ZeroDivisionError):
        _ = z / zero_denom


def test_distributivity_enclosure() -> None:
    # a*(b+c) and a*b + a*c both enclose the true value (no soundness loss).
    rng = random.Random(5)
    for _ in range(1000):
        a = complex(rng.uniform(-2, 2), rng.uniform(-2, 2))
        b = complex(rng.uniform(-2, 2), rng.uniform(-2, 2))
        c = complex(rng.uniform(-2, 2), rng.uniform(-2, 2))
        ca, cb, cc = (ComplexInterval.point(x) for x in (a, b, c))
        lhs = ca * (cb + cc)
        rhs = ca * cb + ca * cc
        assert lhs.contains(a * (b + c))
        assert rhs.contains(a * (b + c))

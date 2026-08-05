# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""D-finite <-> P-recursive transforms and D-finite derivative/integral/compose closures."""

from __future__ import annotations

from fractions import Fraction
from math import factorial

import pytest
from omnibias.holonomic._core.dfinite import DFinite, PRecursive
from omnibias.holonomic._core.ore import diff_algebra, shift_algebra
from omnibias.holonomic._core.rational_poly import to_poly
from omnibias.holonomic._core.transforms import (
    dfinite_compose_poly,
    dfinite_derivative,
    dfinite_integral,
    dfinite_to_precursive,
    precursive_to_dfinite,
)


def _exp() -> DFinite:
    """e^x: (D - 1) f = 0, a_m = 1/m!."""
    return DFinite(diff_algebra().operator([[-1], [1]]), (Fraction(1),))


def _geom() -> DFinite:
    """1/(1-x): ((1-x) D - 1) f = 0, a_m = 1."""
    return DFinite(diff_algebra().operator([[-1], [1, -1]]), (Fraction(1),))


def test_dfinite_to_precursive_exp() -> None:
    d = _exp()
    rec = dfinite_to_precursive(d)
    assert rec.annihilator.order == 1
    assert rec.terms(12) == d.taylor(12)  # both 1/m!


def test_dfinite_to_precursive_geometric() -> None:
    d = _geom()
    rec = dfinite_to_precursive(d)
    assert rec.terms(12) == [Fraction(1)] * 12


def test_precursive_to_dfinite_reciprocal_factorial() -> None:
    # a_n = 1/n!: (n+1) a_{n+1} - a_n = 0; OGF = e^x, ODE D - 1.
    p = PRecursive(shift_algebra().operator([[-1], [1, 1]]), (Fraction(1),))
    d = precursive_to_dfinite(p)
    assert d.annihilator.order == 1
    assert d.taylor(12) == p.terms(12)


def test_transform_round_trip() -> None:
    d = _geom()
    rec = dfinite_to_precursive(d)
    back = precursive_to_dfinite(rec)
    assert back.taylor(12) == d.taylor(12)


def test_dfinite_derivative_geometric() -> None:
    # d/dx 1/(1-x) = 1/(1-x)^2 -> coefficients m+1.
    dd = dfinite_derivative(_geom())
    assert dd.taylor(8) == [Fraction(m + 1) for m in range(8)]


def test_dfinite_derivative_exp() -> None:
    dd = dfinite_derivative(_exp())
    assert dd.taylor(10) == [Fraction(1, factorial(m)) for m in range(10)]


def test_dfinite_integral_exp() -> None:
    # int_0^x e^t dt = e^x - 1 -> [0, 1, 1/2!, 1/3!, ...].
    di = dfinite_integral(_exp())
    expected = [Fraction(0)] + [Fraction(1, factorial(m)) for m in range(1, 10)]
    assert di.taylor(10) == expected


def test_dfinite_integral_geometric() -> None:
    # int_0^x 1/(1-t) dt = -log(1-x) = sum_{m>=1} x^m / m.
    di = dfinite_integral(_geom())
    expected = [Fraction(0)] + [Fraction(1, m) for m in range(1, 10)]
    assert di.taylor(10) == expected


def test_dfinite_compose_poly_exp_square() -> None:
    # e^{x^2} = sum x^{2m}/m!.
    dc = dfinite_compose_poly(_exp(), to_poly([0, 0, 1]))
    expected = [Fraction(0)] * 10
    for m in range(10):
        if 2 * m < 10:
            expected[2 * m] = Fraction(1, factorial(m))
    assert dc.taylor(10) == expected


def test_dfinite_compose_poly_requires_zero_constant() -> None:
    with pytest.raises(ValueError, match="p.0. = 0"):
        dfinite_compose_poly(_exp(), to_poly([1, 1]))  # p(0) = 1


def test_dfinite_compose_poly_geometric_scale() -> None:
    # 1/(1 - 2x) via composing 1/(1-x) with p(x) = 2x -> coefficients 2^m.
    dc = dfinite_compose_poly(_geom(), to_poly([0, 2]))
    assert dc.taylor(9) == [Fraction(2) ** m for m in range(9)]

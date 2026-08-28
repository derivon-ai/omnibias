# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified Laguerre-function basis: low-order values and tails."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.laguerre_basis import (
    LaguerreExpansion,
    laguerre_function,
    laguerre_poly_coeffs_exact,
    laguerre_weight,
)


def test_laguerre_low_order_coeffs() -> None:
    assert laguerre_poly_coeffs_exact(0) == (Fraction(1),)
    assert laguerre_poly_coeffs_exact(1) == (Fraction(1), Fraction(-1))
    # L_2 = (2 - 4x + x^2) / 2 = 1 - 2x + x^2/2
    assert laguerre_poly_coeffs_exact(2) == (Fraction(1), Fraction(-2), Fraction(1, 2))


def test_laguerre_function_contains_reference() -> None:
    x = 0.5
    l0 = laguerre_function(0, x)
    l1 = laguerre_function(1, x)
    weight = float(laguerre_weight(x).mid)
    assert l0.contains(weight)
    assert l1.contains((1.0 - x) * weight)


def test_laguerre_expansion_tail_contains_kept() -> None:
    expansion = LaguerreExpansion.from_coeffs([1.0, -0.1])
    kept = expansion.evaluate_kept(0.25)
    full = expansion.evaluate(0.25, coeff_bound=1.0, ratio=0.4, psi_bound=2.0)
    assert full.re.lo <= kept.re.lo and full.re.hi >= kept.re.hi


def test_laguerre_refuses_negative_x() -> None:
    import pytest

    with pytest.raises(ValueError, match="x >= 0"):
        laguerre_weight(Interval(-0.1, 0.1))

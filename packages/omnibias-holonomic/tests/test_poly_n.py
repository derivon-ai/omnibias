# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sparse n-variate rational polynomials and the 3×3 Jacobian."""

from __future__ import annotations

from fractions import Fraction

from omnibias.holonomic._core.poly_n import (
    PolyN,
    jacobian_det,
    q_from_p,
    sylvester_resultant,
)
from omnibias.holonomic._core.rational_poly import to_poly


def test_add_mul_partial_eval() -> None:
    x = PolyN.var(2, 0)
    y = PolyN.var(2, 1)
    f = (x + y) ** 2
    assert f.eval((2, 3)) == 25
    assert f.partial(0).eval((2, 3)) == 10
    composed = f.compose((PolyN.const(2, 1) + x, y))
    assert composed.eval((1, 1)) == 9


def test_jacobian_det_linear() -> None:
    x, y, z = PolyN.var(3, 0), PolyN.var(3, 1), PolyN.var(3, 2)
    det = jacobian_det((x + y, y + z, z + x))
    assert det.constant_value() == Fraction(2)


def test_q_from_p_alpoge_curve() -> None:
    q = q_from_p(to_poly([0, 4, -3]))
    assert q == to_poly([0, 0, 1, -1])


def test_sylvester_resultant_linear() -> None:
    # res(x-1, x-2) = 1-2 = -1
    assert sylvester_resultant(to_poly([-1, 1]), to_poly([-2, 1])) == Fraction(-1)
    assert sylvester_resultant(to_poly([-1, 1]), to_poly([-1, 1])) == Fraction(0)

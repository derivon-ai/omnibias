# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact-Q residual lift and integer null space."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.proof.lift import integer_null_space, residual_identically_zero


def test_residual_identically_zero_accepts_planted() -> None:
    design = [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]]
    coeffs = [Fraction(1), Fraction(-1, 2)]
    target = [Fraction(0), Fraction(1)]
    assert residual_identically_zero(design, coeffs, target) is True


def test_residual_identically_zero_rejects_perturbation() -> None:
    design = [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]]
    coeffs = [Fraction(1), Fraction(-1, 2)]
    target = [Fraction(0), Fraction(1) + Fraction(1, 1000)]
    assert residual_identically_zero(design, coeffs, target) is False


def test_integer_null_space_rank_deficient() -> None:
    null = integer_null_space([[1, 2], [2, 4]])
    assert len(null) == 1
    vector = null[0]
    assert vector[0] + 2 * vector[1] == 0

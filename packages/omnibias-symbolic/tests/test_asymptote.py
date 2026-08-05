# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Symbolic horizontal asymptote + saturating-law recognition."""

from __future__ import annotations

import math

import numpy as np
from omnibias.symbolic.discovery import SparseEquation
from omnibias.symbolic.expressions import RationalExpression, recognize_known_expression


def _rational(numerator: list[float], denominator_tail: list[float]) -> RationalExpression:
    return RationalExpression(
        numerator=np.asarray(numerator, dtype=float),
        denominator_tail=np.asarray(denominator_tail, dtype=float),
        x_shift=0.0,
        x_scale=1.0,
    )


def test_horizontal_asymptote_equal_degree_is_leading_ratio() -> None:
    # P = 1 + 2t (deg 1), Q = 1 + 3t (deg 1) -> 2/3.
    expr = _rational([1.0, 2.0], [3.0])
    assert math.isclose(expr.horizontal_asymptote(), 2.0 / 3.0)


def test_horizontal_asymptote_numerator_smaller_degree_is_zero() -> None:
    # P = 1 (deg 0), Q = 1 + 3t (deg 1) -> 0.
    expr = _rational([1.0], [3.0])
    assert expr.horizontal_asymptote() == 0.0


def test_horizontal_asymptote_numerator_larger_degree_diverges() -> None:
    # P = 1 + 2t + 5t^2 (deg 2), Q = 1 + 3t (deg 1) -> +inf.
    expr = _rational([1.0, 2.0, 5.0], [3.0])
    assert math.isinf(expr.horizontal_asymptote())


def test_recognize_logistic_saturation() -> None:
    eq = SparseEquation(
        term_names=("y", "y^2"),
        coefficients=np.array([1.0, -1.0]),
        intercept=0.0,
        alpha=0.0,
        threshold=0.0,
        active_mask=np.array([True, True]),
    )
    recognized = recognize_known_expression(eq, lhs="dy")
    assert recognized is not None
    assert recognized.family == "logistic"


def test_recognize_tanh_still_works() -> None:
    eq = SparseEquation(
        term_names=("y^2",),
        coefficients=np.array([-1.0]),
        intercept=1.0,
        alpha=0.0,
        threshold=0.0,
        active_mask=np.array([True]),
    )
    recognized = recognize_known_expression(eq, lhs="dy")
    assert recognized is not None
    assert recognized.family == "tanh_riccati"

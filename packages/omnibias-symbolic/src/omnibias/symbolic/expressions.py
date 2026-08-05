# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Explicit-expression discovery helpers.

This module is deliberately small and conservative. It handles two useful
cases:

* recognizing simple discovered jet laws as known function families;
* fitting a rational/Pade-style explicit surrogate with analytic derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from omnibias.symbolic.discovery import SparseEquation, rmse


@dataclass(frozen=True)
class RecognizedExpression:
    family: str
    expression: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class RationalExpression:
    """Rational expression ``P(t) / Q(t)`` over scaled coordinate ``t``."""

    numerator: np.ndarray
    denominator_tail: np.ndarray
    x_shift: float
    x_scale: float
    variable: str = "x"
    name: str = "rational"

    @property
    def denominator(self) -> np.ndarray:
        return np.concatenate([np.ones(1), self.denominator_tail])

    def evaluate(self, x: np.ndarray, *, derivative_order: int = 0) -> np.ndarray:
        if derivative_order < 0:
            raise ValueError(f"derivative_order must be >= 0, got {derivative_order}")
        x = np.asarray(x, dtype=float)
        t = (x - self.x_shift) / self.x_scale
        deriv_t = self._evaluate_t_derivatives(t, derivative_order)
        return deriv_t[derivative_order] / (self.x_scale**derivative_order)

    def residual_blasius(self, eta: np.ndarray) -> np.ndarray:
        f = self.evaluate(eta, derivative_order=0)
        fpp = self.evaluate(eta, derivative_order=2)
        fppp = self.evaluate(eta, derivative_order=3)
        return fppp + 0.5 * f * fpp

    def formula(self, *, digits: int = 6) -> str:
        p = _poly_formula(self.numerator, "t", digits=digits)
        q = _poly_formula(self.denominator, "t", digits=digits)
        return f"{self.name}({self.variable}) = ({p}) / ({q}), t = ({self.variable} - {self.x_shift:.{digits}g})/{self.x_scale:.{digits}g}"

    def horizontal_asymptote(self, *, atol: float = 1e-12) -> float:
        """Limit ``x -> +/- inf`` of ``P(t)/Q(t)`` (finite iff deg P <= deg Q).

        Because ``t = (x - shift)/scale`` is affine, the limit at ``x = +/- inf``
        is the limit of ``P(t)/Q(t)`` at ``t = +/- inf``: the ratio of leading
        coefficients when numerator and denominator share the same degree, ``0``
        when the numerator degree is smaller, and ``+/- inf`` when it is larger
        (no finite horizontal asymptote). This is the explicit-expression
        counterpart of the jet ``lim`` operator.
        """
        num = np.asarray(self.numerator, dtype=float)
        den = np.asarray(self.denominator, dtype=float)
        p = _degree(num, atol)
        q = _degree(den, atol)
        if p < q:
            return 0.0
        if p == q:
            return float(num[p] / den[q])
        return float(np.sign(num[p] / den[q]) * np.inf)

    def _evaluate_t_derivatives(self, t: np.ndarray, max_order: int) -> list[np.ndarray]:
        p_derivs = [_polyval_derivative(self.numerator, t, order) for order in range(max_order + 1)]
        q_derivs = [_polyval_derivative(self.denominator, t, order) for order in range(max_order + 1)]
        out = [p_derivs[0] / q_derivs[0]]
        for order in range(1, max_order + 1):
            accum = p_derivs[order].copy()
            for k in range(1, order + 1):
                accum = accum - math.comb(order, k) * q_derivs[k] * out[order - k]
            out.append(accum / q_derivs[0])
        return out


def recognize_known_expression(equation: SparseEquation, *, lhs: str) -> RecognizedExpression | None:
    """Recognize a few canonical jet identities as known functions."""

    terms = {name: coef for name, coef in zip(equation.term_names, equation.coefficients, strict=False) if abs(coef) > 1e-8}
    intercept = equation.intercept
    if lhs == "dy" and _close(intercept, 0.0) and _one_term(terms, "y", 1.0):
        return RecognizedExpression("exponential", "y(x) = C*exp(x)", 1.0, "recognized dy = y")
    if lhs == "d2y" and _close(intercept, 0.0) and _one_term(terms, "y", -1.0):
        return RecognizedExpression(
            "harmonic",
            "y(x) = A*sin(x) + B*cos(x)",
            1.0,
            "recognized d2y = -y",
        )
    if lhs == "dy" and _close(intercept, 1.0, atol=1e-4) and _one_term(terms, "y^2", -1.0, atol=1e-4):
        return RecognizedExpression("tanh_riccati", "y(x) = tanh(x + C)", 0.99, "recognized dy = 1 - y^2")
    if lhs == "dy" and _close(intercept, 0.0) and _two_terms(terms, "y", 1.0, "y^2", -1.0, atol=1e-4):
        return RecognizedExpression(
            "logistic",
            "y(x) = 1 / (1 + C*exp(-x))",
            0.99,
            "recognized dy = y - y^2 (logistic saturation to the interval [0, 1])",
        )
    return None


def fit_rational_expression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    numerator_degree: int,
    denominator_degree: int,
    ridge: float = 1e-12,
    x_shift: float | None = None,
    x_scale: float | None = None,
    variable: str = "x",
    name: str = "rational",
) -> RationalExpression:
    """Fit ``P(t)/Q(t)`` with ``Q(0)=1`` by linear least squares."""

    if numerator_degree < 0:
        raise ValueError("numerator_degree must be non-negative")
    if denominator_degree < 0:
        raise ValueError("denominator_degree must be non-negative")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_shift is None:
        x_shift = float(np.min(x))
    if x_scale is None:
        span = float(np.max(x) - np.min(x))
        x_scale = span if span > 1e-12 else 1.0
    t = (x - x_shift) / x_scale

    p_cols = [t**power for power in range(numerator_degree + 1)]
    q_cols = [-(y * (t**power)) for power in range(1, denominator_degree + 1)]
    design = np.stack([*p_cols, *q_cols], axis=1)
    target = y
    reg = ridge * np.eye(design.shape[1])
    coef = np.linalg.solve(design.T @ design + reg, design.T @ target)
    numerator = coef[: numerator_degree + 1]
    denominator_tail = coef[numerator_degree + 1 :]
    return RationalExpression(
        numerator=numerator,
        denominator_tail=denominator_tail,
        x_shift=x_shift,
        x_scale=x_scale,
        variable=variable,
        name=name,
    )


def validate_rational_expression(
    expr: RationalExpression,
    x: np.ndarray,
    y_reference: np.ndarray,
) -> dict[str, float]:
    pred = expr.evaluate(x)
    return {
        "value_rmse": rmse(y_reference, pred),
        "value_max_abs": float(np.max(np.abs(pred - y_reference))),
    }


def _polyval_derivative(coef: np.ndarray, x: np.ndarray, order: int) -> np.ndarray:
    c = np.asarray(coef, dtype=float)
    if order:
        for _ in range(order):
            if c.size <= 1:
                return np.zeros_like(x, dtype=float)
            c = np.asarray([power * c[power] for power in range(1, c.size)], dtype=float)
    out = np.zeros_like(x, dtype=float)
    for value in reversed(c):
        out = out * x + value
    return out


def _poly_formula(coef: np.ndarray, variable: str, *, digits: int) -> str:
    pieces = []
    for power, value in enumerate(coef):
        if abs(value) < 1e-12:
            continue
        mag = abs(value)
        if power == 0:
            term = f"{mag:.{digits}g}"
        elif power == 1:
            term = f"{mag:.{digits}g}*{variable}"
        else:
            term = f"{mag:.{digits}g}*{variable}^{power}"
        sign = "+" if value >= 0 else "-"
        if pieces:
            pieces.append(f"{sign} {term}")
        else:
            pieces.append(term if sign == "+" else f"-{term}")
    return " ".join(pieces) if pieces else "0"


def _one_term(terms: dict[str, float], name: str, value: float, *, atol: float = 1e-6) -> bool:
    return len(terms) == 1 and name in terms and _close(terms[name], value, atol=atol)


def _two_terms(
    terms: dict[str, float],
    name_a: str,
    value_a: float,
    name_b: str,
    value_b: float,
    *,
    atol: float = 1e-6,
) -> bool:
    return (
        len(terms) == 2
        and name_a in terms
        and name_b in terms
        and _close(terms[name_a], value_a, atol=atol)
        and _close(terms[name_b], value_b, atol=atol)
    )


def _degree(coef: np.ndarray, atol: float) -> int:
    """Highest index with a non-negligible coefficient (0 if all vanish)."""
    nz = np.nonzero(np.abs(coef) > atol)[0]
    return int(nz[-1]) if nz.size else 0


def _close(a: float, b: float, *, atol: float = 1e-6) -> bool:
    return abs(a - b) <= atol

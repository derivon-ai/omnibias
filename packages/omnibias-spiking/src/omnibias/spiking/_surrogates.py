# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form surrogate-derivative dispatch built from omnibias-core coeffs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import TypeVar

from omnibias.core.polynomials import hermite_coeffs, sigmoid_polynomial_coeffs

TensorT = TypeVar("TensorT")

SURROGATE_KINDS: tuple[str, ...] = ("fast_sigmoid", "gaussian")

_SURROGATE_ALIASES: Mapping[str, str] = {
    "fast_sigmoid": "fast_sigmoid",
    "fast-sigmoid": "fast_sigmoid",
    "gaussian": "gaussian",
}


def normalize_surrogate_kind(kind: str) -> str:
    """Map a user-facing surrogate name to a canonical key."""
    key = kind.lower().replace("-", "_")
    if key not in _SURROGATE_ALIASES:
        msg = f"unknown surrogate {kind!r}; expected one of {sorted(SURROGATE_KINDS)}"
        raise ValueError(msg)
    return _SURROGATE_ALIASES[key]


def _horner(coeffs: tuple[float, ...], x: TensorT) -> TensorT:
    """Stable Horner evaluation of ``sum_k coeffs[k] * x^k``."""
    deg = len(coeffs) - 1
    result: TensorT = coeffs[deg]  # type: ignore[assignment]
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]  # type: ignore[operator, assignment]
    return result


def _eval_hermite(coeffs: tuple[float, ...], z: TensorT) -> TensorT:
    return _horner(coeffs, z)


def fast_sigmoid_derivative(u: TensorT, *, sigmoid: Callable[[TensorT], TensorT]) -> TensorT:
    """``sigmoid'(u) = P_1(sigmoid(u))`` via :func:`sigmoid_polynomial_coeffs`."""
    coeffs = sigmoid_polynomial_coeffs(1)
    s = sigmoid(u)
    return _horner(coeffs, s)


def gaussian_derivative(u: TensorT, *, exp: Callable[[TensorT], TensorT]) -> TensorT:
    """Gaussian bump ``exp(-u^2/2) / sqrt(2 pi)`` via :func:`hermite_coeffs(0)`."""
    coeffs = hermite_coeffs(0)
    g = exp(-0.5 * u * u)
    he0 = _eval_hermite(coeffs, u)
    return g * he0 / math.sqrt(2.0 * math.pi)


__all__ = [
    "SURROGATE_KINDS",
    "fast_sigmoid_derivative",
    "gaussian_derivative",
    "normalize_surrogate_kind",
]

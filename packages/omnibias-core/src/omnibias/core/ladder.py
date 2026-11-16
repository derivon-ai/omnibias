# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hermite ladder algebra (theory 02-10).

The raw tower ``h_n = He_n(x) exp(-x^2/2)`` is **not** the quantum-harmonic
oscillator eigenbasis. Rodrigues reweighting produces
``psi_n = H_n(x) exp(-x^2/2)``. ``Normalization`` is required, never defaulted.
"""

from __future__ import annotations

import math
from enum import Enum

from omnibias.core.polynomials import hermite_coeffs

_SQRT2 = math.sqrt(2.0)


class Normalization(str, Enum):
    TOWER = "tower"
    OSCILLATOR = "oscillator"


def _horner(coeffs: tuple[float, ...], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _he(n: int, x: float) -> float:
    return _horner(hermite_coeffs(n), x)


def rodrigues_reweight(x: float) -> float:
    """``exp(x^2 / 2)``; converts the tempered tower of ``exp(-x^2)`` into ``psi_n``."""
    return math.exp(0.5 * x * x)


def hermite_function(
    n: int,
    x: float,
    *,
    normalization: Normalization,
    scale: float = 1.0,
    centre: float = 0.0,
) -> float:
    """Evaluate ``h_n`` or ``psi_n``. ``normalization`` is required."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    z = (float(x) - float(centre)) / float(scale)
    gauss = math.exp(-0.5 * z * z)
    if normalization is Normalization.TOWER:
        return _he(n, z) * gauss
    if normalization is Normalization.OSCILLATOR:
        # H_n(z) = 2^{n/2} He_n(z sqrt(2))
        return float((2.0 ** (0.5 * n)) * _he(n, z * _SQRT2) * gauss)
    raise ValueError(f"unknown normalization {normalization!r}")


def tower_raise(n: int, x: float) -> float:
    """``h_{n+1}(x) = - d/dx h_n(x)``."""
    return hermite_function(n + 1, x, normalization=Normalization.TOWER)


def tower_lower(n: int, x: float) -> float:
    """``(x + d/dx) h_n = n h_{n-1}``; ``n = 0`` yields 0."""
    if n <= 0:
        return 0.0
    return float(n) * hermite_function(n - 1, x, normalization=Normalization.TOWER)


def number_operator_action(n: int, x: float) -> float:
    """``N h_n = n h_n`` with ``N = -(d^2/dx^2 + x d/dx + 1)``."""
    return float(n) * hermite_function(n, x, normalization=Normalization.TOWER)


def oscillator_action(n: int, x: float) -> float:
    """``H psi_n = (n + 1/2) psi_n``."""
    return (float(n) + 0.5) * hermite_function(n, x, normalization=Normalization.OSCILLATOR)


def number_operator_apply(n: int, x: float) -> float:
    """Evaluate ``N h_n`` from Hermite derivatives (not via ``h_{n+2}``)."""
    g = math.exp(-0.5 * x * x)
    he = _he(n, x)
    he_p = 0.0 if n == 0 else float(n) * _he(n - 1, x)
    he_pp = 0.0 if n < 2 else float(n * (n - 1)) * _he(n - 2, x)
    h = he * g
    hp = (he_p - x * he) * g
    hpp = (he_pp - he - x * he_p) * g + (he_p - x * he) * (-x) * g
    return -(hpp + x * hp + h)


def _physicist_H(n: int, x: float) -> float:
    return float((2.0 ** (0.5 * n)) * _he(n, x * _SQRT2))


def oscillator_hamiltonian_apply(
    n: int, x: float, *, normalization: Normalization
) -> float:
    """``H = -1/2 d^2/dx^2 + x^2/2`` applied via closed-form derivatives."""
    if normalization is Normalization.TOWER:
        hpp = hermite_function(n + 2, x, normalization=Normalization.TOWER)
        h = hermite_function(n, x, normalization=Normalization.TOWER)
        return -0.5 * hpp + 0.5 * x * x * h
    g = math.exp(-0.5 * x * x)
    hpoly = _physicist_H(n, x)
    hp = 0.0 if n == 0 else 2.0 * n * _physicist_H(n - 1, x)
    hpp_poly = 0.0 if n < 2 else 4.0 * n * (n - 1) * _physicist_H(n - 2, x)
    psi = hpoly * g
    psi_pp = (hpp_poly - 2.0 * x * hp + (x * x - 1.0) * hpoly) * g
    return -0.5 * psi_pp + 0.5 * x * x * psi


def commutator_residual(n_max: int, x: float = 0.7) -> float:
    """``|[L, R] - 1|`` on ``h_n`` for ``n = 0 .. n_max-1`` (top level truncated)."""
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    worst = 0.0
    for n in range(n_max):
        h = hermite_function(n, x, normalization=Normalization.TOWER)
        # [L, R] h_n = L h_{n+1} - R (L h_n) = (n+1) h_n - n h_n
        lrh = tower_lower(n + 1, x)
        rlh = float(n) * h if n > 0 else 0.0
        comm = lrh - rlh
        err = abs(comm - h)
        if err > worst:
            worst = err
    return worst


__all__ = [
    "Normalization",
    "commutator_residual",
    "hermite_function",
    "number_operator_action",
    "number_operator_apply",
    "oscillator_action",
    "oscillator_hamiltonian_apply",
    "rodrigues_reweight",
    "tower_lower",
    "tower_raise",
]

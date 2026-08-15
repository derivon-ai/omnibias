# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fast (non-verified) Cauchy-Hardy dictionary (theory 01-12).

The Hilbert transform on the line is a signed permutation of dictionary
coefficients: no quadrature. Commutation ``H[f^(n)] = (H[f])^(n)`` needs
decay (``alpha > 0``). This is the line Hilbert
``H[f](x) = (1/pi) p.v. int f(t)/(x-t) dt``, not a periodic or
finite-interval operator.

G5 (dictionary capacity vs the CCF stretch residual) is a campaign
artifact, **not** a claim that ``CCF_STRETCH_RESIDUAL_GATE`` is cleared.

Pure Python: no tensor imports. Training-loop twins live in
``omnibias.{torch,jax}.conjugate``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.verified.hardy_line import pochhammer

Parity = Literal["even", "odd"]


def _validate_atom(scale: float, exponent: float, order: int) -> None:
    if not (scale > 0.0 and math.isfinite(scale)):
        raise ValueError(f"Hardy scale a must be finite and > 0, got {scale!r}")
    if not math.isfinite(exponent):
        raise ValueError(f"Hardy exponent alpha must be finite, got {exponent!r}")
    if order < 0:
        raise ValueError(f"derivative order n must be >= 0, got {order}")


def hardy_p(y: float, a: float, alpha: float) -> float:
    """``P_{a,alpha}(y) = r^{-alpha} cos(alpha phi)``."""
    r = math.hypot(a, y)
    phi = math.atan2(y, a)
    return float(r ** (-alpha) * math.cos(alpha * phi))


def hardy_q(y: float, a: float, alpha: float) -> float:
    """``Q_{a,alpha}(y) = r^{-alpha} sin(alpha phi)``."""
    r = math.hypot(a, y)
    phi = math.atan2(y, a)
    return float(r ** (-alpha) * math.sin(alpha * phi))


def _table_kind(n: int) -> tuple[int, Parity, int, Parity]:
    r = n % 4
    if r == 0:
        return 1, "even", 1, "odd"
    if r == 1:
        return -1, "odd", 1, "even"
    if r == 2:
        return -1, "even", -1, "odd"
    return 1, "odd", -1, "even"


def hardy_p_deriv_n(y: float, a: float, alpha: float, n: int) -> float:
    """Fast ``P^(n)`` via the closed table."""
    _validate_atom(a, alpha, n)
    if n == 0:
        return hardy_p(y, a, alpha)
    p_sign, p_kind, _, _ = _table_kind(n)
    factor = pochhammer(alpha, n)
    atom = hardy_p(y, a, alpha + n) if p_kind == "even" else hardy_q(y, a, alpha + n)
    return float(p_sign) * factor * atom


def hardy_q_deriv_n(y: float, a: float, alpha: float, n: int) -> float:
    """Fast ``Q^(n)`` via the closed table."""
    _validate_atom(a, alpha, n)
    if n == 0:
        return hardy_q(y, a, alpha)
    _, _, q_sign, q_kind = _table_kind(n)
    factor = pochhammer(alpha, n)
    atom = hardy_p(y, a, alpha + n) if q_kind == "even" else hardy_q(y, a, alpha + n)
    return float(q_sign) * factor * atom


@dataclass(frozen=True)
class HardyAtom:
    """One dictionary atom ``P^(n)`` or ``Q^(n)`` at scale ``a`` and exponent ``alpha``."""

    scale: float
    exponent: float
    order: int
    parity: Parity

    def __post_init__(self) -> None:
        _validate_atom(self.scale, self.exponent, self.order)
        if self.parity not in ("even", "odd"):
            raise ValueError(f"parity must be 'even' or 'odd', got {self.parity!r}")

    def evaluate(self, y: float) -> float:
        if self.parity == "even":
            return hardy_p_deriv_n(y, self.scale, self.exponent, self.order)
        return hardy_q_deriv_n(y, self.scale, self.exponent, self.order)

    def key(self) -> tuple[float, float, int]:
        return (self.scale, self.exponent, self.order)


@dataclass(frozen=True)
class HardyDictionary:
    """Finite Hardy span, closed under ``H`` when both parities are present.

    ``H`` acts on coefficients as a signed permutation. No quadrature.
    Commutation with derivatives needs ``alpha > 0`` on every atom.
    """

    atoms: tuple[HardyAtom, ...]

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ValueError("HardyDictionary needs at least one atom")

    def hilbert_permutation(self) -> tuple[tuple[int, float], ...]:
        """For each atom ``i``, ``(j, sign)`` so ``H[atom_i] = sign * atom_j``.

        ``H[P^(n)] = Q^(n)`` (sign ``+1``) and ``H[Q^(n)] = -P^(n)`` (sign ``-1``).
        Both parities of each ``(a, alpha, n)`` must be present.
        """
        index: dict[tuple[float, float, int, Parity], int] = {}
        for i, atom in enumerate(self.atoms):
            index[(atom.scale, atom.exponent, atom.order, atom.parity)] = i
        out: list[tuple[int, float]] = []
        for atom in self.atoms:
            if not (atom.exponent > 0.0):
                raise ValueError(
                    "Hilbert-derivative commutation needs decay (alpha > 0); "
                    f"got alpha={atom.exponent!r}"
                )
            opposite: Parity = "odd" if atom.parity == "even" else "even"
            key = (atom.scale, atom.exponent, atom.order, opposite)
            if key not in index:
                raise ValueError(
                    "H-closed dictionary needs both parities of each "
                    f"(a, alpha, n)={atom.key()!r}"
                )
            sign = 1.0 if atom.parity == "even" else -1.0
            out.append((index[key], sign))
        return tuple(out)


def evaluate(dictionary: HardyDictionary, y: float) -> tuple[float, ...]:
    """Atom values at a scalar ``y``."""
    return tuple(atom.evaluate(y) for atom in dictionary.atoms)


def hilbert(dictionary: HardyDictionary, coeffs: Sequence[float]) -> tuple[float, ...]:
    """Apply the signed permutation. No quadrature."""
    if len(coeffs) != len(dictionary.atoms):
        raise ValueError("coeffs length must match the dictionary")
    perm = dictionary.hilbert_permutation()
    out = [0.0] * len(coeffs)
    for i, (j, sign) in enumerate(perm):
        out[j] += sign * float(coeffs[i])
    return tuple(out)


__all__ = [
    "HardyAtom",
    "HardyDictionary",
    "evaluate",
    "hardy_p",
    "hardy_p_deriv_n",
    "hardy_q",
    "hardy_q_deriv_n",
    "hilbert",
]

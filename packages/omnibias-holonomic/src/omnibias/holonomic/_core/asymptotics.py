# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Leading asymptotics of a P-recursive sequence (Birkhoff-Trjitzinsky / Poincare-Perron).

For ``a_n`` annihilated by ``sum_{i=0}^r p_i(n) a_{n+i} = 0`` the growth is governed, in the
regular (distinct-dominant-root) case, by the **characteristic polynomial** built from the
top-degree coefficients: with ``d = max_i deg p_i`` and ``lc_i`` the coefficient of ``n^d`` in
``p_i``,

.. math::

    \chi(t) = \sum_i lc_i\, t^i, \qquad a_{n+1}/a_n \to \rho \ (\text{a root of } \chi),
    \qquad a_n \sim C\,\rho^n\, n^{\theta},

and the sub-exponential exponent follows from the next coefficient balance,
``theta = -\,\Sigma(\rho) / (\rho\,\chi'(\rho))`` with ``\Sigma`` built from the ``n^{d-1}``
coefficients. :func:`precursive_asymptotics` returns the dominant rate ``rho`` and exponent
``theta``.

**Honesty / scope.** This is a **numerical** leading asymptotic (the dominant root and
``theta`` are floats), scoped to the regular case with a single dominant characteristic root;
a sub-maximal leading coefficient (``lc_r = 0``) is reported as ``kind="factorial"`` (growth
faster than any geometric) rather than forced into a rate. Where the generating function's
singularity is known exactly, :func:`certified_asymptotic` bridges to
:func:`omnibias.difference.transfer_theorem` for a **certified** (interval-enclosed)
coefficient -- exact when the scale / radius are rational. The constant ``C`` itself is not
recovered here (it needs the full singular analysis of the OGF).
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.holonomic._core.dfinite import PRecursive
from omnibias.holonomic._core.rational_poly import degree

if TYPE_CHECKING:
    from omnibias.difference._core.generating import TransferEstimate

Rational = Fraction | int


@dataclass(frozen=True)
class AsymptoticEstimate:
    """The leading asymptotic ``a_n ~ C rate^n n^exponent`` of a P-recursive sequence."""

    kind: str
    rate: float
    signed_rate: float | None
    exponent: float
    char_coeffs: tuple[float, ...]
    note: str = ""

    def model(self, n: int) -> float:
        """The shape ``base^n n^exponent`` (drops the unknown constant ``C``); ``n >= 1``."""
        base = self.signed_rate if self.signed_rate is not None else self.rate
        return float(base**n * float(n) ** self.exponent)


def _char_coeffs(coeffs: list[tuple[Fraction, ...]], d: int, sub: int) -> list[Fraction]:
    """Coefficient of ``n^sub`` in each ``p_i`` (padded to zero); ``sub in {d, d-1}``."""
    out: list[Fraction] = []
    for c in coeffs:
        out.append(c[sub] if 0 <= sub < len(c) else Fraction(0))
    return out


def _roots(coeffs: list[float]) -> list[complex]:
    """Complex roots of ``sum_i coeffs[i] t^i`` (ascending order).

    Closed form to degree 2; a pure-Python Durand-Kerner (Weierstrass) iteration above
    (the core stays numpy-free, per the package purity invariant).
    """
    trimmed = list(coeffs)
    while trimmed and abs(trimmed[-1]) == 0.0:
        trimmed.pop()
    deg = len(trimmed) - 1
    if deg <= 0:
        return []
    if deg == 1:
        return [complex(-trimmed[0] / trimmed[1])]
    if deg == 2:
        a, b, c = trimmed[2], trimmed[1], trimmed[0]
        disc = cmath.sqrt(b * b - 4 * a * c)
        return [(-b + disc) / (2 * a), (-b - disc) / (2 * a)]
    lead = trimmed[-1]
    monic = [c / lead for c in trimmed]  # ascending, monic

    def _peval(x: complex) -> complex:
        acc = 0j
        for c in reversed(monic):
            acc = acc * x + c
        return acc

    roots = [(0.4 + 0.9j) ** k for k in range(deg)]
    for _ in range(500):
        max_delta = 0.0
        updated = roots[:]
        for i in range(deg):
            xi = roots[i]
            denom = 1 + 0j
            for j in range(deg):
                if j != i:
                    denom *= xi - roots[j]
            if denom == 0:
                continue
            delta = _peval(xi) / denom
            updated[i] = xi - delta
            max_delta = max(max_delta, abs(delta))
        roots = updated
        if max_delta < 1e-14:
            break
    return roots


def precursive_asymptotics(rec: PRecursive, *, tol: float = 1e-9) -> AsymptoticEstimate:
    r"""The Poincare-Perron leading asymptotic of ``rec`` (numerical; regular case)."""
    op = rec.annihilator
    r = op.order
    coeffs = [op.coeffs[i] if i < len(op.coeffs) else () for i in range(r + 1)]
    d = max((degree(c) for c in coeffs if c), default=-1)
    if d < 0:
        raise ValueError("empty recurrence operator")
    lc = _char_coeffs(coeffs, d, d)
    sl = _char_coeffs(coeffs, d, d - 1) if d >= 1 else [Fraction(0)] * (r + 1)
    char = [float(x) for x in lc]
    if lc[r] == 0:
        # The top-shift coefficient has sub-maximal degree: the characteristic polynomial
        # loses degree, the shift ratio a_{n+1}/a_n grows without bound, and the dominant
        # solution is super-exponential (factorial-type). Honestly out of the geometric scope.
        return AsymptoticEstimate(
            kind="factorial",
            rate=float("inf"),
            signed_rate=None,
            exponent=float("nan"),
            char_coeffs=tuple(char),
            note="top-shift coefficient has sub-maximal degree: super-exponential growth",
        )
    roots = _roots(char)
    if not roots:
        raise ValueError("degenerate characteristic polynomial (no non-zero rate)")
    dominant = max(roots, key=lambda z: (abs(z), z.real))
    is_real = abs(dominant.imag) <= tol * max(1.0, abs(dominant))
    signed = dominant.real if is_real else None
    if is_real:
        # rho * chi'(rho) = sum_i i lc_i rho^i ; Sigma = sum_i sl_i rho^i.
        rho = dominant.real
        rho_chi_prime = sum(i * float(lc[i]) * rho**i for i in range(len(lc)))
        sigma = sum(float(sl[i]) * rho**i for i in range(len(sl)))
        theta = -sigma / rho_chi_prime if rho_chi_prime != 0 else float("nan")
    else:
        theta = float("nan")
    if abs(abs(dominant) - 1.0) <= tol:
        kind, note = "polynomial", "dominant characteristic root on the unit circle"
    else:
        kind, note = "geometric", ""
    return AsymptoticEstimate(
        kind=kind,
        rate=abs(dominant),
        signed_rate=signed,
        exponent=theta,
        char_coeffs=tuple(char),
        note=note,
    )


def empirical_rate(rec: PRecursive, *, samples: int = 80) -> float:
    """The empirical ratio ``a_{n+1}/a_n`` at the largest available index (baseline estimate)."""
    terms = rec.terms(samples)
    for n in range(len(terms) - 1, 0, -1):
        if terms[n - 1] != 0 and terms[n] != 0:
            return float(terms[n]) / float(terms[n - 1])
    raise ValueError("sequence has too many zeros to estimate a ratio")


def certified_asymptotic(
    rate: Rational, exponent_alpha: Rational, scale: Rational | float, n: int
) -> TransferEstimate:
    r"""Certified coefficient asymptotic via :func:`omnibias.difference.transfer_theorem`.

    Bridges a known singularity: rate ``rho`` -> singular radius ``1/rho`` and ``exponent_alpha
    = theta + 1`` (so ``a_n ~ scale * rho^n * n^{alpha - 1} / Gamma(alpha)``). Returns the
    ``TransferEstimate`` whose ``exact`` field is a rigorous :class:`Interval` (exact when
    ``scale`` and ``rate`` are rational). ``rate`` must be non-zero.
    """
    from omnibias.difference import transfer_theorem

    if rate == 0:
        raise ValueError("rate (growth ratio) must be non-zero")
    radius = Fraction(1) / Fraction(rate)
    return transfer_theorem(scale, radius, Fraction(exponent_alpha), n)


__all__ = [
    "AsymptoticEstimate",
    "certified_asymptotic",
    "empirical_rate",
    "precursive_asymptotics",
]

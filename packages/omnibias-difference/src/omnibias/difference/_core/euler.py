# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Euler (secant) numbers read off the closed-form ``sech`` tower (exact).

The **Euler numbers** ``E_n = sech^(n)(0)`` are the derivatives of
``sech(x) = 1/cosh(x)`` at the origin. ``sech`` closes its whole derivative tower
on ``Q_n(tanh) * sech`` (the tanh/sech Riccati recurrence in
:func:`omnibias.core.polynomials.sech_polynomial_coeffs`), so ``E_n`` is just the
constant term ``Q_n(0)`` of that tower -- read straight off it, exact by
construction (``E_0 = 1``, ``E_2 = -1``, ``E_4 = 5``, ``E_6 = -61``, ...; odd
indices vanish).

Do **not** confuse the *Euler numbers* ``E_n`` (secant numbers, above) with the
*Eulerian numbers* ``A(n, k)`` (the triangle counting permutation ascents).
:func:`eulerian_number` computes the latter from the classical **Worpitzky
summation** -- they are *not* the coefficients of the logistic-sigmoid derivative
tower (verified: ``P_3 = (0, 1, -7, 12, -6)`` is not the Eulerian row
``(1, 4, 1)``), so they are computed here rather than read off a tower.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from omnibias.core.verified.coeffs import sech_poly_coeffs_exact


def euler_number(n: int) -> int:
    """The ``n``-th Euler (secant) number ``E_n = sech^(n)(0) = Q_n(0)`` (exact int)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return sech_poly_coeffs_exact(n)[0]


def euler_polynomial(n: int) -> tuple[Fraction, ...]:
    r"""Coefficients of the Euler polynomial ``E_n(x)``.

    Uses ``E_n(x) = sum_k C(n,k) (E_k / 2^k) (x - 1/2)^{n-k}`` (with ``E_k`` the
    Euler numbers), so ``E_n(1/2) = E_n / 2^n``. Returns ``(c_0, ..., c_n)`` with
    ``E_n(x) = sum_j c_j x^j`` (e.g. ``E_1(x) = x - 1/2``, ``E_2(x) = x^2 - x``).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    out = [Fraction(0)] * (n + 1)
    for k in range(n + 1):
        coeff = Fraction(comb(n, k)) * Fraction(euler_number(k), 2**k)
        m = n - k
        for j in range(m + 1):
            out[j] += coeff * Fraction(comb(m, j)) * Fraction((-1) ** (m - j), 2 ** (m - j))
    return tuple(out)


def eulerian_number(n: int, k: int) -> int:
    r"""Eulerian number ``A(n, k)`` via the Worpitzky summation (classical, not a tower).

    ``A(n, k) = sum_{j=0}^{k} (-1)^j C(n+1, j) (k+1-j)^n`` counts permutations of
    ``n`` with exactly ``k`` ascents (``0 <= k <= n-1`` for ``n >= 1``). This is the
    genuine Eulerian triangle; see the module docstring for why it is *not* read
    off the logistic-sigmoid tower.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return 1 if k == 0 else 0
    if k < 0 or k > n - 1:
        return 0
    return sum((-1) ** j * comb(n + 1, j) * (k + 1 - j) ** n for j in range(k + 1))


__all__ = [
    "euler_number",
    "euler_polynomial",
    "eulerian_number",
]

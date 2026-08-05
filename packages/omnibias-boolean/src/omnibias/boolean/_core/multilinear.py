# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The multilinear extension and the discrete <-> continuous derivative bridge.

For values ``v`` on the cube ``{0,1}^n`` the unique multilinear polynomial agreeing
with them is

.. math::

    F(x) = \sum_{a \in \{0,1\}^n} v[a] \prod_{i:\, a_i = 1} x_i \prod_{i:\, a_i = 0} (1 - x_i)
         = \sum_{S \subseteq [n]} m_S \prod_{i \in S} x_i ,

the *multilinear (Reed-Muller-over-the-reals) extension*. The monomial
coefficients ``m_S`` come from the real Mobius transform (a subtraction butterfly)
and satisfy the bridge identities used by the differentiable spectrum engine:

* ``m_S = d^{|S|} F / prod_{i in S} d x_i`` evaluated at ``x = 0`` -- the Mobius
  coefficient is a mixed partial, readable off a jet via ``jet_partials``;
* ``dF/dx_i = F|_{x_i=1} - F|_{x_i=0}`` -- the continuous partial is the arithmetic
  Boolean difference;
* for a ``{0,1}``-valued ``v``, ``m_S mod 2`` is the GF(2) ANF coefficient (the
  integer Mobius transform reduces to the binary one mod 2).
"""

from __future__ import annotations

from collections.abc import Sequence

Number = float | int


def _num_vars(size: int) -> int:
    if size < 1 or (size & (size - 1)) != 0:
        raise ValueError(f"values length must be a power of two, got {size}")
    return size.bit_length() - 1


def multilinear_coeffs(values: Sequence[Number]) -> tuple[Number, ...]:
    """Monomial coefficients ``m_S`` of the multilinear extension (real Mobius)."""
    a = list(values)
    size = len(a)
    _num_vars(size)
    step = 1
    while step < size:
        for i in range(0, size, step << 1):
            for j in range(i, i + step):
                a[j + step] = a[j + step] - a[j]
        step <<= 1
    return tuple(a)


def values_from_multilinear_coeffs(coeffs: Sequence[Number]) -> tuple[Number, ...]:
    """Inverse transform (zeta / addition butterfly): cube values from ``m_S``."""
    a = list(coeffs)
    size = len(a)
    _num_vars(size)
    step = 1
    while step < size:
        for i in range(0, size, step << 1):
            for j in range(i, i + step):
                a[j + step] = a[j + step] + a[j]
        step <<= 1
    return tuple(a)


def multilinear_eval(values: Sequence[Number], x: Sequence[float]) -> float:
    """Evaluate the multilinear extension ``F(x)`` from cube values via products."""
    size = len(values)
    n = _num_vars(size)
    if len(x) != n:
        raise ValueError(f"x must have length {n}, got {len(x)}")
    total = 0.0
    for a in range(size):
        prod = 1.0
        for i in range(n):
            prod *= x[i] if (a >> i) & 1 else (1.0 - x[i])
        total += float(values[a]) * prod
    return total


def multilinear_eval_from_coeffs(coeffs: Sequence[Number], x: Sequence[float]) -> float:
    """Evaluate ``F(x) = sum_S m_S prod_{i in S} x_i`` from monomial coefficients."""
    size = len(coeffs)
    n = _num_vars(size)
    if len(x) != n:
        raise ValueError(f"x must have length {n}, got {len(x)}")
    total = 0.0
    for s in range(size):
        coeff = coeffs[s]
        if coeff == 0:
            continue
        prod = 1.0
        for i in range(n):
            if (s >> i) & 1:
                prod *= x[i]
        total += float(coeff) * prod
    return total


def mixed_partial(values: Sequence[Number], subset: int, x: Sequence[float]) -> float:
    """Mixed partial ``d^{|subset|} F / prod_{i in subset} d x_i`` at ``x``.

    ``subset`` is a variable bitmask. At ``x = 0`` this returns ``m_subset``
    (the Mobius coefficient), demonstrating the partial-derivative bridge.
    """
    coeffs = multilinear_coeffs(values)
    size = len(coeffs)
    n = _num_vars(size)
    if len(x) != n:
        raise ValueError(f"x must have length {n}, got {len(x)}")
    total = 0.0
    for t in range(size):
        if (t & subset) != subset:
            continue
        coeff = coeffs[t]
        if coeff == 0:
            continue
        remaining = t & ~subset
        prod = 1.0
        for i in range(n):
            if (remaining >> i) & 1:
                prod *= x[i]
        total += float(coeff) * prod
    return total


def anf_from_multilinear_coeffs(coeffs: Sequence[Number]) -> tuple[int, ...]:
    """GF(2) ANF coefficients = integer Mobius coefficients reduced mod 2.

    Valid for a ``{0,1}``-valued function; the integer Mobius transform coincides
    with the binary one modulo 2.
    """
    return tuple(int(round(float(c))) & 1 for c in coeffs)


__all__ = [
    "anf_from_multilinear_coeffs",
    "mixed_partial",
    "multilinear_coeffs",
    "multilinear_eval",
    "multilinear_eval_from_coeffs",
    "values_from_multilinear_coeffs",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Boolean differential calculus: the Boolean derivative and its integral.

The **Boolean derivative** (Boolean difference) of ``f`` with respect to ``x_i`` is

.. math::

    \frac{\partial f}{\partial x_i}(x) = f|_{x_i = 0}(x) \oplus f|_{x_i = 1}(x),

a function independent of ``x_i`` that is ``1`` exactly where flipping ``x_i``
flips the output (fault sensitization in test generation; the per-bit term of
differential cryptanalysis). Iterating over a set ``S`` gives the mixed Boolean
*differential*.

Its inverse, **Boolean integration**, recovers ``f`` from a derivative ``g`` only
up to a free function ``c`` independent of ``x_i`` -- the discrete "constant of
integration":

.. math::

    f(x) = c(x_{\neq i}) \oplus \big(x_i \wedge g(x_{\neq i})\big),

and it exists iff ``g`` is itself independent of ``x_i`` (the discrete exactness
condition).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from omnibias.boolean._core.truth_table import TruthTable, check_truth_table, num_vars


def reduced_index(x: int, i: int, n: int) -> int:
    """Index of assignment ``x`` with bit ``i`` deleted (over the ``n-1`` rest)."""
    low = x & ((1 << i) - 1)
    high = (x >> (i + 1)) << i
    return low | high


def restrict(table: TruthTable, i: int, value: int) -> TruthTable:
    """Cofactor ``f|_{x_i = value}`` as a function of the remaining ``n-1`` variables."""
    check_truth_table(table)
    n = num_vars(table)
    if not 0 <= i < n:
        raise ValueError(f"variable index {i} out of range for n={n}")
    if value not in (0, 1):
        raise ValueError(f"value must be 0 or 1, got {value}")
    size = 1 << n
    out = [0] * (size >> 1)
    for x in range(size):
        if ((x >> i) & 1) == value:
            out[reduced_index(x, i, n)] = table[x]
    return tuple(out)


def boolean_derivative(table: TruthTable, i: int) -> TruthTable:
    """``df/dx_i`` as a full ``n``-variable truth table (constant in ``x_i``)."""
    check_truth_table(table)
    n = num_vars(table)
    if not 0 <= i < n:
        raise ValueError(f"variable index {i} out of range for n={n}")
    bit = 1 << i
    size = 1 << n
    out = [0] * size
    for x in range(size):
        out[x] = table[x & ~bit] ^ table[x | bit]
    return tuple(out)


def boolean_derivative_reduced(table: TruthTable, i: int) -> TruthTable:
    """``df/dx_i`` as an ``(n-1)``-variable truth table over the remaining vars."""
    return restrict(boolean_derivative(table, i), i, 0)


def boolean_derivative_set(table: TruthTable, indices: Iterable[int]) -> TruthTable:
    """Iterated mixed Boolean derivative over a set of variables (order-independent)."""
    result = table
    for i in sorted(set(indices)):
        result = boolean_derivative(result, i)
    return result


def is_independent_of(table: TruthTable, i: int) -> bool:
    """``True`` iff ``f`` does not depend on ``x_i`` (its Boolean derivative is 0)."""
    return all(v == 0 for v in boolean_derivative(table, i))


@dataclass(frozen=True)
class BooleanAntiderivative:
    """General antiderivative of ``g`` w.r.t. ``x_i`` with a free constant ``c``.

    ``general(c)`` builds ``f(x) = c(x_{!=i}) ^ (x_i & g(x))`` for an
    ``(n-1)``-variable constant ``c``; :attr:`particular` is the ``c = 0`` choice.
    """

    var: int
    n: int
    derivative: TruthTable
    particular: TruthTable

    def general(self, constant: TruthTable) -> TruthTable:
        """Antiderivative for a given ``(n-1)``-variable constant of integration."""
        expected = 1 << (self.n - 1)
        if len(constant) != expected:
            raise ValueError(
                f"constant must have length {expected} (n-1={self.n - 1} vars), "
                f"got {len(constant)}"
            )
        size = 1 << self.n
        out = [0] * size
        for x in range(size):
            xi = (x >> self.var) & 1
            c = constant[reduced_index(x, self.var, self.n)]
            out[x] = c ^ (xi & self.derivative[x])
        return tuple(out)


def boolean_integral(g: TruthTable, i: int) -> BooleanAntiderivative:
    """Antiderivative of ``g`` w.r.t. ``x_i`` (requires ``g`` independent of ``x_i``).

    Raises ``ValueError`` if ``g`` depends on ``x_i`` (no antiderivative exists --
    the discrete analogue of a non-exact differential).
    """
    check_truth_table(g)
    n = num_vars(g)
    if not 0 <= i < n:
        raise ValueError(f"variable index {i} out of range for n={n}")
    if not is_independent_of(g, i):
        raise ValueError(
            f"g depends on x_{i}; no antiderivative exists "
            "(integrability/exactness condition violated)"
        )
    zero_constant = tuple([0] * (1 << (n - 1)))
    anti = BooleanAntiderivative(var=i, n=n, derivative=g, particular=())
    particular = anti.general(zero_constant)
    return BooleanAntiderivative(var=i, n=n, derivative=g, particular=particular)


__all__ = [
    "BooleanAntiderivative",
    "boolean_derivative",
    "boolean_derivative_reduced",
    "boolean_derivative_set",
    "boolean_integral",
    "is_independent_of",
    "reduced_index",
    "restrict",
]

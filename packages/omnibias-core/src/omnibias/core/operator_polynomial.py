# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Compressed differential functionals for proven scalar program regions.

``B[k]`` multiplies the RAW outer derivative ``H**(k)(value)`` in
``D**alpha H(value) = sum(B[k] * H**(k)(value))``. This is deliberately
different from the Taylor-normalized activation towers accepted below.

Binary rules require disjoint selected spatial supports. They are not rules
for arbitrary overlapping dependencies. Coefficients may be ordinary tensors;
no backend imports, detached values, numerical branches, or partitions occur.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial
from operator import index as integer_index
from typing import Any

from omnibias.core.jet_algebra import JetBudget

ProductPlan = tuple[tuple[int, int, int, int], ...]
UnaryPlan = tuple[tuple[tuple[int, int], ...], ...]


def _order(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("operator polynomial order must be a nonnegative integer")
    try:
        order = integer_index(value)
    except TypeError as exc:
        raise ValueError("operator polynomial order must be a nonnegative integer") from exc
    if order < 0:
        raise ValueError("operator polynomial order must be nonnegative")
    return order


def _size(values: Sequence[Any]) -> int:
    if not values:
        raise ValueError("operator polynomial needs its constant coefficient")
    return len(values) - 1


def _scale(value: Any, weight: int) -> Any:
    # Exact references remain exact. Tensor backends may convert Python ints
    # through int64 before multiplication, which overflows already at 21!.
    return value * (weight if isinstance(value, (int, Fraction)) else float(weight))


def coordinate_polynomial(order: int, one: Any = 1, *, budget: JetBudget | None = None) -> list[Any]:
    """Coordinate leaf: D**order H(x) = H**(order)(x). Order zero returns [one]."""
    order = _order(order)
    (budget or JetBudget()).check_layout(order + 1, 0)
    return [one * 0] * order + [one]


def operator_sum(left: Sequence[Any], right: Sequence[Any], *,
                 budget: JetBudget | None = None) -> list[Any]:
    """Differential polynomial of a SUM with disjoint selected spatial supports."""
    n, m = _size(left), _size(right)
    (budget or JetBudget()).check_layout(n + m + 1, (n + 1) * (m + 1))
    result = [left[0] * 0 + right[0] * 0] * (n + m + 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] = result[i + j] + a * b
    return result


def _product_work(n: int, m: int) -> int:
    r = min(n, m)
    return ((r + 1) * (n + 1) * (m + 1) - (n + m + 2) * r * (r + 1) // 2
            + r * (r + 1) * (2 * r + 1) // 6)


def _make_product_plan(n: int, m: int) -> ProductPlan:
    return tuple((k, ell, j, comb(k, j) * comb(ell, j) * factorial(j))
                 for k in range(n + 1) for ell in range(m + 1) for j in range(min(k, ell) + 1))


@lru_cache(maxsize=32)
def _cached_product_plan(n: int, m: int) -> ProductPlan:
    return _make_product_plan(n, m)


def operator_product_plan(left_order: int, right_order: int, *,
                          budget: JetBudget | None = None) -> ProductPlan:
    """Bounded exact integer schedule. Prepare outside a backend's traced call."""
    n, m = _order(left_order), _order(right_order)
    work = _product_work(n, m)
    (budget or JetBudget()).check_layout(2 * (n + m) + 3, work)
    # Do not retain arbitrarily large schedules merely because one call allows
    # them. The process-wide cache holds at most 32 modest host-only plans.
    return _cached_product_plan(n, m) if work <= 16_384 else _make_product_plan(n, m)


def operator_product(left: Sequence[Any], right: Sequence[Any], left_value: Any, right_value: Any, *,
                     budget: JetBudget | None = None, plan: ProductPlan | None = None) -> list[Any]:
    """Differential polynomial of a PRODUCT with disjoint supports, including zeros.

    A term (k,l,j) contributes binom(k,j) binom(l,j) j! times
    ``left[k] right[l] right_value**(k-j) left_value**(l-j)`` to degree k+l-j.
    No quotient by an operand or derivative is used.
    """
    n, m = _size(left), _size(right)
    limits = budget or JetBudget()
    limits.check_layout(2 * (n + m) + 3, _product_work(n, m))
    schedule = operator_product_plan(n, m, budget=limits) if plan is None else plan
    if len(schedule) != _product_work(n, m):
        raise ValueError("product schedule does not match the operand orders")
    zero = left[0] * 0 + right[0] * 0
    left_powers = [left_value * 0 + 1]
    right_powers = [right_value * 0 + 1]
    for _ in range(m):
        left_powers.append(left_powers[-1] * left_value)
    for _ in range(n):
        right_powers.append(right_powers[-1] * right_value)
    result = [zero] * (n + m + 1)
    for k, ell, j, weight in schedule:
        value = left[k] * right[ell] * right_powers[k - j] * left_powers[ell - j]
        result[k + ell - j] = result[k + ell - j] + _scale(value, weight)
    return result


@lru_cache(maxsize=32)
def _cached_unary_plan(n: int) -> UnaryPlan:
    return _make_unary_plan(n)


def _make_unary_plan(n: int) -> UnaryPlan:
    # E_k(t) = t sum_j binom(k-1,j-1) (j! phi_j) E_(k-j)(t).
    # Each row stores (j, exact weight); phi_j is Taylor normalized.
    return ((),) + tuple(tuple((j, comb(k - 1, j - 1) * factorial(j))
                              for j in range(1, k + 1)) for k in range(1, n + 1))


def operator_unary_plan(order: int, *, budget: JetBudget | None = None) -> UnaryPlan:
    """Plan a Bell transform without enumerating integer or set partitions."""
    n = _order(order)
    work = n * (n + 1) * (n + 2) // 6 + (n + 1) * (n + 2) // 2
    slots = (n + 1) * (n + 2) // 2 + n + 1
    (budget or JetBudget()).check_layout(slots, work)
    return _cached_unary_plan(n) if work <= 16_384 else _make_unary_plan(n)


def operator_unary(coefficients: Sequence[Any], normalized_tower: Sequence[Any], *,
                   budget: JetBudget | None = None, plan: UnaryPlan | None = None) -> list[Any]:
    """Lift an arbitrary scalar unary region using phi**(j)/j! coefficients.

    Returns the differential polynomial for H(phi(u)), given the polynomial
    for arbitrary outer functions of u. The primal phi(u) is tower[0].
    Forward scratch is quadratic and schoolbook Bell work cubic in order.
    """
    n = _size(coefficients)
    if len(normalized_tower) < n + 1:
        raise ValueError(f"unary region needs a normalized derivative tower through order {n}")
    expected = operator_unary_plan(n, budget=budget) if plan is None else plan
    if plan is not None:
        work = n * (n + 1) * (n + 2) // 6 + (n + 1) * (n + 2) // 2
        (budget or JetBudget()).check_layout((n + 1) * (n + 2) // 2 + n + 1, work)
    if len(expected) != n + 1:
        raise ValueError("unary schedule does not match the operator order")
    zero = coefficients[0] * 0
    bell: list[list[Any]] = [[zero + 1]]
    result = [coefficients[0]] + [zero] * n
    for k in range(1, n + 1):
        current = [zero] * (k + 1)
        for j, weight in expected[k]:
            scale = _scale(normalized_tower[j], weight)
            for degree, coefficient in enumerate(bell[k - j]):
                current[degree + 1] = current[degree + 1] + scale * coefficient
        bell.append(current)
        for degree in range(1, k + 1):
            result[degree] = result[degree] + coefficients[k] * current[degree]
    return result


def operator_readout(coefficients: Sequence[Any], normalized_tower: Sequence[Any], *,
                     budget: JetBudget | None = None) -> Any:
    """Contract B with an outer Taylor tower, returning the RAW requested derivative."""
    n = _size(coefficients)
    if len(normalized_tower) < n + 1:
        raise ValueError(f"operator readout needs a normalized tower through order {n}")
    (budget or JetBudget()).check_layout(n + 1, n + 1)
    total = coefficients[0] * normalized_tower[0]
    for k in range(1, n + 1):
        total = total + coefficients[k] * _scale(normalized_tower[k], factorial(k))
    return total


__all__ = ["coordinate_polynomial", "operator_product", "operator_product_plan", "operator_readout",
           "operator_sum", "operator_unary", "operator_unary_plan"]

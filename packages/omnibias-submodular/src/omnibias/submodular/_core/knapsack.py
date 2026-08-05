# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Knapsack-constrained monotone submodular maximization (a non-matroid constraint).

A **knapsack** constraint ``sum_{i in S} c_i <= B`` (nonnegative costs, a budget) is *not*
a matroid -- feasibility depends on the total cost, not a rank -- so the matroid pipeline
does not apply. Two algorithms ship:

* :func:`cost_benefit_greedy` -- the fast ratio heuristic (repeatedly add the affordable
  element of largest marginal-gain-per-cost). No stand-alone guarantee (a single heavy,
  high-value element defeats it), but the workhorse inner loop.
* :func:`knapsack_maximize` -- Sviridenko's partial-enumeration greedy: take the best over
  all feasible sets of size ``< 3`` and every size-``3`` seed greedily filled by cost-benefit
  ratio. This carries the **(1 - 1/e)** guarantee for monotone submodular ``f``.

:func:`brute_force_max_knapsack` is the exact ``O(2^n)`` oracle (small ``n`` self-check) and
:func:`certify_knapsack_gap` sandwiches ``OPT`` with the **fractional-knapsack** marginal
upper bound ``U(S) = f(S) + max{ sum_i m_i z_i : sum_i c_i z_i <= B, z in [0,1] }`` where
``m_i = [f(S + i) - f(S)]_+`` -- sound because the optimal residual ``O \ S`` is a feasible
integral (hence fractional) knapsack solution. Everything here is numpy-only (no
differentiable twin this tier).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular.functions import SubmodularFunction, indicator
from omnibias.submodular.problem import ONE_MINUS_INV_E, SubmodularCertificate

FloatArray = NDArray[np.float64]

_TOL = 1e-9
_MAX_EXACT_N = 20


@dataclass(frozen=True)
class KnapsackConstraint:
    r"""A budget constraint ``sum_{i in S} costs_i <= budget`` (nonnegative costs).

    Unlike a matroid this is *not* rank-based; feasibility is a total-cost test. Pair it
    with a monotone submodular function and maximize via :func:`knapsack_maximize`.
    """

    costs: FloatArray
    budget: float

    def __post_init__(self) -> None:
        c = np.asarray(self.costs, dtype=float).reshape(-1)
        if c.shape[0] < 1:
            raise ValueError("costs must have at least one element")
        if np.any(c < 0.0):
            raise ValueError("costs must be nonnegative")
        if self.budget < 0.0:
            raise ValueError("budget must be nonnegative")
        object.__setattr__(self, "costs", c)
        object.__setattr__(self, "budget", float(self.budget))

    @property
    def n(self) -> int:
        """Ground-set size."""
        return int(self.costs.shape[0])

    def total_cost(self, x: object) -> float:
        """The total cost ``sum_i costs_i x_i`` of a ``0/1`` selection."""
        xv = np.asarray(x, dtype=float).reshape(-1)
        return float(xv @ self.costs)

    def is_feasible(self, x: object, *, tol: float = _TOL) -> bool:
        """Whether the ``0/1`` selection ``x`` respects the budget."""
        return self.total_cost(x) <= self.budget + tol


def _ratio_fill(
    function: SubmodularFunction, constraint: KnapsackConstraint, x: FloatArray
) -> FloatArray:
    r"""Greedily add affordable elements by largest marginal-gain-per-cost ratio."""
    x = x.copy()
    costs = constraint.costs
    while True:
        gains = function.marginal_gains(x)
        remaining = constraint.budget - constraint.total_cost(x)
        best_i, best_ratio = -1, 0.0
        for i in range(function.n):
            if x[i] == 1.0 or gains[i] <= 1e-12 or costs[i] > remaining + _TOL:
                continue
            ratio = float(gains[i] / costs[i]) if costs[i] > _TOL else float("inf")
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i
        if best_i < 0:
            break
        x[best_i] = 1.0
    return x


def cost_benefit_greedy(
    function: SubmodularFunction, constraint: KnapsackConstraint
) -> tuple[tuple[int, ...], float]:
    r"""The cost-benefit ratio heuristic from the empty set; ``(selection, f(selection))``.

    Fast and feasible, but with *no* stand-alone approximation guarantee (a single heavy,
    high-value item defeats pure ratio greedy). :func:`knapsack_maximize` wraps it in
    Sviridenko's partial enumeration to recover the ``(1 - 1/e)`` bound.
    """
    x = _ratio_fill(function, constraint, np.zeros(function.n, dtype=float))
    return tuple(int(v) for v in x), float(function.value(x))


def knapsack_maximize(
    function: SubmodularFunction,
    constraint: KnapsackConstraint,
    *,
    enumerate_size: int = 3,
) -> tuple[tuple[int, ...], float]:
    r"""Sviridenko's partial-enumeration greedy -> ``(1 - 1/e)`` for monotone submodular ``f``.

    Returns the best over (a) every feasible set of size ``< enumerate_size`` and (b) every
    size-``enumerate_size`` feasible seed greedily filled by cost-benefit ratio. With
    ``enumerate_size = 3`` this carries the ``(1 - 1/e)`` guarantee; it enumerates
    ``O(n^{enumerate_size})`` seeds, so keep ``enumerate_size`` small.
    """
    if enumerate_size < 1:
        raise ValueError("enumerate_size must be >= 1")
    n = function.n
    best_sel = tuple(0 for _ in range(n))
    best_val = float(function.value(np.zeros(n, dtype=float)))
    for r in range(enumerate_size):  # all feasible subsets of size < enumerate_size
        for combo in combinations(range(n), r):
            x = indicator(combo, n)
            if constraint.is_feasible(x):
                val = float(function.value(x))
                if val > best_val:
                    best_val, best_sel = val, tuple(int(v) for v in x)
    for combo in combinations(range(n), enumerate_size):  # size-k seed + ratio fill
        seed = indicator(combo, n)
        if not constraint.is_feasible(seed):
            continue
        x = _ratio_fill(function, constraint, seed)
        val = float(function.value(x))
        if val > best_val:
            best_val, best_sel = val, tuple(int(v) for v in x)
    return best_sel, best_val


def brute_force_max_knapsack(
    function: SubmodularFunction, constraint: KnapsackConstraint, *, max_n: int = _MAX_EXACT_N
) -> tuple[tuple[int, ...], float]:
    r"""Exact knapsack-constrained maximum by enumerating all ``2^n`` subsets.

    Exponential (``O(2^n)``); the small-``n`` oracle that self-checks the certificate
    sandwich. Raises :class:`ValueError` for ``n > max_n``.
    """
    n = function.n
    if n > max_n:
        raise ValueError(
            f"brute_force_max_knapsack is exponential (O(2^n)); n={n} exceeds the {max_n} cap. "
            "Use knapsack_maximize + certify_knapsack_gap for a certified heuristic instead."
        )
    idx = np.arange(1 << n, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    feasible = bits @ constraint.costs <= constraint.budget + _TOL
    vals = np.asarray(function.value(bits), dtype=float)
    vals = np.where(feasible, vals, -np.inf)
    best = int(np.argmax(vals))
    return tuple(int(v) for v in bits[best]), float(vals[best])


def _fractional_knapsack(values: FloatArray, costs: FloatArray, budget: float) -> float:
    r"""The fractional-knapsack optimum ``max{ <v, z> : <c, z> <= budget, z in [0, 1] }``.

    Greedy by value-per-cost ratio (free positive-value items taken fully); the last item
    fills the residual budget fractionally. An upper bound on the ``0/1`` knapsack optimum.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(costs, dtype=float)
    ratio = np.where(c > _TOL, v / np.where(c > _TOL, c, 1.0), np.inf)
    ratio = np.where(v > 0.0, ratio, -np.inf)  # never spend budget on nonpositive value
    order = np.argsort(-ratio, kind="stable")
    total, remaining = 0.0, float(budget)
    for i in order:
        vi, ci = float(v[i]), float(c[i])
        if vi <= 0.0:
            break
        if ci <= _TOL:  # free item: take it fully
            total += vi
            continue
        if ci <= remaining:
            total += vi
            remaining -= ci
        else:
            total += vi * (remaining / ci)
            break
    return total


def certify_knapsack_gap(
    function: SubmodularFunction, constraint: KnapsackConstraint, selection: object
) -> SubmodularCertificate:
    r"""Sandwich ``OPT`` for a budget-feasible ``selection`` with the fractional-knapsack bound.

    ``value = f(S)`` (a lower bound, since ``S`` is feasible) and
    ``U(S) = f(S) + fractional_knapsack([f(S+i)-f(S)]_+, costs, budget) >= OPT`` (sound: the
    optimal residual ``O \ S`` is a feasible integral, hence fractional, knapsack solution).
    Records the a-priori ``approx_ratio = 1 - 1/e`` (Sviridenko).

    **Monotone ``f`` only**, for the same reason as
    :func:`~omnibias.submodular.marginal_upper_bound`: bounding ``f(O)`` through the
    marginals at ``S`` passes through ``f(O u S)``, and only monotonicity closes that
    step. Sviridenko's ``1 - 1/e`` is likewise a monotone-only theorem.
    """
    if not function.is_monotone:
        raise ValueError(
            "certify_knapsack_gap requires a monotone f: the residual-marginal bound "
            "passes through f(O u S), which need not dominate f(O) otherwise, and the "
            "1 - 1/e guarantee does not apply."
        )
    xv = np.asarray(selection, dtype=float).reshape(-1)
    if xv.shape[0] != function.n:
        raise ValueError(f"selection must have length {function.n}, got {xv.shape[0]}")
    if not np.all((xv == 0.0) | (xv == 1.0)):
        raise ValueError("selection must be a 0/1 indicator")
    if not constraint.is_feasible(xv):
        raise ValueError("selection must be budget-feasible")
    value = float(function.value(xv))
    marginals = np.maximum(function.marginal_gains(xv), 0.0)
    upper = value + _fractional_knapsack(marginals, constraint.costs, constraint.budget)
    return SubmodularCertificate(
        value=value,
        upper_bound=upper,
        fractional_value=None,
        approx_ratio=ONE_MINUS_INV_E,
        method="knapsack-fractional",
    )


__all__ = [
    "KnapsackConstraint",
    "brute_force_max_knapsack",
    "certify_knapsack_gap",
    "cost_benefit_greedy",
    "knapsack_maximize",
]

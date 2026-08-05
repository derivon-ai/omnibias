# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""ApproxMC-style ``(epsilon, delta)`` approximate model counting -- STATISTICAL, not sound.

:func:`approx_model_count` estimates ``#models`` with a **probabilistic** guarantee: with
probability at least ``1 - delta`` the returned interval brackets the true count within a
multiplicative ``(1 + epsilon)`` factor (Chakraborty-Meel-Vardi ApproxMC). It shrinks the
solution space with random ``GF(2)`` XOR "hash" constraints -- each roughly halving the count
-- until a random cell holds at most ``pivot`` models, then rescales the small exact cell
count by the number of cells; the median over several trials gives the ``(epsilon, delta)``
estimate.

Honest scope -- this is **NOT** worst-case sound. It returns an :class:`ApproxCount`, a type
that can never be mistaken for a :class:`~omnibias.logic.model_count.certificate.CountCertificate`:

* the guarantee is probabilistic (it can fail with probability ``delta``);
* the per-cell oracle here is a **capped exact enumeration** (the ``O(2^n)`` sound oracle,
  short-circuited at ``pivot`` survivors), so this reference build targets small instances;
* the ``pivot`` / ``trials`` defaults are modest for a CPU-tiny build -- the theoretical
  ApproxMC settings (a larger ``pivot`` and ``O(log(1/delta))`` trials) tighten the constants
  but the *estimator* here is the faithful hashing scheme.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import TYPE_CHECKING

from omnibias.logic.approx.result import ApproxCount
from omnibias.logic.model_count.problem import _formula_satisfied

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.discrete.maxsat.problem import Clause
    from omnibias.logic.model_count.problem import ModelCountProblem

#: Enumeration cap for the reference per-cell exact oracle (``O(2^n)``).
_MAX_APPROX_N = 20

Hash = tuple[tuple[int, ...], int]  # (variable subset, target parity bit)


def _cell_count(
    clauses: tuple[Clause, ...], hashes: list[Hash], n: int, cap: int
) -> int:
    """Exact number of models in the hashed cell, short-circuited once it exceeds ``cap``."""
    count = 0
    for bits in itertools.product((0, 1), repeat=n):
        in_cell = True
        for variables, target in hashes:
            parity = 0
            for v in variables:
                parity ^= bits[v - 1]
            if parity != target:
                in_cell = False
                break
        if in_cell and _formula_satisfied(bits, clauses):
            count += 1
            if count > cap:
                return count  # only "> cap" matters to the caller
    return count


def _one_trial(clauses: tuple[Clause, ...], n: int, pivot: int, rng: random.Random) -> float:
    """One ApproxMC trial: add random XOR hashes until a cell fits ``pivot``; rescale."""
    cell = _cell_count(clauses, [], n, pivot)
    if cell <= pivot:
        return float(cell)  # small instance -> the cell is the whole space (2^0 cells)
    hashes: list[Hash] = []
    for i in range(1, n + 1):
        variables = tuple(v for v in range(1, n + 1) if rng.random() < 0.5)
        hashes.append((variables, rng.randint(0, 1)))
        cell = _cell_count(clauses, hashes, n, pivot)
        if cell <= pivot:
            return float(cell) * float(1 << i)
    return float(cell) * float(1 << n)


def approx_model_count(
    problem: ModelCountProblem,
    *,
    epsilon: float = 0.8,
    delta: float = 0.2,
    seed: int = 0,
    trials: int = 7,
    pivot: int | None = None,
    max_n: int = _MAX_APPROX_N,
) -> ApproxCount:
    r"""Estimate ``#models`` with an ``(epsilon, delta)`` XOR-hashing guarantee (NOT sound).

    Returns an :class:`ApproxCount` whose interval ``[estimate/(1+epsilon),
    estimate*(1+epsilon)]`` brackets the true count with probability ``>= 1 - delta`` (a
    *statistical* guarantee). Unweighted only. Deterministic given ``seed``.
    """
    if problem.is_weighted:
        raise ValueError("approx_model_count (XOR hashing) supports unweighted #SAT only")
    if not 0.0 < epsilon:
        raise ValueError("epsilon must be > 0")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    n = problem.n
    if n > max_n:
        raise ValueError(f"approx_model_count reference oracle is O(2^n); n={n} exceeds {max_n}")

    if pivot is None:
        pivot = int(math.ceil(9.84 * (1.0 + 1.0 / epsilon) * (1.0 + epsilon) ** 2))
    pivot = max(1, pivot)

    clauses = problem.cnf.clauses
    rng = random.Random(seed)
    estimates = sorted(_one_trial(clauses, n, pivot, rng) for _ in range(trials))
    median = estimates[len(estimates) // 2]

    return ApproxCount(
        estimate=median,
        interval=(median / (1.0 + epsilon), median * (1.0 + epsilon)),
        method="xor_hashing",
        epsilon=epsilon,
        delta=delta,
        confidence=1.0 - delta,
    )


__all__ = ["approx_model_count"]

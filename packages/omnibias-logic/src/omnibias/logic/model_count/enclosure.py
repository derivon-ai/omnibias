# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified lower+upper enclosure of the (weighted) #SAT model count.

Let ``A_c`` be the set of assignments that **falsify** clause ``c`` (every literal forced to
its falsifying value). The (weighted) number of satisfying assignments is

.. math::
    \#\mathrm{models} = Z_0 - \Bigl|\bigcup_c A_c\Bigr|,

where ``Z0`` is the total measure of the assignment space (``2^n`` unweighted, or
``prod_i (w_i(0) + w_i(1))`` weighted). Truncated inclusion-exclusion (the Bonferroni
inequalities) sandwiches the union measure -- and hence the count -- rigorously:

* an **odd** partial sum ``S_1 - S_2 + ... + S_t`` (``t`` odd) is an *upper* bound on the
  union measure, hence a *lower* bound on the count;
* an **even** partial sum is a *lower* bound on the union, hence an *upper* bound on the
  count;
* once the order reaches the number of clauses the sandwich collapses to the exact count.

Each term ``S_k = sum_{|S|=k} |intersection_{c in S} A_c|`` is closed form: the intersection
is empty (measure ``0``) when two clauses force a variable to conflicting values, else the
product of the forced literals' weights times the free variables' weight sums. Everything is
computed in **exact arithmetic** (:class:`~fractions.Fraction`); the reported float
``lower`` / ``upper`` are then rounded **outward** (down / up) so the sandwich stays sound in
floating point -- the ``omnibias.core.verified`` outward-rounding convention.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
from omnibias.logic.model_count.certificate import CountCertificate
from omnibias.logic.model_count.problem import ModelCountProblem


def _clause_forced(literals: tuple[int, ...]) -> dict[int, int]:
    """Map ``{variable: falsifying value}`` for a clause (all literals false)."""
    forced: dict[int, int] = {}
    for literal in literals:
        forced[abs(literal) - 1] = 0 if literal > 0 else 1
    return forced


def _subset_measure(
    subset: tuple[int, ...],
    clause_forced: list[dict[int, int]],
    fracs: list[tuple[Fraction, Fraction]],
    n: int,
) -> Fraction:
    """Exact measure of the assignments that falsify **every** clause in ``subset``."""
    forced: dict[int, int] = {}
    for ci in subset:
        for var, val in clause_forced[ci].items():
            if forced.get(var, val) != val:
                return Fraction(0)  # two clauses force this variable to conflicting values
            forced[var] = val
    measure = Fraction(1)
    for var in range(n):
        if var in forced:
            measure *= fracs[var][forced[var]]
        else:
            measure *= fracs[var][0] + fracs[var][1]
    return measure


def _round_down(q: Fraction) -> float:
    """Largest float ``<= q`` (outward rounding for a lower bound)."""
    f = float(q)
    if Fraction(f) > q:
        f = math.nextafter(f, -math.inf)
    return f


def _round_up(q: Fraction) -> float:
    """Smallest float ``>= q`` (outward rounding for an upper bound)."""
    f = float(q)
    if Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    return f


def _inclusion_exclusion_terms(
    problem: ModelCountProblem, order: int
) -> tuple[Fraction, list[Fraction]]:
    r"""Exact ``(Z0, [S_1, ..., S_t])`` for the truncated inclusion-exclusion, ``t = min(order, m)``.

    ``Z0`` is the total assignment measure and ``S_k = sum_{|T|=k} |intersection_{c in T} A_c|``
    is the exact ``k``-th Bonferroni term (all :class:`~fractions.Fraction`). This is the single
    source of truth for the enclosure and for the sealed exact-count identity (the full
    ``order >= m`` sum is the inclusion-exclusion theorem, ``Z0 - S_1 + S_2 - ... = #models``).
    """
    n = problem.n
    clauses = problem.cnf.clauses
    m = len(clauses)
    fracs = problem.weight_fractions()

    z0 = Fraction(1)
    for w0, w1 in fracs:
        z0 *= w0 + w1

    clause_forced = [_clause_forced(clause.literals) for clause in clauses]
    s_terms: list[Fraction] = []
    for k in range(1, min(order, m) + 1):
        sk = Fraction(0)
        for subset in itertools.combinations(range(m), k):
            sk += _subset_measure(subset, clause_forced, fracs, n)
        s_terms.append(sk)
    return z0, s_terms


def _witness_lower(problem: ModelCountProblem, witnesses: object) -> Fraction:
    """Exact (weighted) lower bound from a set of distinct **verified** model witnesses."""
    fracs = problem.weight_fractions()
    n = problem.n
    seen: set[tuple[int, ...]] = set()
    total = Fraction(0)
    rows = np.asarray(witnesses, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    for row in rows:
        bits = tuple(int(round(v)) for v in row)
        if len(bits) != n or any(b not in (0, 1) for b in bits) or bits in seen:
            continue
        if not problem.is_model(np.asarray(bits, dtype=float)):
            continue  # only verified models are sound witnesses
        seen.add(bits)
        prod = Fraction(1)
        for i in range(n):
            prod *= fracs[i][bits[i]]
        total += prod
    return total


def count_enclosure(
    problem: ModelCountProblem,
    *,
    order: int = 2,
    witnesses: object | None = None,
) -> CountCertificate:
    r"""Certify a ``lower <= #models <= upper`` enclosure of the (weighted) model count.

    Parameters
    ----------
    problem:
        The :class:`~omnibias.logic.model_count.problem.ModelCountProblem` to count.
    order:
        The inclusion-exclusion (Bonferroni) truncation order ``>= 0``. A higher order only
        tightens the enclosure; ``order >= #clauses`` counts exactly (but is exponential in
        the number of clauses).
    witnesses:
        Optional binary assignments (a single ``(n,)`` point or a batch ``(m, n)``, e.g. from
        the annealed :func:`~omnibias.logic.torch.sat_relaxation` + decoder) whose distinct
        **verified** models strengthen the lower bound (a sound witness count is folded in via
        ``max``).

    Returns
    -------
    :class:`~omnibias.logic.model_count.certificate.CountCertificate` with the rigorous
    outward-rounded ``lower`` / ``upper`` bounds.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    # Z0 and the exact Bonferroni terms S_1..S_min(order,m) (shared with the sealed identity).
    z0, s_terms = _inclusion_exclusion_terms(problem, order)

    # partial[t] = sum_{k=1}^{t} (-1)^{k+1} S_k, the t-term inclusion-exclusion estimate of
    # the union measure. partial[t] = partial[min(t, m)] for t >= m (extra terms are zero).
    partial = [Fraction(0)]
    for idx, sk in enumerate(s_terms):
        partial.append(partial[-1] + (-1 if idx % 2 else 1) * sk)

    def partial_at(t: int) -> Fraction:
        return partial[min(t, len(partial) - 1)]

    # Largest odd / even truncation <= order: odd -> upper bound on the union (lower on the
    # count); even (>= 0 always available) -> lower bound on the union (upper on the count).
    t_odd = order if order % 2 == 1 else order - 1
    t_even = order if order % 2 == 0 else order - 1
    upper_union = partial_at(t_odd) if t_odd >= 1 else z0  # no odd term -> trivial union <= Z0
    lower_union = partial_at(t_even)  # t_even >= 0, partial[0] = 0

    lower_frac = z0 - upper_union
    upper_frac = z0 - lower_union

    # Clamp to the valid range [0, Z0] (still sound: the true count lies in it too).
    lower_frac = max(Fraction(0), min(lower_frac, z0))
    upper_frac = max(Fraction(0), min(upper_frac, z0))

    method = "inclusion_exclusion" if order >= 1 else "trivial"
    if witnesses is not None:
        wl = _witness_lower(problem, witnesses)
        if wl > lower_frac:
            lower_frac = wl
            method = f"{method}+witness"

    tight = lower_frac == upper_frac
    return CountCertificate(
        lower=_round_down(lower_frac),
        upper=_round_up(upper_frac),
        method=method,
        order=int(order),
        weighted=problem.is_weighted,
        tight=bool(tight),
        total=float(z0),
    )


__all__ = ["count_enclosure"]

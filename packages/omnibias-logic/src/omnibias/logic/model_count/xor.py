# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Exact model counting for the **affine (XOR / parity) fragment** -- poly-time and sound.

By the Creignou-Hermann counting dichotomy, ``#SAT`` is in FP (polynomial time) exactly when
every constraint is **affine** (an XOR/parity constraint); otherwise it is ``#P``-complete.
For a conjunction of parity constraints ``XOR_{i in S} x_i = b`` over ``n`` variables, the
satisfying assignments form an affine subspace of ``GF(2)^n``, so the exact model count is

.. math::
    \#\mathrm{models} = \begin{cases} 2^{\,n - \mathrm{rank}} & \text{if consistent} \\ 0 &
    \text{otherwise,}\end{cases}

computed by Gaussian elimination over ``GF(2)`` (``O(n \cdot m)`` word operations). This is
the same elimination used by :func:`omnibias.boolean.gf2_solve` in
:mod:`omnibias.boolean._core.systems`, adapted here to accept XOR clauses **directly** so we
never build the exponential ``2^n`` truth-table its public interface expects.

Honesty / scope:

* This is the **unweighted** count. Weighted counting over an affine subspace couples the
  variables and is **not** poly-time in general, so :func:`xor_model_count` refuses weights;
  weighted exact counting is served by the bounded-treewidth DP and the DPLL counter.
* :func:`detect_xor_system` recognises the standard CNF-of-XOR encoding *conservatively*: it
  returns the parity system only when the whole CNF is exactly a conjunction of parity
  constraints (each an XOR group of ``2^{k-1}`` full-width clauses of one sign parity), and
  ``None`` otherwise -- so the affine fast path is only taken when it is provably exact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from omnibias.logic.model_count.problem import ModelCountProblem


@dataclass(frozen=True)
class XORClause:
    r"""A parity constraint ``XOR_{v in variables} x_v = parity`` (1-based positive vars)."""

    variables: tuple[int, ...]
    parity: int

    def __post_init__(self) -> None:
        variables = tuple(int(v) for v in self.variables)
        if any(v < 1 for v in variables):
            raise ValueError("XORClause variables must be positive 1-based integers")
        if self.parity not in (0, 1):
            raise ValueError(f"parity must be 0 or 1, got {self.parity}")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "parity", int(self.parity))


def _as_xor(clause: XORClause | tuple[Sequence[int], int]) -> tuple[tuple[int, ...], int]:
    """Normalise an ``XORClause`` or a raw ``(variables, parity)`` tuple."""
    if isinstance(clause, XORClause):
        return clause.variables, clause.parity
    variables, parity = clause
    return tuple(int(v) for v in variables), int(parity)


def xor_model_count(
    clauses: Sequence[XORClause | tuple[Sequence[int], int]], n_vars: int
) -> int:
    r"""Exact (unweighted) number of assignments satisfying every XOR/parity constraint.

    ``2^(n_vars - rank)`` when the affine system is consistent, else ``0``. Poly-time (Gaussian
    elimination over ``GF(2)``); sound and exact. A signed variable in a clause is allowed as a
    convenience -- it flips that constraint's parity.
    """
    if n_vars < 1:
        raise ValueError("n_vars must be >= 1")
    pivots: dict[int, tuple[int, int]] = {}  # pivot bit position -> (row mask, rhs)
    rank = 0
    for clause in clauses:
        variables, parity = _as_xor(clause)
        mask = 0
        rhs = parity & 1
        for v in variables:
            idx = abs(v) - 1
            if not 0 <= idx < n_vars:
                raise ValueError(f"variable {v} outside 1..{n_vars}")
            mask ^= 1 << idx  # XOR toggles -> a repeated variable correctly cancels
            if v < 0:
                rhs ^= 1
        m, r = mask, rhs
        while m:
            low = m & (-m)
            bitpos = low.bit_length() - 1
            if bitpos in pivots:
                pmask, prhs = pivots[bitpos]
                m ^= pmask
                r ^= prhs
            else:
                pivots[bitpos] = (m, r)
                rank += 1
                break
        else:  # m reduced to 0
            if r == 1:
                return 0  # 0 = 1 is inconsistent -> no solutions
    return 1 << (n_vars - rank)


def detect_xor_system(problem: ModelCountProblem) -> list[XORClause] | None:
    r"""Recognise a CNF as an exact conjunction of XOR constraints, else ``None``.

    Conservative and sound: returns the parity system only when *every* clause belongs to a
    variable-set group of exactly ``2^{k-1}`` distinct full-width clauses that all share the
    same negative-literal parity (the CNF encoding of one ``XOR_{v} x_v = b`` constraint).
    Unit clauses (``k = 1``) are the arity-1 case. Returns ``None`` on any mismatch, so the
    caller safely falls back to a CNF-native method.
    """
    groups: dict[frozenset[int], list[tuple[int, ...]]] = defaultdict(list)
    for clause in problem.cnf.clauses:
        variables = frozenset(abs(literal) for literal in clause.literals)
        if len(variables) != len(clause.literals):
            return None  # a repeated variable is not the clean XOR encoding
        groups[variables].append(clause.literals)

    xors: list[XORClause] = []
    for variables, literal_sets in groups.items():
        k = len(variables)
        if len(literal_sets) != (1 << (k - 1)):
            return None  # wrong number of clauses for a parity constraint over these vars
        patterns: set[tuple[tuple[int, int], ...]] = set()
        parities: set[int] = set()
        for literals in literal_sets:
            pattern = tuple(sorted((abs(v), 0 if v > 0 else 1) for v in literals))
            if pattern in patterns:
                return None  # a duplicated clause is not the clean encoding
            patterns.add(pattern)
            parities.add(sum(1 for v in literals if v < 0) % 2)
        if len(parities) != 1:
            return None  # forbidden points must all share one parity
        forbidden_parity = parities.pop()
        xors.append(XORClause(variables=tuple(sorted(variables)), parity=1 - forbidden_parity))
    return xors


__all__ = ["XORClause", "detect_xor_system", "xor_model_count"]

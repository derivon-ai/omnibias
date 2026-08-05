# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Exact ``{0,1}`` <-> ``{-1,+1}`` conversion and the SOS polynomial encoder.

The spin change of variables is ``s_i = 2 x_i - 1`` (``x_i = (s_i + 1) / 2``), an exact
affine bijection between ``x in {0, 1}^n`` and ``s in {-1, +1}^n``. Substituting it into
one quadratic form yields the other with matched energy on every vertex -- the
guardrail is the round-trip energy test.

:func:`to_polynomial` builds the quadratic energy as an
:class:`omnibias.sos.Polynomial` for the certified Lasserre lower bound.
:func:`boolean_constraints` (the Boolean ideal ``x_i^2 = x_i`` as two inequalities) is
generic and re-exported from the ``omnibias-discrete`` substrate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from omnibias.discrete import boolean_constraints
from omnibias.qubo.problem import IsingProblem, QUBOProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial


def qubo_to_ising(problem: QUBOProblem) -> IsingProblem:
    r"""Exact conversion of a QUBO to the Ising model via ``s = 2 x - 1``.

    With ``x = (s + 1) / 2`` one has ``J = Q / 4``,
    ``h = (Q 1) / 2 + c / 2`` and a matched constant; the diagonal of ``J`` is folded
    into the constant by :class:`~omnibias.qubo.problem.IsingProblem` (``s_i^2 = 1``).
    """
    q = np.asarray(problem.Q, dtype=float)
    c = np.asarray(problem.c, dtype=float)
    ones = np.ones(problem.n)
    j = q / 4.0
    h = 0.5 * (q @ ones) + 0.5 * c
    const = problem.const + 0.25 * float(ones @ q @ ones) + 0.5 * float(c @ ones)
    return IsingProblem(J=j, h=h, const=const, name=problem.name)


def ising_to_qubo(problem: IsingProblem) -> QUBOProblem:
    r"""Exact conversion of an Ising model to a QUBO via ``x = (s + 1) / 2``.

    With ``s = 2 x - 1`` one has ``Q = 4 J``, ``c = 2 h - 4 (J 1)`` and a matched
    constant (``J`` already carries a zero diagonal).
    """
    j = np.asarray(problem.J, dtype=float)
    h = np.asarray(problem.h, dtype=float)
    ones = np.ones(problem.n)
    q = 4.0 * j
    c = 2.0 * h - 4.0 * (j @ ones)
    const = problem.const + float(ones @ j @ ones) - float(h @ ones)
    return QUBOProblem(Q=q, c=c, const=const, name=problem.name)


def to_polynomial(problem: QUBOProblem) -> Polynomial:
    r"""The QUBO energy ``x^T Q x + c^T x + const`` as an :class:`omnibias.sos.Polynomial`.

    Iterating every ``(i, j)`` and adding ``Q[i, j]`` to the monomial ``x_i x_j``
    reproduces the energy exactly (the symmetric off-diagonal pair sums to the true
    ``x_i x_j`` coefficient; the diagonal gives the ``x_i^2`` term).
    """
    from omnibias.sos import Polynomial

    n = problem.n
    q = np.asarray(problem.Q, dtype=float)
    c = np.asarray(problem.c, dtype=float)
    coeffs: dict[tuple[int, ...], float] = {}

    def _add(exp: tuple[int, ...], value: float) -> None:
        if value != 0.0:
            coeffs[exp] = coeffs.get(exp, 0.0) + value

    if problem.const != 0.0:
        _add((0,) * n, problem.const)
    for i in range(n):
        _add(tuple(1 if k == i else 0 for k in range(n)), float(c[i]))
        for j in range(n):
            qij = float(q[i, j])
            if qij == 0.0:
                continue
            exp = [0] * n
            exp[i] += 1
            exp[j] += 1
            _add(tuple(exp), qij)
    return Polynomial(n, coeffs)


__all__ = [
    "boolean_constraints",
    "ising_to_qubo",
    "qubo_to_ising",
    "to_polynomial",
]

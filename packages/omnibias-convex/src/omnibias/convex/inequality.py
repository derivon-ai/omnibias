# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Linear inequality adapter (theory 09-30).

Propose via arrangement soft membership (temperature collapse). Check
slacks over ``Q``. Emptiness only via vertex enum under
``VERTEX_ENUM_MAX_*``. ``InfeasibleProblemError`` is never ``DISPROVED``.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import ClassVar

import numpy as np
from omnibias.convex.arrangement._core import (
    VERTEX_ENUM_MAX_D,
    VERTEX_ENUM_MAX_N,
    LearnedPolytope,
    enumerate_vertices,
    honesty_payload,
    soft_membership,
)
from omnibias.core.proof.discovery import ExactCheck
from omnibias.core.proof.inequality import (
    InequalitySort,
    InequalitySystem,
    Proposal,
    RationalWitness,
    register_inequality_backend,
)

_FRAC_LIM = 10_000


def _as_frac(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def _matrix(raw: object) -> tuple[tuple[Fraction, ...], ...]:
    rows = []
    for row in raw if isinstance(raw, Sequence) else ():
        rows.append(tuple(_as_frac(item) for item in row))
    return tuple(rows)


def _vector(raw: object) -> tuple[Fraction, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(_as_frac(item) for item in raw)


def _slacks_q(
    a_mat: Sequence[Sequence[Fraction]],
    b_vec: Sequence[Fraction],
    x: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    slacks: list[Fraction] = []
    for row, rhs in zip(a_mat, b_vec, strict=True):
        slacks.append(rhs - sum(ai * xj for ai, xj in zip(row, x, strict=True)))
    return tuple(slacks)


class LinearInequalityBackend:
    sort: ClassVar[InequalitySort] = "linear"

    def propose(self, system: InequalitySystem) -> Proposal:
        a_mat = _matrix(system.data.get("A", ()))
        b_vec = _vector(system.data.get("b", ()))
        dim = len(a_mat[0]) if a_mat else 0
        candidate = [0.0] * dim
        poly = LearnedPolytope(
            np.asarray([[float(v) for v in row] for row in a_mat], dtype=np.float64),
            np.asarray([float(v) for v in b_vec], dtype=np.float64),
        )
        weight = float(soft_membership(poly, candidate, beta=8.0))
        return Proposal(
            method="soft_membership",
            values=tuple(str(value) for value in candidate),
            payload={"membership": weight, "beta": 8.0},
        )

    def rationalize(
        self, system: InequalitySystem, proposal: Proposal
    ) -> RationalWitness:
        values = tuple(
            str(Fraction(item).limit_denominator(_FRAC_LIM)) for item in proposal.values
        )
        return RationalWitness(method="nearest_q", values=values, payload=dict(proposal.payload))

    def check(
        self, system: InequalitySystem, witness: RationalWitness
    ) -> ExactCheck:
        a_mat = _matrix(system.data.get("A", ()))
        b_vec = _vector(system.data.get("b", ()))
        x = tuple(_as_frac(item) for item in witness.values)
        honesty = dict(honesty_payload())
        honesty["new_lp_algorithm_claim"] = False
        honesty["complete_solver"] = False
        if len(x) != (len(a_mat[0]) if a_mat else 0):
            return ExactCheck(
                ok=False,
                payload={
                    "role": "inconclusive",
                    "method": "dimension_mismatch",
                    "honesty": honesty,
                    "detail": "dimension_mismatch",
                },
            )
        slacks = _slacks_q(a_mat, b_vec, x)
        if all(s >= 0 for s in slacks):
            return ExactCheck(
                ok=True,
                payload={
                    "role": "witness",
                    "method": "exact_slack",
                    "honesty": honesty,
                    "detail": "feasible_q",
                    "slacks": [str(s) for s in slacks],
                },
            )
        dim = len(a_mat[0]) if a_mat else 0
        n_ineq = len(a_mat)
        if dim > VERTEX_ENUM_MAX_D or n_ineq > VERTEX_ENUM_MAX_N:
            return ExactCheck(
                ok=False,
                payload={
                    "role": "inconclusive",
                    "method": "enum_refused",
                    "honesty": honesty,
                    "detail": "vertex_enum_refused",
                },
            )
        poly = LearnedPolytope(
            np.asarray([[float(v) for v in row] for row in a_mat], dtype=np.float64),
            np.asarray([float(v) for v in b_vec], dtype=np.float64),
        )
        verts = enumerate_vertices(poly)
        if verts.shape[0] == 0:
            honesty["unsat_proof"] = True
            return ExactCheck(
                ok=False,
                payload={
                    "role": "empty",
                    "method": "vertex_enum_empty",
                    "honesty": honesty,
                    "detail": "vertex_enum_empty",
                    "complete": True,
                },
            )
        return ExactCheck(
            ok=False,
            payload={
                "role": "inconclusive",
                "method": "point_infeasible",
                "honesty": honesty,
                "detail": "proposed_point_infeasible",
            },
        )


def _register() -> None:
    register_inequality_backend(LinearInequalityBackend())


_register()


__all__ = ["LinearInequalityBackend"]

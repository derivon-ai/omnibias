# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Finite-domain CSP inequality adapter (theory 09-30).

Propose via ``csp_solve``. Witnessed SAT through ``certify_csp``.
UNSAT ``DISPROVED`` only via labelled ``brute_force_sat``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from omnibias.core.proof.discovery import ExactCheck
from omnibias.core.proof.inequality import (
    InequalitySort,
    InequalitySystem,
    Proposal,
    RationalWitness,
    register_inequality_backend,
)
from omnibias.discrete.csp._core import (
    CSP,
    Relation,
    Variable,
    brute_force_sat,
    certify_csp,
    csp_solve,
    honesty_payload,
    is_satisfying_vertex,
)


def _csp_from_data(data: Mapping[str, object]) -> CSP:
    variables_raw = data.get("variables", ())
    relations_raw = data.get("relations", ())
    variables: list[Variable] = []
    for item in variables_raw if isinstance(variables_raw, Sequence) else ():
        if not isinstance(item, Mapping):
            continue
        domain = tuple(item.get("domain", ()))
        variables.append(Variable(str(item.get("name", "x")), domain))
    relations: list[Relation] = []
    for item in relations_raw if isinstance(relations_raw, Sequence) else ():
        if not isinstance(item, Mapping):
            continue
        scope = tuple(int(idx) for idx in item.get("scope", ()))
        allowed = frozenset(
            tuple(int(v) for v in tup)
            for tup in item.get("allowed", ())
        )
        relations.append(Relation(scope, allowed))
    if not variables:
        raise ValueError("csp data needs variables")
    return CSP(tuple(variables), tuple(relations), name="inequality")


class CspInequalityBackend:
    sort: ClassVar[InequalitySort] = "csp"

    def propose(self, system: InequalitySystem) -> Proposal:
        csp = _csp_from_data(system.data)
        result = csp_solve(csp, seed=0, restarts=2, steps=4)
        assignment = result.assignment
        if not is_satisfying_vertex(csp, assignment):
            for rel in csp.relations:
                for tup in rel.allowed:
                    trial = [0] * len(csp.variables)
                    for idx, value in zip(rel.scope, tup, strict=True):
                        trial[idx] = value
                    candidate = tuple(trial)
                    if is_satisfying_vertex(csp, candidate):
                        assignment = candidate
                        break
                else:
                    continue
                break
        return Proposal(
            method="csp_solve",
            values=tuple(str(v) for v in assignment),
            payload={"energy": result.energy, "satisfied": result.satisfied},
        )

    def rationalize(
        self, system: InequalitySystem, proposal: Proposal
    ) -> RationalWitness:
        return RationalWitness(
            method="onehot_decode",
            values=proposal.values,
            payload=dict(proposal.payload),
        )

    def check(
        self, system: InequalitySystem, witness: RationalWitness
    ) -> ExactCheck:
        csp = _csp_from_data(system.data)
        honesty = dict(honesty_payload())
        oracle = bool(system.data.get("oracle", False))
        if oracle:
            sat, assign = brute_force_sat(csp)
            if sat and assign is not None:
                cert = certify_csp(csp, assign, beta=8.0)
                return ExactCheck(
                    ok=True,
                    payload={
                        "role": "witness",
                        "method": "brute_force_sat",
                        "honesty": honesty,
                        "detail": "csp_sat_oracle",
                        "claimed_sat": cert.claimed_sat,
                    },
                )
            honesty["unsat_proof"] = True
            return ExactCheck(
                ok=False,
                payload={
                    "role": "empty",
                    "method": "brute_force_sat",
                    "honesty": honesty,
                    "detail": "csp_unsat_oracle",
                    "complete": True,
                },
            )
        assign = tuple(int(v) for v in witness.values) if witness.values else ()
        if assign and is_satisfying_vertex(csp, assign):
            cert = certify_csp(csp, assign, beta=8.0)
            return ExactCheck(
                ok=True,
                payload={
                    "role": "witness",
                    "method": "certify_csp",
                    "honesty": honesty,
                    "detail": "csp_sat_witness",
                    "claimed_sat": cert.claimed_sat,
                },
            )
        return ExactCheck(
            ok=False,
            payload={
                "role": "inconclusive",
                "method": "anneal_miss",
                "honesty": honesty,
                "detail": "anneal_miss",
            },
        )


def _register() -> None:
    register_inequality_backend(CspInequalityBackend())


_register()


__all__ = ["CspInequalityBackend"]

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Polynomial inequality adapter (theory 09-30).

Reuses ``Observation.poly_terms`` encoding. Universal positivity goes
through ``certify_sos``; a failed SDP is ``BLOCKED``. A rational point
with ``p < 0`` is a counterexample. Soft RMSE is never an ExactCheck.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import ClassVar

from omnibias.core.proof.discovery import ExactCheck
from omnibias.core.proof.inequality import (
    InequalitySort,
    InequalitySystem,
    Proposal,
    RationalWitness,
    register_inequality_backend,
)
from omnibias.sos.certify import certify_sos
from omnibias.sos.honesty import GLOBAL_POLYNOMIAL, SOSScope, honesty_labels
from omnibias.sos.problem import Polynomial


def _as_frac(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def _parse_terms(raw: object) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    terms: list[tuple[tuple[int, ...], Fraction]] = []
    for item in raw if isinstance(raw, Sequence) else ():
        exp_raw, coeff = item[0], item[1]
        exp = tuple(int(power) for power in exp_raw)
        terms.append((exp, _as_frac(coeff)))
    return tuple(terms)


def _eval_q(
    terms: Sequence[tuple[tuple[int, ...], Fraction]],
    point: Sequence[Fraction],
) -> Fraction:
    total = Fraction(0)
    for exp, coeff in terms:
        term = coeff
        for value, power in zip(point, exp, strict=False):
            term *= value ** int(power)
        total += term
    return total


def _to_polynomial(n_vars: int, terms: Sequence[tuple[tuple[int, ...], Fraction]]) -> Polynomial:
    coeffs: dict[tuple[int, ...], float] = {}
    for exp, coeff in terms:
        padded = exp + (0,) * (n_vars - len(exp)) if len(exp) < n_vars else exp[:n_vars]
        coeffs[padded] = float(coeff)
    return Polynomial(n_vars, coeffs)


class PolynomialInequalityBackend:
    sort: ClassVar[InequalitySort] = "polynomial"

    def propose(self, system: InequalitySystem) -> Proposal:
        n_vars = int(system.data.get("poly_n_vars", 1))
        origin = tuple("0" for _ in range(max(n_vars, 1)))
        return Proposal(method="origin", values=origin, payload={"sdp": True})

    def rationalize(
        self, system: InequalitySystem, proposal: Proposal
    ) -> RationalWitness:
        values = tuple(str(_as_frac(item)) for item in proposal.values)
        return RationalWitness(method="nearest_q", values=values)

    def check(
        self, system: InequalitySystem, witness: RationalWitness
    ) -> ExactCheck:
        n_vars = int(system.data.get("poly_n_vars", 1))
        terms = _parse_terms(system.data.get("poly_terms", ()))
        point = tuple(_as_frac(item) for item in witness.values) or (Fraction(0),) * n_vars
        if len(point) < n_vars:
            point = point + (Fraction(0),) * (n_vars - len(point))
        value = _eval_q(terms, point[:n_vars])
        honesty = dict(honesty_labels(SOSScope(kind=GLOBAL_POLYNOMIAL)))
        honesty["complete_solver"] = False
        if not system.existential and value < 0:
            return ExactCheck(
                ok=True,
                payload={
                    "role": "counterexample",
                    "method": "eval_q",
                    "honesty": honesty,
                    "detail": f"p<0 at {tuple(str(v) for v in point)}",
                    "value": str(value),
                },
            )
        if system.existential:
            constraints = system.data.get("poly_constraints", ())
            ok = True
            if isinstance(constraints, Sequence) and constraints:
                for raw in constraints:
                    c_terms = _parse_terms(raw)
                    if _eval_q(c_terms, point[:n_vars]) < 0:
                        ok = False
                        break
            return ExactCheck(
                ok=ok,
                payload={
                    "role": "witness" if ok else "inconclusive",
                    "method": "eval_q",
                    "honesty": honesty,
                    "detail": "feasible_point" if ok else "point_violates_g",
                },
            )
        poly = _to_polynomial(n_vars, terms)
        cert = certify_sos(poly)
        if cert.certified:
            return ExactCheck(
                ok=True,
                payload={
                    "role": "positivity",
                    "method": "certify_sos",
                    "honesty": honesty,
                    "detail": "sos_certified",
                },
            )
        return ExactCheck(
            ok=False,
            payload={
                "role": "inconclusive",
                "method": "certify_sos",
                "honesty": honesty,
                "detail": str(cert.detail or "sos_inconclusive"),
            },
        )


def _register() -> None:
    register_inequality_backend(PolynomialInequalityBackend())


_register()


__all__ = ["PolynomialInequalityBackend"]

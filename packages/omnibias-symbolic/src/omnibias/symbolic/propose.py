# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Proposers emit a :class:`Statement` plus hypothesis; snap is the accept gate.

``NeuralJetDiscoverer`` / ``fit_sparse_equation`` / a PINN residual stay
``empirical`` until :func:`snap_sparse_equation` accepts. This module does not
train a PINN and does not add names to ``get_proposer``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omnibias.core.proof.condition import (
    ConditionHypothesis,
    ConditionSort,
    ConditionToken,
    condition_honesty,
    emit_condition,
)
from omnibias.core.proof.discovery import DiscoveredEquation, Statement
from omnibias.core.proof.lift import as_fraction
from omnibias.symbolic.discovery import SparseEquation
from omnibias.symbolic.lift import snap_sparse_equation, sparse_from_coeffs

Number = int | float


def hypothesis_from_sparse(
    equation: SparseEquation,
    *,
    sort: ConditionSort = "jet_monomial",
    denom_bound: int = 32,
) -> ConditionHypothesis:
    """Name the active terms of a float equation as a hypothesis (not a certificate)."""

    tokens: list[ConditionToken] = []
    coeffs: list[str] = []
    for name, coef in zip(equation.term_names, equation.coefficients, strict=False):
        snapped = as_fraction(float(coef), denom_bound=denom_bound)
        coeffs.append(str(snapped))
        if snapped == 0:
            continue
        tokens.append(ConditionToken(sort, str(name)))
    return ConditionHypothesis(
        sort=sort,
        tokens=tuple(tokens),
        coefficients=tuple(coeffs),
    )


def propose_jet_condition(
    equation: SparseEquation,
    design: Sequence[Sequence[Number]],
    target: Sequence[Number],
    *,
    sort: ConditionSort = "jet_monomial",
    denom_bound: int = 32,
    parent: str = "jet identities",
    kind: str = "polynomial_identity",
) -> dict[str, Any]:
    """Snap ``equation``; emit a statement only when the residual is identically 0."""

    hypothesis = hypothesis_from_sparse(equation, sort=sort, denom_bound=denom_bound)
    snapped = snap_sparse_equation(
        equation, design, target, denom_bound=denom_bound, kind=kind
    )
    if snapped is None:
        return {
            "mode": "empirical",
            "check": None,
            "hypothesis": hypothesis,
            "statement": None,
            "equation": None,
            "honesty": condition_honesty(discovered=False),
        }
    accepted = ConditionHypothesis(
        sort=hypothesis.sort,
        tokens=hypothesis.tokens,
        constructors=hypothesis.constructors,
        coefficients=snapped.coefficients,
        grammar_id=hypothesis.grammar_id,
    )
    statement = emit_condition(
        accepted,
        parent=parent,
        parent_status="already_true",
        obligation=f"snapped {accepted.pretty()} with identically zero residual",
    )
    return {
        "mode": "exact_search",
        "check": True,
        "hypothesis": accepted,
        "statement": statement,
        "equation": snapped,
        "honesty": condition_honesty(discovered=True),
    }


def propose_from_residual(
    residual: Sequence[Number],
    *,
    design: Sequence[Sequence[Number]] | None = None,
    coeffs: Sequence[Number] | None = None,
    names: Sequence[str] | None = None,
    target: Sequence[Number] | None = None,
    denom_bound: int = 32,
    sort: ConditionSort = "pde_operator",
) -> dict[str, Any]:
    """PINN-shaped residual proposer. Empirical unless an exact design snaps."""

    _ = residual
    if design is None or coeffs is None or names is None or target is None:
        return {
            "mode": "empirical",
            "check": None,
            "hypothesis": None,
            "statement": None,
            "equation": None,
            "honesty": condition_honesty(discovered=False),
        }
    equation = sparse_from_coeffs(coeffs, names)
    return propose_jet_condition(
        equation,
        design,
        target,
        sort=sort,
        denom_bound=denom_bound,
        parent="residual identity",
        kind="pde_span",
    )


def snapped_statement(payload: dict[str, Any]) -> Statement | None:
    raw = payload.get("statement")
    return raw if isinstance(raw, Statement) else None


def snapped_equation(payload: dict[str, Any]) -> DiscoveredEquation | None:
    raw = payload.get("equation")
    return raw if isinstance(raw, DiscoveredEquation) else None


__all__ = [
    "hypothesis_from_sparse",
    "propose_from_residual",
    "propose_jet_condition",
    "snapped_equation",
    "snapped_statement",
]

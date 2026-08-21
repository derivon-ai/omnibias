# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Register exact symbolic discovery families on :class:`ProofMachine`.

===============================  ==========================================
kind                             prover
===============================  ==========================================
``recurrence_span``              :class:`~omnibias.symbolic.families.RecurrenceSpanFamily`
``activation_identity_exact``    :class:`~omnibias.symbolic.families.ActivationIdentityFamily`
``pde_operator_span``            :class:`~omnibias.symbolic.families.PdeOperatorSpanFamily`
===============================  ==========================================

``PROVED`` certifies the finite obligation in that span. It does not mean a
new identity was found beyond the named family.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from omnibias.core.proof import (
    CatalogEntry,
    Certificate,
    Conjecture,
    FunctionProver,
    KindMetaFamily,
    ProofAttempt,
    ProofMachine,
    list_condition_sorts,
    register_catalog,
    run_discovery,
)
from omnibias.core.proof.observe import Observation
from omnibias.symbolic.conditions import (
    ConservationConditionFamily,
    FractionalOrderFamily,
    JetConditionFamily,
    PdeConditionFamily,
    PiecewiseHybridFamily,
    ResidualSignFamily,
    jet_growth_family,
)
from omnibias.symbolic.families import (
    ActivationIdentityFamily,
    PdeOperatorSpanFamily,
    RecurrenceSpanFamily,
    catalan_samples,
)
from omnibias.symbolic.propose import propose_from_residual, propose_jet_condition

RECURRENCE_SPAN = "recurrence_span"
ACTIVATION_IDENTITY_EXACT = "activation_identity_exact"
PDE_OPERATOR_SPAN = "pde_operator_span"
CONDITION_JET = "condition_jet_monomial"
CONDITION_PDE = "condition_pde_operator"
CONDITION_CONSERVATION = "condition_conservation"
CONDITION_FRACTIONAL = "condition_fractional_order"
CONDITION_PIECEWISE = "condition_piecewise_hybrid"
CONDITION_KIND_META = "condition_kind_meta"
CONDITION_GRAMMAR_GROWTH = "condition_grammar_growth"
CONDITION_RESIDUAL_SIGN = "condition_residual_sign"

FAMILY_CATALOG: dict[str, dict[str, str]] = {
    RECURRENCE_SPAN: {
        "parent_status": "already_true",
        "obligation": "a unique P-recurrence in the (order, index_degree) box",
        "complete": "True",
    },
    ACTIVATION_IDENTITY_EXACT: {
        "parent_status": "already_true",
        "obligation": "a unique linear relation in {1, y, y^2, y', y''} for tanh",
        "complete": "True",
    },
    PDE_OPERATOR_SPAN: {
        "parent_status": "already_true",
        "obligation": "snapped heat-operator coefficients with identically zero residual",
        "complete": "True",
    },
}


def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _from_result(result: Any) -> ProofAttempt:
    payload = result.as_dict()
    if result.status == "PROVED":
        return ProofAttempt(status="PROVED", certificate=payload, detail=result.detail)
    if result.status == "DISPROVED":
        return ProofAttempt(status="DISPROVED", certificate=payload, detail=result.detail)
    return _blocked(result.detail)


def _samples(data: Mapping[str, Any]) -> tuple[int | Fraction, ...]:
    raw = data.get("samples")
    if raw is None:
        return catalan_samples()
    return tuple(Fraction(item) if not isinstance(item, int) else item for item in raw)


def _prove_recurrence(conjecture: Conjecture) -> ProofAttempt:
    family = RecurrenceSpanFamily(
        samples=_samples(conjecture.data),
        max_order=int(conjecture.data.get("max_order", 2)),
        max_index_degree=int(conjecture.data.get("max_index_degree", 1)),
    )
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    budget = int(conjecture.data.get("budget", 8))
    return _from_result(run_discovery(family.statement, family, proposer, budget=budget))


def _prove_activation(conjecture: Conjecture) -> ProofAttempt:
    family = ActivationIdentityFamily()
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    budget = int(conjecture.data.get("budget", 32))
    return _from_result(run_discovery(family.statement, family, proposer, budget=budget))


def _prove_pde(conjecture: Conjecture) -> ProofAttempt:
    raw = conjecture.data.get("diffusivity", "1/8")
    family = PdeOperatorSpanFamily(diffusivity=Fraction(str(raw)))
    return _from_result(run_discovery(family.statement, family, "score_guided", budget=1))


def _schema_errors(certificate: Certificate) -> list[str]:
    honesty = certificate.get("honesty", {})
    if not isinstance(honesty, Mapping):
        return ["honesty must be a mapping"]
    errors: list[str] = []
    for key in (
        "jacobian_conjecture_proof_claim",
        "jacobian_n2_claim",
        "navier_stokes_proof_claim",
    ):
        if honesty.get(key):
            errors.append(f"{key} must be False")
    return errors


def _register() -> None:
    factories = {
        RECURRENCE_SPAN: lambda **kwargs: _from_result(
            run_discovery(
                RecurrenceSpanFamily(
                    samples=tuple(kwargs.get("samples", catalan_samples())),
                    max_order=int(kwargs.get("max_order", 2)),
                    max_index_degree=int(kwargs.get("max_index_degree", 1)),
                ).statement,
                RecurrenceSpanFamily(
                    samples=tuple(kwargs.get("samples", catalan_samples())),
                    max_order=int(kwargs.get("max_order", 2)),
                    max_index_degree=int(kwargs.get("max_index_degree", 1)),
                ),
                str(kwargs.get("proposer", "score_guided")),
                budget=int(kwargs.get("budget", 8)),
            )
        ),
        ACTIVATION_IDENTITY_EXACT: lambda **kwargs: _from_result(
            run_discovery(
                ActivationIdentityFamily().statement,
                ActivationIdentityFamily(),
                str(kwargs.get("proposer", "score_guided")),
                budget=int(kwargs.get("budget", 32)),
            )
        ),
        PDE_OPERATOR_SPAN: lambda **kwargs: _from_result(
            run_discovery(
                PdeOperatorSpanFamily().statement,
                PdeOperatorSpanFamily(),
                "score_guided",
                budget=1,
            )
        ),
    }
    parents = {
        RECURRENCE_SPAN: "P-recursive sequences",
        ACTIVATION_IDENTITY_EXACT: "Riccati identities",
        PDE_OPERATOR_SPAN: "heat equation",
    }
    for kind, meta in FAMILY_CATALOG.items():
        register_catalog(
            CatalogEntry(
                kind=kind,
                obligation=meta["obligation"],
                parent=parents[kind],
                parent_status=meta["parent_status"],  # type: ignore[arg-type]
                package="omnibias.symbolic",
                mode="exact_search",
                complete=meta.get("complete", "True") == "True",
            ),
            factories[kind],
        )


def symbolic_provers() -> list[FunctionProver]:
    return [
        FunctionProver(
            name=RECURRENCE_SPAN,
            kinds=frozenset({RECURRENCE_SPAN}),
            prove_fn=_prove_recurrence,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
        FunctionProver(
            name=ACTIVATION_IDENTITY_EXACT,
            kinds=frozenset({ACTIVATION_IDENTITY_EXACT}),
            prove_fn=_prove_activation,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
        FunctionProver(
            name=PDE_OPERATOR_SPAN,
            kinds=frozenset({PDE_OPERATOR_SPAN}),
            prove_fn=_prove_pde,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
    ]


def build_symbolic_machine() -> ProofMachine:
    machine = ProofMachine()
    for prover in symbolic_provers():
        machine.register(prover)
    return machine


def _run_family(family: Any, **kwargs: Any) -> Any:
    return run_discovery(
        family.statement,
        family,
        str(kwargs.get("proposer", "score_guided")),
        budget=int(kwargs.get("budget", 16)),
        collect=bool(kwargs.get("collect", False)),
    )


def _register_condition_catalog() -> None:
    for kind, obligation, parent, factory in (
        (
            CONDITION_JET,
            "a jet-monomial identity for tanh in this grammar",
            "Riccati identities",
            lambda **kwargs: _run_family(JetConditionFamily(), **kwargs),
        ),
        (
            CONDITION_PDE,
            "snapped heat-operator coefficients with identically zero residual",
            "heat equation",
            lambda **kwargs: _run_family(PdeConditionFamily(), **kwargs),
        ),
        (
            CONDITION_CONSERVATION,
            "ρ_t + j_x = 0 on the planted integer grid",
            "continuity equation",
            lambda **kwargs: _run_family(ConservationConditionFamily(), **kwargs),
        ),
        (
            CONDITION_FRACTIONAL,
            "an integer order n with D^n (x^2) identically 2x",
            "integer differential order",
            lambda **kwargs: _run_family(FractionalOrderFamily(), **kwargs),
        ),
        (
            CONDITION_PIECEWISE,
            "y' = sign(x) on the two-region split of y=|x|",
            "piecewise identities",
            lambda **kwargs: _run_family(PiecewiseHybridFamily(), **kwargs),
        ),
        (
            CONDITION_GRAMMAR_GROWTH,
            "a grown jet hypothesis is an exact identity for y=x^2",
            "jet identities",
            lambda **kwargs: _run_family(
                jet_growth_family(complete=bool(kwargs.get("complete", False))),
                **kwargs,
            ),
        ),
        (
            CONDITION_KIND_META,
            "some registered condition sort has an exact witness in its box",
            "condition language",
            lambda **kwargs: _run_family(
                KindMetaFamily(
                    sorts=tuple(kwargs.get("sorts") or list_condition_sorts()),
                    observation=kwargs.get("observation"),
                    grammar_complete=bool(kwargs.get("grammar_complete", False)),
                    budget=int(kwargs.get("inner_budget", kwargs.get("budget", 8))),
                ),
                **kwargs,
            ),
        ),
        (
            CONDITION_RESIDUAL_SIGN,
            "every packed residual sample has the same sign",
            "residual inequalities",
            lambda **kwargs: _run_family(
                ResidualSignFamily(
                    observation=kwargs.get("observation")
                    or Observation(extra=(("residual", "1,1,1"),))
                ),
                **kwargs,
            ),
        ),
    ):
        register_catalog(
            CatalogEntry(
                kind=kind,
                obligation=obligation,
                parent=parent,
                parent_status="already_true",
                package="omnibias.symbolic",
                mode="exact_search",
                complete=False,
            ),
            factory,
        )


_register()
_register_condition_catalog()

__all__ = [
    "ACTIVATION_IDENTITY_EXACT",
    "CONDITION_CONSERVATION",
    "CONDITION_FRACTIONAL",
    "CONDITION_GRAMMAR_GROWTH",
    "CONDITION_JET",
    "CONDITION_KIND_META",
    "CONDITION_PDE",
    "CONDITION_PIECEWISE",
    "CONDITION_RESIDUAL_SIGN",
    "FAMILY_CATALOG",
    "PDE_OPERATOR_SPAN",
    "RECURRENCE_SPAN",
    "build_symbolic_machine",
    "propose_from_residual",
    "propose_jet_condition",
    "symbolic_provers",
]

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Register finite combinatorics provers on :class:`ProofMachine`.

===============================  ==================================================
kind                             prover
===============================  ==================================================
``dgg_cost_replay``              :func:`omnibias.combinatorics.unsplittable.verify_rybin_instance`
``dgg_cost_search``              :func:`omnibias.combinatorics.unsplittable.search_dgg`
``dgg_dag_le6``                  capped 3-terminal DAG ≤6 via :func:`omnibias.core.proof.run_discovery`
``ramsey_triangle_free_colouring`` :func:`omnibias.combinatorics.ramsey.verify_pentagon_colouring`
``ramsey_saturated_matrix``      :func:`omnibias.combinatorics.ramsey.verify_saturated_smoke`
``extremal_forbidden_family``    :func:`omnibias.combinatorics.extremal.verify_forbidden_family`
``extremal_pair_graph``          :func:`omnibias.combinatorics.extremal.verify_pair_graph`
===============================  ==================================================

``PROVED`` certifies the named finite obligation. Parent Erdős / Goemans
statements stay external. ``omnibias-pinn`` must not import this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omnibias.combinatorics.conditions import (
    EdgeColouringFamily,
    ExtremalTemplateConditionFamily,
)
from omnibias.combinatorics.extremal import (
    ExtremalSearchFamily,
    verify_forbidden_family,
    verify_pair_graph,
)
from omnibias.combinatorics.minors import ForbiddenMinorFamily
from omnibias.combinatorics.ramsey import verify_pentagon_colouring, verify_saturated_smoke
from omnibias.combinatorics.ramsey_search import RamseyColouringFamily
from omnibias.combinatorics.unsplittable import (
    search_dgg,
    search_hit_certificate,
    verify_rybin_instance,
)
from omnibias.combinatorics.unsplittable_dags import dag_hit_certificate, search_dgg_dags
from omnibias.core.proof import (
    CatalogEntry,
    Certificate,
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    register_catalog,
    run_discovery,
)

DGG_COST_REPLAY = "dgg_cost_replay"
DGG_COST_SEARCH = "dgg_cost_search"
DGG_DAG_LE6 = "dgg_dag_le6"
RAMSEY_TRIANGLE_FREE = "ramsey_triangle_free_colouring"
RAMSEY_SATURATED = "ramsey_saturated_matrix"
EXTREMAL_FORBIDDEN = "extremal_forbidden_family"
EXTREMAL_PAIR_GRAPH = "extremal_pair_graph"
RAMSEY_COLOURING_SEARCH = "ramsey_colouring_search"
EXTREMAL_TEMPLATE_SEARCH = "extremal_template_search"
CONDITION_FORBIDDEN_MINOR = "condition_forbidden_minor"
CONDITION_EDGE_COLOURING = "condition_edge_colouring"
CONDITION_EXTREMAL_TEMPLATE = "condition_extremal_template"

FAMILY_CATALOG: dict[str, dict[str, str]] = {
    DGG_COST_REPLAY: {
        "parent_status": "already_false",
        "obligation": "AFP instance: fractional cost < min congestion-legal unsplittable",
    },
    DGG_COST_SEARCH: {
        "parent_status": "already_false",
        "obligation": "some H* parameter-box instance with a cost separation",
        "complete": "False",
    },
    DGG_DAG_LE6: {
        "parent_status": "already_false",
        "obligation": "a 3-terminal DAG on at most 6 vertices with a cost separation",
        "complete": "False",
    },
    RAMSEY_TRIANGLE_FREE: {
        "parent_status": "already_true",
        "obligation": "a triangle-free 2-edge-colouring of K_5 (R_2(3) > 5)",
    },
    RAMSEY_SATURATED: {
        "parent_status": "already_true",
        "obligation": "the IsSaturated predicate on a tiny matrix",
    },
    EXTREMAL_FORBIDDEN: {
        "parent_status": "already_true",
        "obligation": "C4/C6/jTemplate/kTemplate are connected bipartite and have a cycle",
    },
    EXTREMAL_PAIR_GRAPH: {
        "parent_status": "already_true",
        "obligation": "pairGraph(4,2) is connected, bipartite, 2-degenerate, and has deg > 2",
    },
    RAMSEY_COLOURING_SEARCH: {
        "parent_status": "already_true",
        "obligation": "a triangle-free 2-edge-colouring of K_5 (R_2(3) > 5)",
        "complete": "True",
    },
    EXTREMAL_TEMPLATE_SEARCH: {
        "parent_status": "already_true",
        "obligation": "a named template that is connected, bipartite, and has a cycle",
        "complete": "True",
    },
}


def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _from_dict(payload: Mapping[str, Any]) -> ProofAttempt:
    if not payload.get("replay_ok"):
        return _blocked(str(payload.get("kind", "combinatorics")))
    return ProofAttempt(status="PROVED", certificate=dict(payload), detail=str(payload.get("kind")))


def _prove_dgg_replay(_conjecture: Conjecture) -> ProofAttempt:
    return _from_dict(verify_rybin_instance())


def _prove_dgg_search(conjecture: Conjecture) -> ProofAttempt:
    family = str(conjecture.data.get("family", "hstar_parameter_box"))
    planted = conjecture.data.get("instances")
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    hits = search_dgg(
        family=family,
        instances=planted,
        max_hits=int(conjecture.data.get("max_hits", 1)),
        proposer=proposer,
    )
    if not hits:
        return _blocked("no cost-separating H* instance in the search box")
    return _from_dict(search_hit_certificate(hits[0]))


def _from_discovery(result: object) -> ProofAttempt:
    payload = result.as_dict()  # type: ignore[attr-defined]
    status = result.status  # type: ignore[attr-defined]
    detail = result.detail  # type: ignore[attr-defined]
    if status == "PROVED":
        return ProofAttempt(status="PROVED", certificate=payload, detail=detail)
    if status == "DISPROVED":
        return ProofAttempt(status="DISPROVED", certificate=payload, detail=detail)
    return _blocked(detail)


def _prove_ramsey_search(conjecture: Conjecture) -> ProofAttempt:
    family = RamseyColouringFamily()
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    budget = int(conjecture.data.get("budget", 256))
    return _from_discovery(run_discovery(family.statement, family, proposer, budget=budget))


def _prove_extremal_search(conjecture: Conjecture) -> ProofAttempt:
    family = ExtremalSearchFamily()
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    return _from_discovery(run_discovery(family.statement, family, proposer, budget=4))


def _prove_dgg_dag(conjecture: Conjecture) -> ProofAttempt:
    family = str(conjecture.data.get("family", "three_terminal_dag_le6"))
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    hits = search_dgg_dags(family=family, proposer=proposer)
    if not hits:
        return _blocked("search_incomplete")
    return _from_dict(dag_hit_certificate(hits[0]))


def _schema_errors(certificate: Certificate) -> list[str]:
    errors: list[str] = []
    honesty = certificate.get("honesty", {})
    if not isinstance(honesty, Mapping):
        return ["honesty must be a mapping"]
    for key in (
        "dgg_congestion_theorem_refuted",
        "erdos_183_claim",
        "erdos_146_claim",
        "erdos_180_claim",
        "ten_proofs_formalization_claim",
    ):
        if honesty.get(key):
            errors.append(f"{key} must be False")
    return errors


def _replay_flag(certificate: Certificate, builder: Any) -> bool | None:
    fresh = builder()
    return bool(fresh.get("replay_ok")) and fresh.get("kind") == certificate.get("kind")


def combinatorics_provers() -> list[FunctionProver]:
    return [
        FunctionProver(
            name="dgg_cost_replay",
            kinds=frozenset({DGG_COST_REPLAY}),
            prove_fn=_prove_dgg_replay,
            schema_fn=_schema_errors,
            replay_fn=lambda c: _replay_flag(c, verify_rybin_instance),
        ),
        FunctionProver(
            name="dgg_cost_search",
            kinds=frozenset({DGG_COST_SEARCH}),
            prove_fn=_prove_dgg_search,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
        FunctionProver(
            name="dgg_dag_le6",
            kinds=frozenset({DGG_DAG_LE6}),
            prove_fn=_prove_dgg_dag,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
        FunctionProver(
            name="ramsey_triangle_free_colouring",
            kinds=frozenset({RAMSEY_TRIANGLE_FREE}),
            prove_fn=lambda _c: _from_dict(verify_pentagon_colouring()),
            schema_fn=_schema_errors,
            replay_fn=lambda c: _replay_flag(c, verify_pentagon_colouring),
        ),
        FunctionProver(
            name="ramsey_saturated_matrix",
            kinds=frozenset({RAMSEY_SATURATED}),
            prove_fn=lambda _c: _from_dict(verify_saturated_smoke()),
            schema_fn=_schema_errors,
            replay_fn=lambda c: _replay_flag(c, verify_saturated_smoke),
        ),
        FunctionProver(
            name="extremal_forbidden_family",
            kinds=frozenset({EXTREMAL_FORBIDDEN}),
            prove_fn=lambda _c: _from_dict(verify_forbidden_family()),
            schema_fn=_schema_errors,
            replay_fn=lambda c: _replay_flag(c, verify_forbidden_family),
        ),
        FunctionProver(
            name="extremal_pair_graph",
            kinds=frozenset({EXTREMAL_PAIR_GRAPH}),
            prove_fn=lambda _c: _from_dict(verify_pair_graph()),
            schema_fn=_schema_errors,
            replay_fn=lambda c: _replay_flag(c, verify_pair_graph),
        ),
        FunctionProver(
            name=RAMSEY_COLOURING_SEARCH,
            kinds=frozenset({RAMSEY_COLOURING_SEARCH}),
            prove_fn=_prove_ramsey_search,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
        FunctionProver(
            name=EXTREMAL_TEMPLATE_SEARCH,
            kinds=frozenset({EXTREMAL_TEMPLATE_SEARCH}),
            prove_fn=_prove_extremal_search,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
    ]


def build_combinatorics_machine() -> ProofMachine:
    machine = ProofMachine()
    for prover in combinatorics_provers():
        machine.register(prover)
    return machine


def _register() -> None:
    replay = {
        DGG_COST_REPLAY,
        RAMSEY_TRIANGLE_FREE,
        RAMSEY_SATURATED,
        EXTREMAL_FORBIDDEN,
        EXTREMAL_PAIR_GRAPH,
    }
    parents = {
        DGG_COST_REPLAY: "Goemans cost / DGG",
        DGG_COST_SEARCH: "Goemans cost / DGG",
        DGG_DAG_LE6: "Goemans cost / DGG",
        RAMSEY_TRIANGLE_FREE: "Erdős 183",
        RAMSEY_SATURATED: "Erdős 183",
        EXTREMAL_FORBIDDEN: "Erdős 146 / 180",
        EXTREMAL_PAIR_GRAPH: "Erdős 146 / 180",
        RAMSEY_COLOURING_SEARCH: "Erdős 183",
        EXTREMAL_TEMPLATE_SEARCH: "Erdős 146 / 180",
    }
    factories = {
        DGG_COST_REPLAY: lambda **_k: verify_rybin_instance(),
        RAMSEY_TRIANGLE_FREE: lambda **_k: verify_pentagon_colouring(),
        RAMSEY_COLOURING_SEARCH: lambda **kwargs: run_discovery(
            RamseyColouringFamily().statement,
            RamseyColouringFamily(),
            str(kwargs.get("proposer", "score_guided")),
            budget=int(kwargs.get("budget", 256)),
        ),
        EXTREMAL_TEMPLATE_SEARCH: lambda **kwargs: run_discovery(
            ExtremalSearchFamily().statement,
            ExtremalSearchFamily(),
            "score_guided",
            budget=4,
        ),
    }
    for kind, meta in FAMILY_CATALOG.items():
        register_catalog(
            CatalogEntry(
                kind=kind,
                obligation=meta["obligation"],
                parent=parents[kind],
                parent_status=meta["parent_status"],  # type: ignore[arg-type]
                package="omnibias.combinatorics",
                mode="exact_replay" if kind in replay else "exact_search",
                complete=meta.get("complete", "True") != "False",
            ),
            factories.get(kind),
        )


_register()
register_catalog(
    CatalogEntry(
        kind=CONDITION_FORBIDDEN_MINOR,
        obligation="the host contains a named minor in this grammar",
        parent="graph minors",
        parent_status="already_true",
        package="omnibias.combinatorics",
        mode="exact_search",
        complete=False,
    ),
    lambda **kwargs: run_discovery(
        ForbiddenMinorFamily().statement,
        ForbiddenMinorFamily(),
        "score_guided",
        budget=int(kwargs.get("budget", 4)),
    ),
)
register_catalog(
    CatalogEntry(
        kind=CONDITION_EDGE_COLOURING,
        obligation="a triangle-free 2-edge-colouring of K_5 (R_2(3) > 5)",
        parent="finite Ramsey colourings",
        parent_status="already_true",
        package="omnibias.combinatorics",
        mode="exact_search",
        complete=False,
    ),
    lambda **kwargs: run_discovery(
        EdgeColouringFamily().statement,
        EdgeColouringFamily(),
        "score_guided",
        budget=int(kwargs.get("budget", 32)),
    ),
)
register_catalog(
    CatalogEntry(
        kind=CONDITION_EXTREMAL_TEMPLATE,
        obligation="a named template that is connected, bipartite, and has a cycle",
        parent="extremal graph templates",
        parent_status="already_true",
        package="omnibias.combinatorics",
        mode="exact_search",
        complete=False,
    ),
    lambda **kwargs: run_discovery(
        ExtremalTemplateConditionFamily().statement,
        ExtremalTemplateConditionFamily(),
        "score_guided",
        budget=int(kwargs.get("budget", 4)),
    ),
)


__all__ = [
    "CONDITION_EDGE_COLOURING",
    "CONDITION_EXTREMAL_TEMPLATE",
    "CONDITION_FORBIDDEN_MINOR",
    "DGG_COST_REPLAY",
    "DGG_COST_SEARCH",
    "DGG_DAG_LE6",
    "EXTREMAL_FORBIDDEN",
    "EXTREMAL_PAIR_GRAPH",
    "EXTREMAL_TEMPLATE_SEARCH",
    "FAMILY_CATALOG",
    "RAMSEY_COLOURING_SEARCH",
    "RAMSEY_SATURATED",
    "RAMSEY_TRIANGLE_FREE",
    "build_combinatorics_machine",
    "combinatorics_provers",
]

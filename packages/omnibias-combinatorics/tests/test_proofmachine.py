# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Combinatorics ProofMachine kinds, catalog, and planted recovery."""

from __future__ import annotations

from omnibias.combinatorics.proofmachine import (
    DGG_COST_REPLAY,
    DGG_COST_SEARCH,
    DGG_DAG_LE6,
    EXTREMAL_FORBIDDEN,
    EXTREMAL_PAIR_GRAPH,
    FAMILY_CATALOG,
    RAMSEY_SATURATED,
    RAMSEY_TRIANGLE_FREE,
    build_combinatorics_machine,
)
from omnibias.core.proof import Conjecture


def test_kinds_and_catalog() -> None:
    machine = build_combinatorics_machine()
    assert sorted(machine.kinds()) == sorted(FAMILY_CATALOG)
    assert FAMILY_CATALOG[DGG_COST_REPLAY]["parent_status"] == "already_false"
    assert FAMILY_CATALOG[DGG_DAG_LE6]["complete"] == "False"
    assert FAMILY_CATALOG[RAMSEY_TRIANGLE_FREE]["parent_status"] == "already_true"


def test_replay_kinds_prove() -> None:
    machine = build_combinatorics_machine()
    for kind in (
        DGG_COST_REPLAY,
        RAMSEY_TRIANGLE_FREE,
        RAMSEY_SATURATED,
        EXTREMAL_FORBIDDEN,
        EXTREMAL_PAIR_GRAPH,
    ):
        verdict = machine.evaluate(Conjecture(name=kind, kind=kind))
        assert verdict.status == "PROVED", kind
        assert verdict.replay_ok is True


def test_dgg_search_and_planted() -> None:
    machine = build_combinatorics_machine()
    planted = machine.evaluate(
        Conjecture(
            name="planted",
            kind=DGG_COST_SEARCH,
            data={
                "family": "planted",
                "instances": (
                    ((8, 8, 8), (0, 0, 0), (4, 4, 4)),
                    ((15, 10, 15), (2, 3, 2), (10, 6, 10)),
                ),
            },
        )
    )
    assert planted.status == "PROVED"
    boxed = machine.evaluate(Conjecture(name="box", kind=DGG_COST_SEARCH))
    assert boxed.status == "PROVED"
    assert boxed.certificate is not None
    assert boxed.certificate["honesty"]["dgg_congestion_theorem_refuted"] is False
    dag_miss = machine.evaluate(Conjecture(name="dag", kind=DGG_DAG_LE6))
    assert dag_miss.status == "BLOCKED"
    assert "search_incomplete" in dag_miss.obligations
    dag_hit = machine.evaluate(
        Conjecture(name="planted", kind=DGG_DAG_LE6, data={"family": "planted"})
    )
    assert dag_hit.status == "PROVED"
    assert dag_hit.certificate is not None
    assert dag_hit.certificate["honesty"]["dgg_congestion_theorem_refuted"] is False

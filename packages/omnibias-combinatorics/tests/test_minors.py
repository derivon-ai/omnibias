# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Named forbidden-minor hypotheses on hosts with n<=5."""

from __future__ import annotations

from omnibias.combinatorics.minors import (
    ForbiddenMinorFamily,
    bind_forbidden_minor,
    contains_minor,
    k3,
    k4,
    observation_k4,
    observation_path,
    path_graph,
)
from omnibias.core.proof import (
    ConditionHypothesis,
    ConditionToken,
    GrammarSpec,
    Observation,
    bind_sorts,
    run_discovery,
)


def test_k4_contains_k3_not_path() -> None:
    assert contains_minor(k4(), k3()) is True
    assert contains_minor(path_graph(4), k3()) is False


def test_forbidden_minor_k4_finds_k3() -> None:
    family = ForbiddenMinorFamily(host=k4(), patterns=("K2", "K3", "C4"))
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["no_condition_exists_claim"] is False
    assert result.check.payload["honesty"]["erdos_146_claim"] is False


def test_path_has_no_k3_in_complete_grammar() -> None:
    family = ForbiddenMinorFamily(host=path_graph(4), patterns=("K3", "C4"))
    family.grammar = GrammarSpec(
        sorts=("forbidden_minor",),
        tokens_by_sort={"forbidden_minor": ("K3", "C4")},
        complete=True,
    )
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is False
    assert result.detail == "no witness in enumerated grammar"


def test_graph_only_observation_binds_only_minor() -> None:
    obs = observation_k4()
    bound = bind_sorts(obs)
    assert set(bound) == {"forbidden_minor"}
    family = bind_forbidden_minor(obs)
    assert family is not None
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    path = observation_path(4)
    path_family = bind_forbidden_minor(path)
    assert path_family is not None
    assert bind_forbidden_minor(Observation(tag="tanh")) is None


def test_path_k3_incomplete_grammar_is_search_incomplete() -> None:
    family = ForbiddenMinorFamily(host=path_graph(4), patterns=("K3",))
    hyp = ConditionHypothesis(
        sort="forbidden_minor",
        tokens=(ConditionToken("forbidden_minor", "K3"),),
    )
    checked = family.check(hyp)
    assert checked is not None
    assert checked.ok is False
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True

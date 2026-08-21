# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Ore condition family: Fibonacci hit, Bell miss."""

from __future__ import annotations

from omnibias.core.proof import GrammarSpec, Observation, bind_sorts, run_discovery
from omnibias.holonomic.conditions import (
    DfiniteConditionFamily,
    OreConditionFamily,
    bind_dfinite,
    bind_ore,
    observation_exp_series,
    observation_fibonacci,
)
from omnibias.symbolic.families import bell_samples


def test_ore_condition_fibonacci_hits() -> None:
    family = OreConditionFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["discovered_by_omnibias"] is True
    assert result.check.payload["honesty"]["no_condition_exists_claim"] is False
    assert result.check.payload["honesty"]["jacobian_conjecture_proof_claim"] is False


def test_ore_bell_incomplete_grammar_is_search_incomplete() -> None:
    family = OreConditionFamily(samples=bell_samples())
    assert family.complete is False
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True


def test_sequence_only_observation_binds_only_ore() -> None:
    obs = observation_fibonacci()
    bound = bind_sorts(obs)
    assert set(bound) == {"ore"}
    family = bind_ore(obs)
    assert family is not None
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"
    empty = Observation(tag="tanh", jet_names=("y",), jet_rows=(("1",),))
    assert bind_ore(empty) is None
    assert bind_dfinite(obs) is None


def test_dfinite_exp_series_hits() -> None:
    obs = observation_exp_series()
    family = bind_dfinite(obs)
    assert family is not None
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"
    catalog = DfiniteConditionFamily()
    replay = run_discovery(catalog.statement, catalog, "score_guided", budget=8)
    assert replay.status == "PROVED"


def test_ore_bell_complete_grammar_empty_box() -> None:
    family = OreConditionFamily(samples=bell_samples())
    family.grammar = GrammarSpec(
        sorts=("ore",),
        tokens_by_sort={"ore": ("S", "n")},
        constructors=("raise_ore_degree",),
        complete=True,
    )
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is False
    assert result.detail == "no witness in enumerated grammar"

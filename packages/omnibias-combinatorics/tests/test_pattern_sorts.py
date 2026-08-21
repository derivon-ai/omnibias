# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Edge-colouring and extremal-template condition sorts."""

from __future__ import annotations

from omnibias.combinatorics.conditions import (
    bind_edge_colouring,
    bind_extremal_template,
    observation_c4,
    observation_k5,
)
from omnibias.combinatorics.minors import observation_k4
from omnibias.core.proof import Observation, bind_sorts, run_discovery, select_class


def test_k4_does_not_bind_pattern_sorts() -> None:
    obs = observation_k4()
    bound = bind_sorts(obs)
    assert "forbidden_minor" in bound
    assert "edge_colouring" not in bound
    assert "extremal_template" not in bound


def test_c4_binds_extremal_not_colouring() -> None:
    obs = observation_c4()
    assert "extremal_template" in bind_sorts(obs)
    assert bind_edge_colouring(obs) is None
    selection = select_class(obs, sorts=("extremal_template",))
    assert selection.best is not None
    assert selection.honesty["erdos_146_claim"] is False
    assert selection.honesty["erdos_180_claim"] is False


def test_k5_colouring_planted_bits() -> None:
    bits = "0,1,1,0,0,1,1,0,1,0"
    obs = Observation(graph_n=5, graph_edges=observation_k5().graph_edges, extra=(("colouring", bits),))
    family = bind_edge_colouring(obs)
    assert family is not None
    result = run_discovery(family.statement, family, "score_guided", budget=1)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["erdos_183_claim"] is False


def test_k5_without_bits_binds() -> None:
    obs = observation_k5()
    assert bind_extremal_template(obs) is None
    assert bind_edge_colouring(obs) is not None

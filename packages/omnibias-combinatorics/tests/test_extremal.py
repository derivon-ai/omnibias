# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Structural replay of C4/C6/templates and pairGraph(4,2)."""

from __future__ import annotations

from omnibias.combinatorics.extremal import (
    c4,
    c6,
    is_bipartite,
    is_connected,
    is_two_degenerate,
    j_template,
    k_template,
    max_degree,
    pair_graph,
    verify_forbidden_family,
    verify_pair_graph,
)


def test_cycles() -> None:
    assert len(c4()) == 4 and is_connected(c4()) and is_bipartite(c4())
    assert len(c6()) == 6 and is_connected(c6()) and is_bipartite(c6())


def test_templates() -> None:
    jg = j_template()
    kg = k_template()
    assert len(jg) == 21
    assert len(kg) == 30
    assert is_connected(jg) and is_bipartite(jg)
    assert is_connected(kg) and is_bipartite(kg)
    cert = verify_forbidden_family()
    assert cert["replay_ok"] is True
    assert cert["honesty"]["erdos_146_claim"] is False
    assert cert["honesty"]["erdos_180_claim"] is False
    assert cert["honesty"]["extremal_graph_replay"] is True


def test_pair_graph_smoke() -> None:
    graph = pair_graph(4, 2)
    assert is_connected(graph)
    assert is_bipartite(graph)
    assert is_two_degenerate(graph)
    assert max_degree(graph) > 2
    cert = verify_pair_graph()
    assert cert["replay_ok"] is True
    assert cert["honesty"]["erdos_146_claim"] is False

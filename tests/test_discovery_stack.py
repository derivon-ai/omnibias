# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stack loader and tag-free ingest. Missing extras skip."""

from __future__ import annotations

from omnibias.core.proof import FEATURE_DIM, Observation, bind_sorts, select_class
from omnibias.symbolic.classloop import discover_observation, load_discovery_stack
from omnibias.symbolic.conditions import observation_square, observation_tanh
from omnibias.symbolic.ingest import pack_graph, pack_jets, pack_sequence


def test_load_stack_registers_available_sorts() -> None:
    sorts = load_discovery_stack()
    assert "jet_monomial" in sorts
    assert FEATURE_DIM == 16


def test_discover_observation_tanh_stack() -> None:
    payload = discover_observation(
        observation_tanh(),
        sorts=("jet_monomial",),
        proposers=("stlsq",),
    )
    assert payload["best"] == "jet_monomial"
    assert payload["no_condition_exists_claim"] is False
    assert payload["grammar_complete"] is False


def test_square_grow_without_tag() -> None:
    obs = observation_square()
    assert obs.tag == ""
    selection = select_class(obs, sorts=("jet_monomial",), grow=True)
    assert selection.best is not None
    assert selection.best.tier == "exact"


def test_pack_helpers_are_tag_free() -> None:
    seq = pack_sequence((0, 1, 1, 2))
    jets = pack_jets({"y": (1, 4), "yp": (2, 4)}, sample_x=(1, 2))
    graph = pack_graph(4, ((0, 1), (1, 2), (2, 3)))
    assert seq.tag == jets.tag == graph.tag == ""
    assert isinstance(Observation.from_dict(jets.as_dict()), Observation)
    bound = bind_sorts(jets, ("jet_monomial",))
    assert "jet_monomial" in bound

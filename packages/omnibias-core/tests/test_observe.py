# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Observation, ranking, memory, and select_class honesty."""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibias.core.proof import (
    FEATURE_DIM,
    ClassHit,
    ClassMemory,
    ExactCheck,
    FrequencyGate,
    KindMetaFamily,
    LinearSpanFamily,
    Observation,
    Statement,
    bind_sorts,
    condition_honesty,
    rank_class_hits,
    run_discovery,
    select_class,
)
from omnibias.core.proof.condition import _SORT_FACTORIES, _reset_condition_sorts_for_tests
from omnibias.core.proof.discovery import Candidate


@dataclass
class _AlwaysMiss:
    name: str = "always_miss"
    complete: bool = True
    empty_miss_detail: str = "no witness in enumerated grammar"
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="always_miss",
            obligation="planted miss",
            parent="condition language",
            parent_status="open",
        )
    )

    def cardinality(self) -> int:
        return 1

    def origin(self) -> int:
        return 0

    def neighbors(self, candidate: Candidate) -> tuple[int, ...]:
        return ()

    def score(self, candidate: Candidate) -> int:
        return 0

    def check(self, candidate: Candidate) -> ExactCheck:
        return ExactCheck(ok=False, payload={"honesty": condition_honesty(discovered=False)})


@dataclass
class _AlwaysHit:
    name: str = "always_hit"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="always_hit",
            obligation="planted hit",
            parent="condition language",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return 1

    def origin(self) -> int:
        return 1

    def neighbors(self, candidate: Candidate) -> tuple[int, ...]:
        return ()

    def score(self, candidate: Candidate) -> int:
        return 0

    def check(self, candidate: Candidate) -> ExactCheck:
        return ExactCheck(
            ok=True,
            payload={"tier": "exact", "tokens": 2, "honesty": condition_honesty(discovered=True)},
        )


def _isolated() -> dict[str, object]:
    saved = dict(_SORT_FACTORIES)
    _reset_condition_sorts_for_tests()
    return saved


def _restore(saved: dict[str, object]) -> None:
    _SORT_FACTORIES.clear()
    _SORT_FACTORIES.update(saved)  # type: ignore[arg-type]


def test_observation_json_and_features() -> None:
    obs = Observation(tag="fibonacci", sequence=("0", "1", "1", "2"))
    features = obs.features()
    assert len(features) == FEATURE_DIM
    assert features[0] == 1
    assert features[1] == 0
    assert features[5] == 4
    payload = obs.as_dict()
    restored = Observation.from_dict(payload)
    assert restored.tag == "fibonacci"
    assert restored.sequence == obs.sequence
    assert restored.sample_x == ()
    assert restored.poly_constraints == ()
    assert restored.features() == features


def test_binder_none_drops_sort_and_forces_incomplete() -> None:
    saved = _isolated()
    try:
        def _bind_ore(obs: Observation | None) -> object | None:
            if obs is None:
                return _AlwaysHit()
            return None

        from omnibias.core.proof.condition import register_condition_sort

        register_condition_sort("ore", _bind_ore)
        obs = Observation(tag="tanh", jet_names=("y",), jet_rows=(("1",),))
        assert bind_sorts(obs, ("ore",)) == {}
        meta = KindMetaFamily(
            sorts=("ore",),
            observation=obs,
            grammar_complete=True,
        )
        assert meta.complete is False
        result = run_discovery(meta.statement, meta, "score_guided", budget=4)
        assert result.status == "BLOCKED"
        assert result.search_incomplete is True
    finally:
        _restore(saved)


def test_select_class_miss_is_search_incomplete() -> None:
    saved = _isolated()
    try:
        from omnibias.core.proof.condition import register_condition_sort

        register_condition_sort("jet_monomial", lambda obs: _AlwaysMiss())
        obs = Observation(tag="tanh", jet_names=("y",), jet_rows=(("1",),))
        selection = select_class(obs, sorts=("jet_monomial",))
        assert selection.best is None
        assert selection.search_incomplete is True
        assert selection.grammar_complete is False
        assert selection.honesty["no_condition_exists_claim"] is False
        assert selection.honesty["unnamed_condition_complete_claim"] is False
        assert selection.result is not None
        assert selection.result.detail == "search_incomplete"
    finally:
        _restore(saved)


def test_rank_exact_beats_empirical() -> None:
    exact = ClassHit(
        sort="jet_monomial",
        check=ExactCheck(ok=True, payload={"tier": "exact", "honesty": {}}),
        tier="exact",
        tokens=3,
        degree=2,
    )
    empirical = ClassHit(
        sort="pde_operator",
        check=ExactCheck(ok=False, payload={"mode": "empirical"}),
        tier="empirical",
        tokens=1,
        degree=0,
        rmse="0.001",
    )
    ranked = rank_class_hits([empirical, exact])
    assert ranked[0].sort == "jet_monomial"
    assert all(hit.tier != "empirical" for hit in ranked)


def test_rank_two_exact_by_token_count() -> None:
    fat = ClassHit(
        sort="ore",
        check=ExactCheck(ok=True, payload={"tier": "exact"}),
        tier="exact",
        tokens=5,
        degree=0,
    )
    thin = ClassHit(
        sort="jet_monomial",
        check=ExactCheck(ok=True, payload={"tier": "exact"}),
        tier="exact",
        tokens=2,
        degree=4,
    )
    ranked = rank_class_hits([fat, thin])
    assert ranked[0].sort == "jet_monomial"
    assert ranked[0].tokens == 2


def test_class_memory_reorders_after_record() -> None:
    memory = ClassMemory()
    obs = Observation(tag="tanh", jet_names=("y",), jet_rows=(("1",),))
    before = memory.propose(obs)
    memory.record(obs, "jet_monomial")
    after = memory.propose(obs)
    assert after[0] == "jet_monomial"
    assert FrequencyGate(memory=memory).propose(obs)[0] == "jet_monomial"
    assert before[0] != "jet_monomial" or after[0] == "jet_monomial"


def test_zero_budget_select_class_is_incomplete() -> None:
    saved = _isolated()
    try:
        from omnibias.core.proof.condition import register_condition_sort

        register_condition_sort("jet_monomial", lambda obs: _AlwaysHit())
        obs = Observation(tag="tanh", jet_names=("y",), jet_rows=(("1",),))
        selection = select_class(obs, sorts=("jet_monomial",), budget=0)
        assert selection.result is not None
        assert selection.result.status == "BLOCKED"
        assert selection.result.search_incomplete is True
        assert selection.result.evaluated == 0
        assert selection.honesty["no_condition_exists_claim"] is False
    finally:
        _restore(saved)


def test_kind_meta_does_not_prefer_jet_score() -> None:
    saved = _isolated()
    try:
        meta = KindMetaFamily(
            sorts=("sos_template", "jet_monomial"),
            families={"sos_template": _AlwaysHit(), "jet_monomial": _AlwaysHit()},
            grammar_complete=True,
            budget=2,
        )
        assert meta.score("jet_monomial") == 0
        assert meta.score("sos_template") == 0
        result = run_discovery(meta.statement, meta, "score_guided", budget=4)
        assert result.candidate == "sos_template"
    finally:
        _restore(saved)


def test_class_memory_json_roundtrip(tmp_path) -> None:
    memory = ClassMemory()
    obs = Observation(tag="tanh", jet_names=("y",), jet_rows=(("1",),))
    memory.record(obs, "jet_monomial")
    path = tmp_path / "memory.json"
    memory.save(path)
    loaded = ClassMemory.load(path)
    assert loaded.propose(obs)[0] == "jet_monomial"


def test_select_class_grows_linear_span_products() -> None:
    saved = _isolated()
    try:
        from omnibias.core.proof.condition import register_condition_sort

        family = LinearSpanFamily(
            design=((1, 2, 2), (4, 4, 2), (9, 6, 2), (16, 8, 2)),
            term_names=("y", "yp", "ypp"),
            sort="jet_monomial",
        )
        register_condition_sort("jet_monomial", lambda obs: family)
        obs = Observation(
            jet_names=("y", "yp", "ypp"),
            jet_rows=(("1", "2", "2"), ("4", "4", "2"), ("9", "6", "2"), ("16", "8", "2")),
        )
        selection = select_class(obs, sorts=("jet_monomial",), grow=True, inner_budget=2)
        assert selection.best is not None
        assert selection.best.tier == "exact"
        assert selection.grammar_complete is False
    finally:
        _restore(saved)


def test_linear_span_product_growth() -> None:
    family = LinearSpanFamily(
        design=((1, 2, 2), (4, 4, 2), (9, 6, 2), (16, 8, 2)),
        term_names=("y", "yp", "ypp"),
        sort="jet_monomial",
    )
    assert family.check(7) is None or family.check(7).ok is False
    grown = family.grown_product_check()
    assert grown is not None
    assert grown.ok is True
    assert grown.payload["grown"] is True


def test_linear_span_nullity_one() -> None:
    family = LinearSpanFamily(
        design=((1, 1), (2, 2)),
        term_names=("a", "b"),
        sort="jet_monomial",
    )
    both = family.check(3)
    assert both is not None
    assert both.ok is True
    assert both.payload["tier"] == "exact"
    single = family.check(1)
    assert single is not None
    assert single.ok is False

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Condition language: hypotheses, constructors, meta-family honesty."""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibias.core.proof import (
    ConditionHypothesis,
    ConditionToken,
    GrammarSpec,
    KindMetaFamily,
    Statement,
    apply_constructor,
    condition_honesty,
    emit_condition,
    run_discovery,
)
from omnibias.core.proof.condition import _SORT_FACTORIES, _reset_condition_sorts_for_tests
from omnibias.core.proof.discovery import Candidate, ExactCheck


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
        return ExactCheck(ok=True, payload={"honesty": condition_honesty(discovered=True)})


def _jet(*names: str) -> ConditionHypothesis:
    return ConditionHypothesis(
        sort="jet_monomial",
        tokens=tuple(ConditionToken("jet_monomial", name) for name in names),
    )


def test_hypothesis_hash_and_dict() -> None:
    left = _jet("y", "yp")
    right = _jet("y", "yp")
    assert left == right
    assert hash(left) == hash(right)
    payload = left.as_dict()
    assert payload["sort"] == "jet_monomial"
    assert payload["pretty"] == "jet_monomial[y, yp]"


def test_emit_condition_statement() -> None:
    statement = emit_condition(_jet("yp"), parent="Riccati identities", parent_status="already_true")
    assert statement.name == "condition_jet_monomial"
    assert "yp" in statement.obligation
    assert statement.parent_status == "already_true"


def test_constructor_depth_cap() -> None:
    grammar = GrammarSpec(
        sorts=("jet_monomial",),
        tokens_by_sort={"jet_monomial": ("y", "yp")},
        constructors=("compose_jets",),
        max_growth_depth=1,
    )
    seed = _jet("y", "yp")
    grown = apply_constructor(seed, "compose_jets", grammar, left="y", right="yp")
    assert grown is not None
    assert "y*yp" in grown.token_names()
    assert apply_constructor(grown, "compose_jets", grammar, left="y", right="y") is None


def test_untyped_constructor_rejected() -> None:
    grammar = GrammarSpec(
        sorts=("jet_monomial",),
        tokens_by_sort={"jet_monomial": ("y",)},
        constructors=("compose_jets",),
    )
    assert apply_constructor(_jet("y"), "llm_emit_checker", grammar) is None


def test_honesty_keys_never_claim_no_condition() -> None:
    honesty = condition_honesty(discovered=True)
    assert honesty["discovered_by_omnibias"] is True
    assert honesty["no_condition_exists_claim"] is False
    assert honesty["unnamed_condition_complete_claim"] is False
    assert honesty["jacobian_conjecture_proof_claim"] is False


def _isolated_sorts() -> dict[str, object]:
    saved = dict(_SORT_FACTORIES)
    _reset_condition_sorts_for_tests()
    return saved


def _restore_sorts(saved: dict[str, object]) -> None:
    _SORT_FACTORIES.clear()
    _SORT_FACTORIES.update(saved)  # type: ignore[arg-type]


def test_meta_missing_sort_forces_incomplete() -> None:
    saved = _isolated_sorts()
    try:
        meta = KindMetaFamily(
            sorts=("jet_monomial", "ore"),
            families={"jet_monomial": _AlwaysMiss()},
            grammar_complete=True,
        )
        assert meta.complete is False
        assert meta.grammar_complete is False
        result = run_discovery(meta.statement, meta, "score_guided", budget=4)
        assert result.status == "BLOCKED"
        assert result.search_incomplete is True
        assert result.detail == "search_incomplete"
    finally:
        _restore_sorts(saved)


def test_meta_complete_empty_grammar_is_blocked_not_parent() -> None:
    saved = _isolated_sorts()
    try:
        meta = KindMetaFamily(
            sorts=("sos_template",),
            families={"sos_template": _AlwaysMiss()},
            grammar_complete=True,
        )
        assert meta.complete is True
        result = run_discovery(meta.statement, meta, "score_guided", budget=4)
        assert result.status == "BLOCKED"
        assert result.search_incomplete is False
        assert result.detail == "no witness in enumerated grammar"
        checked = meta.check("sos_template")
        assert checked is not None
        assert checked.ok is False
        assert checked.payload["honesty"]["no_condition_exists_claim"] is False
    finally:
        _restore_sorts(saved)


def test_meta_hit_on_injected_kind() -> None:
    saved = _isolated_sorts()
    try:
        meta = KindMetaFamily(
            sorts=("jet_monomial", "sos_template"),
            families={
                "jet_monomial": _AlwaysHit(),
                "sos_template": _AlwaysMiss(),
            },
            grammar_complete=True,
            budget=2,
        )
        result = run_discovery(meta.statement, meta, "score_guided", budget=4)
        assert result.status == "PROVED"
        assert result.candidate == "jet_monomial"
        assert result.check is not None
        assert result.check.payload["honesty"]["no_condition_exists_claim"] is False
        assert result.check.payload["honesty"]["discovered_by_omnibias"] is True
    finally:
        _restore_sorts(saved)


def test_meta_score_is_zero_without_gate() -> None:
    saved = _isolated_sorts()
    try:
        meta = KindMetaFamily(
            sorts=("jet_monomial", "sos_template"),
            families={"jet_monomial": _AlwaysHit(), "sos_template": _AlwaysMiss()},
            grammar_complete=True,
        )
        assert meta.score("jet_monomial") == 0
        assert meta.score("sos_template") == 0
        scored = KindMetaFamily(
            sorts=("jet_monomial", "sos_template"),
            families={"jet_monomial": _AlwaysHit(), "sos_template": _AlwaysMiss()},
            sort_scores={"sos_template": 3, "jet_monomial": 1},
            grammar_complete=True,
        )
        assert scored.score("sos_template") == 3
        assert scored.score("jet_monomial") == 1
    finally:
        _restore_sorts(saved)


def test_zero_budget_meta_is_search_incomplete() -> None:
    saved = _isolated_sorts()
    try:
        meta = KindMetaFamily(
            sorts=("jet_monomial",),
            families={"jet_monomial": _AlwaysHit()},
            grammar_complete=True,
        )
        result = run_discovery(meta.statement, meta, "score_guided", budget=0)
        assert result.status == "BLOCKED"
        assert result.search_incomplete is True
        assert result.evaluated == 0
    finally:
        _restore_sorts(saved)

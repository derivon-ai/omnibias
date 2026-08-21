# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite discovery loop: statement → family → proposer → exact checker."""

from __future__ import annotations

from omnibias.core.proof import (
    IntegerIntervalFamily,
    Statement,
    get_proposer,
    run_discovery,
)


def _toy() -> IntegerIntervalFamily:
    return IntegerIntervalFamily(lo=-3, hi=3, target_square=4)


def test_score_guided_finds_square_witness() -> None:
    family = _toy()
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"
    assert result.candidate in (-2, 2)
    assert result.check is not None
    assert result.check.ok is True
    assert result.check.payload["honesty"]["discovered_by_omnibias"] is True
    assert result.search_incomplete is False
    assert result.characterization is not None
    assert result.characterization.unique_in_family is False
    assert result.characterization.note.startswith("uniqueness is span/box-scoped")


def test_zero_budget_is_search_incomplete() -> None:
    family = _toy()
    result = run_discovery(family.statement, family, "score_guided", budget=0)
    assert result.status == "BLOCKED"
    assert result.evaluated == 0
    assert result.detail == "search_incomplete"
    assert result.search_incomplete is True
    assert result.characterization is not None
    assert result.characterization.exhausted is False


def test_incomplete_family_miss_is_blocked() -> None:
    family = IntegerIntervalFamily(lo=-1, hi=1, target_square=4)
    family.complete = False
    family.statement = Statement(
        name="exists_square_incomplete",
        obligation="some integer x with x^2 = 4",
        parent="toy",
        parent_status="open",
    )
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True
    assert result.detail == "search_incomplete"
    assert result.statement.parent_status == "open"


def test_exhausted_existential_miss_is_blocked_not_parent() -> None:
    family = IntegerIntervalFamily(lo=-1, hi=1, target_square=4)
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is False
    assert result.detail == "no witness in enumerated family"
    assert result.characterization is not None
    assert result.characterization.exhausted is True
    assert result.characterization.solution_count == 0


def test_universal_exhausted_miss_is_proved() -> None:
    family = IntegerIntervalFamily(lo=-1, hi=1, target_square=4)
    family.statement = Statement(
        name="no_square_in_box",
        obligation="every integer x in the interval has x^2 not equal to the target",
        parent="toy",
        parent_status="already_true",
        existential=False,
    )
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"
    assert result.search_incomplete is False
    assert result.detail == "no counterexample in complete family"
    assert result.characterization is not None
    assert result.characterization.exhausted is True
    assert result.characterization.solution_count == 0


def test_universal_hit_is_disproved() -> None:
    family = _toy()
    family.statement = Statement(
        name="no_square_false",
        obligation="every integer x in the interval has x^2 not equal to the target",
        parent="toy",
        parent_status="already_false",
        existential=False,
    )
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "DISPROVED"
    assert result.candidate in (-2, 2)
    assert result.detail == "counterexample to universal obligation"


def test_collect_finds_both_squares() -> None:
    family = _toy()
    result = run_discovery(family.statement, family, "score_guided", budget=16, collect=True)
    assert result.status == "PROVED"
    assert set(result.solutions) == {-2, 2}
    assert result.characterization is not None
    assert result.characterization.exhausted is True
    assert result.characterization.solution_count == 2
    assert result.characterization.unique_in_family is False


def test_optional_proposers_also_hit() -> None:
    family = _toy()
    for name in ("coordinate_newton", "onehot_anneal"):
        result = run_discovery(family.statement, family, name, budget=16)
        assert result.status == "PROVED", name
        assert result.candidate in (-2, 2)
        assert result.proposer == name


def test_get_proposer_rejects_unknown() -> None:
    try:
        get_proposer("cubic_newton")
    except ValueError as exc:
        assert "unknown proposer" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_as_dict_carries_honesty() -> None:
    family = _toy()
    payload = run_discovery(family.statement, family, "score_guided", budget=8).as_dict()
    assert payload["kind"] == "finite_discovery"
    assert payload["honesty"]["jacobian_conjecture_proof_claim"] is False
    assert payload["replay_ok"] is True
    assert payload["characterization"]["note"].startswith("uniqueness is span/box-scoped")

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verdict collapse: {0} proves, exclusion disproves, fat zero is blocked."""

from __future__ import annotations

import pytest
from omnibias.core.collapse import (
    VERDICT_SPEC,
    adjudicate_residual,
    are_distinct,
    get_collapse,
    list_collapses,
    reset_collapse_registry,
    search_residuals,
)
from omnibias.core.verified.interval import Interval


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_verdict_is_registered_and_distinct_from_the_founding_three() -> None:
    assert get_collapse("verdict") == VERDICT_SPEC
    assert VERDICT_SPEC.founding is False
    assert "verdict" in {spec.name for spec in list_collapses()}
    for name in ("bias", "temperature", "enclosure"):
        assert are_distinct(VERDICT_SPEC, get_collapse(name)).distinct


def test_singleton_zero_is_proved() -> None:
    verdict = adjudicate_residual(Interval.point(0.0))
    assert verdict.proved
    assert verdict.outcome.collapsed
    assert verdict.outcome.honesty["float_residual_is_proof"] is False
    assert verdict.outcome.honesty["theorem_prover_verified"] is False
    assert verdict.outcome.honesty["founding_bias_collapse"] is False
    assert verdict.outcome.honesty["verdict_collapse"] is True


def test_excluded_zero_is_disproved() -> None:
    verdict = adjudicate_residual(Interval.point(1.0))
    assert verdict.disproved
    assert verdict.outcome.excluded


def test_fat_interval_containing_zero_is_blocked_not_false() -> None:
    verdict = adjudicate_residual(Interval(-0.1, 0.1))
    assert verdict.blocked
    assert verdict.outcome.inconclusive
    assert "not false" in verdict.detail


def test_float_residual_is_refused() -> None:
    with pytest.raises(TypeError, match="float residual is not a certificate"):
        adjudicate_residual(0.0)
    with pytest.raises(TypeError, match="float residual is not a certificate"):
        search_residuals([0.0])


def test_existential_search_hits_then_stops() -> None:
    verdict = search_residuals(
        [Interval.point(2.0), Interval.point(0.0), Interval.point(3.0)],
        existential=True,
    )
    assert verdict.proved
    assert verdict.evaluated == 2
    assert verdict.outcome.residual == Interval.point(0.0)


def test_complete_existential_miss_is_blocked_not_parent_false() -> None:
    verdict = search_residuals(
        [Interval.point(1.0), Interval.point(2.0)],
        existential=True,
        complete=True,
    )
    assert verdict.blocked
    assert verdict.complete is True
    assert "not a parent claim" in verdict.detail


def test_complete_universal_all_zero_is_proved() -> None:
    verdict = search_residuals(
        [Interval.point(0.0), Interval.point(0.0)],
        existential=False,
        complete=True,
    )
    assert verdict.proved
    assert verdict.complete is True


def test_universal_counterexample_is_disproved() -> None:
    verdict = search_residuals(
        [Interval.point(0.0), Interval.point(4.0)],
        existential=False,
        complete=True,
    )
    assert verdict.disproved


def test_incomplete_or_inconclusive_walk_is_blocked() -> None:
    incomplete = search_residuals(
        [Interval.point(1.0)],
        existential=True,
        complete=False,
    )
    assert incomplete.blocked
    fat = search_residuals(
        [Interval(-1.0, 1.0), Interval.point(1.0)],
        existential=True,
        complete=True,
    )
    assert fat.blocked
    empty = search_residuals([], complete=True)
    assert empty.blocked
    assert empty.evaluated == 0

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the data-driven refinement harness (omnibias.difference.validation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.difference.validation import (
    BaselineComparison,
    Finding,
    FindingsLedger,
    baseline_compare,
    enclosure_soundness,
    grid_and_random_points,
    high_precision_derivative,
    seed_sweep,
)


def test_finding_rejects_bad_severity() -> None:
    with pytest.raises(ValueError):
        Finding(workstream="w", severity="nope", summary="s")


def test_ledger_add_counts_and_json(tmp_path: Path) -> None:
    ledger = FindingsLedger("probe")
    ledger.add("W1", "gap", "missing constant", detail="no PI_IV", where="transcend")
    ledger.add("W1", "bug", "escape", value=1.5)
    assert len(ledger) == 2
    counts = ledger.counts()
    assert counts["gap"] == 1
    assert counts["bug"] == 1
    assert counts["info"] == 0

    path = ledger.write(str(tmp_path / "probe.json"))
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["name"] == "probe"
    assert payload["counts"]["gap"] == 1
    assert len(payload["findings"]) == 2
    assert "missing constant" in ledger.summary()


def test_grid_and_random_points_count_and_determinism() -> None:
    box = Interval(-1.0, 2.0)
    pts = grid_and_random_points(box, grid=10, random_samples=5, seed=0)
    assert len(pts) == 15
    assert all(box.lo <= p <= box.hi for p in pts)
    assert pts == grid_and_random_points(box, grid=10, random_samples=5, seed=0)


def test_enclosure_soundness_detects_containment_and_escape() -> None:
    box = Interval(0.0, 1.0)
    ok = enclosure_soundness(Interval(-0.1, 1.1), lambda x: x, box, grid=11, random_samples=10)
    assert ok.sound
    assert not ok.failures
    assert ok.max_escape == 0.0
    assert ok.n_points == 21

    bad = enclosure_soundness(Interval(0.0, 0.5), lambda x: x, box, grid=11, random_samples=0)
    assert not bad.sound
    assert bad.failures
    assert bad.max_escape > 0.0


def test_baseline_compare_lower_is_better() -> None:
    win = baseline_compare("fd", candidate=1e-12, baseline=1e-3)
    assert isinstance(win, BaselineComparison)
    assert win.wins
    assert win.ratio < 1.0
    lose = baseline_compare("fd", candidate=1e-1, baseline=1e-3)
    assert not lose.wins


def test_high_precision_derivative_matches_known() -> None:
    mpmath = pytest.importorskip("mpmath")
    # d^3/dz^3 tanh(0) = -2.
    got = high_precision_derivative(mpmath.tanh, 0.0, 3)
    assert abs(got - (-2.0)) < 1e-9


def test_seed_sweep_aggregates() -> None:
    stats = seed_sweep(float, range(5))
    assert stats["min"] == 0.0
    assert stats["max"] == 4.0
    assert stats["mean"] == 2.0
    assert stats["n"] == 5.0

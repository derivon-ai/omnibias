# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Beta-annealing scheduler (pure-Python; no backend required)."""

from __future__ import annotations

import math

import pytest
from omnibias.binary import BetaAnnealScheduler
from omnibias.binary.schedule import BetaAnnealScheduler as DirectImport


def test_top_level_and_module_export_are_same_object() -> None:
    assert BetaAnnealScheduler is DirectImport


def test_linear_endpoints_and_midpoint() -> None:
    s = BetaAnnealScheduler(beta_start=1.0, beta_end=11.0, num_steps=10, schedule="linear")
    assert s.value(0) == pytest.approx(1.0)
    assert s.value(10) == pytest.approx(11.0)
    assert s.value(5) == pytest.approx(6.0)


def test_exp_is_geometric() -> None:
    s = BetaAnnealScheduler(beta_start=1.0, beta_end=100.0, num_steps=10, schedule="exp")
    assert s.value(0) == pytest.approx(1.0)
    assert s.value(10) == pytest.approx(100.0)
    assert s.value(5) == pytest.approx(math.sqrt(1.0 * 100.0))  # geometric mean


def test_cosine_endpoints_and_monotone() -> None:
    s = BetaAnnealScheduler(beta_start=2.0, beta_end=20.0, num_steps=20, schedule="cosine")
    assert s.value(0) == pytest.approx(2.0)
    assert s.value(20) == pytest.approx(20.0)
    vals = [s.value(k) for k in range(21)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:], strict=False))


def test_value_is_clamped_outside_range() -> None:
    s = BetaAnnealScheduler(beta_start=1.0, beta_end=5.0, num_steps=4)
    assert s.value(-100) == pytest.approx(1.0)
    assert s.value(10_000) == pytest.approx(5.0)


def test_step_advances_counter() -> None:
    s = BetaAnnealScheduler(beta_start=1.0, beta_end=5.0, num_steps=4)
    seen = [s.step() for _ in range(5)]
    assert seen[0] == pytest.approx(1.0)
    assert seen[-1] == pytest.approx(5.0)  # clamped at the end
    assert s.current_step == 5
    s.reset()
    assert s.current_step == 0
    assert s.step() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"beta_start": 0.0, "beta_end": 1.0, "num_steps": 4}, "must be > 0"),
        ({"beta_start": 1.0, "beta_end": 1.0, "num_steps": 0}, "num_steps"),
        ({"beta_start": 1.0, "beta_end": 1.0, "num_steps": 4, "schedule": "nope"}, "unknown schedule"),
    ],
)
def test_invalid_args_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BetaAnnealScheduler(**kwargs)

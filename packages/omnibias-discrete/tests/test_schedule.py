# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The annealing schedule: geometric betas and input validation."""

from __future__ import annotations

import pytest
from omnibias.discrete import AnnealSchedule


def test_betas_are_geometric_and_length_matches_stages() -> None:
    sched = AnnealSchedule(beta0=0.5, beta_growth=2.0, stages=4)
    betas = sched.betas()
    assert betas == [0.5, 1.0, 2.0, 4.0]
    assert len(betas) == sched.stages


def test_fast_schedule_is_lighter() -> None:
    assert AnnealSchedule.fast().stages < AnnealSchedule().stages


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta0": 0.0},
        {"beta0": -1.0},
        {"beta_growth": 0.5},
        {"stages": 0},
        {"steps": 0},
        {"step_safety": 0.0},
        {"step_safety": 1.5},
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AnnealSchedule(**kwargs)

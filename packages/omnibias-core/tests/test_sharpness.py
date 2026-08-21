# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sharpness schedule algebra (theory 08-06)."""

from __future__ import annotations

import math

import pytest
from omnibias.core.sharpness import (
    SharpnessSchedule,
    make_report,
    scheduled_value,
)


def test_cubic_sigma_is_c_times_max_ell() -> None:
    schedule = SharpnessSchedule(c=2.0, ell_min=1e-6, target="cubic_sigma")
    assert scheduled_value(10.0, schedule) == pytest.approx(20.0)
    assert scheduled_value(-3.0, schedule) == pytest.approx(2.0e-6)


def test_lr_is_c_over_sharpness() -> None:
    schedule = SharpnessSchedule(c=1.0, ell_min=1e-6, target="lr")
    assert scheduled_value(1e4, schedule) == pytest.approx(1e-4)


def test_report_records_named_knobs() -> None:
    schedule = SharpnessSchedule(n_lanczos=4, c=1e-3, ell_min=1e-6)
    report = make_report(1e4, schedule)
    assert report.n_lanczos == 4
    assert report.c == pytest.approx(1e-3)
    assert report.ell_min == pytest.approx(1e-6)
    assert report.ell_k == pytest.approx(1e4)
    assert report.scheduled == pytest.approx(10.0)
    assert report.target == "cubic_sigma"


def test_zero_ell_min_and_zero_ell_gives_zero_sigma() -> None:
    schedule = SharpnessSchedule(c=1.0, ell_min=0.0, target="cubic_sigma")
    assert scheduled_value(0.0, schedule) == 0.0
    lr = SharpnessSchedule(c=1.0, ell_min=0.0, target="lr")
    assert not math.isfinite(scheduled_value(0.0, lr))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_lanczos": 0},
        {"c": 0.0},
        {"c": -1.0},
        {"ell_min": -1e-6},
        {"target": "damping"},
    ],
)
def test_invalid_schedule_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SharpnessSchedule(**kwargs)  # type: ignore[arg-type]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Product API for inverse imaging (theory 05-01)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.verified.interval import Interval
from omnibias.pinn.inverse import (
    GuaranteeKindError,
    InverseRegularizerError,
    honesty_payload,
    identifiability,
    locate_interface,
    merge_guarantees,
    piecewise_jump_field,
    place_sensors,
    worked_example,
)


def test_worked_example_localizes_the_kink() -> None:
    ex = worked_example()
    assert abs(float(ex["tau_hat"]) - 0.37) < 1e-3
    assert ex["unique"] is True
    assert ex["contains_truth"] is True
    assert ex["channel"] == 3


def test_merge_guarantees_raises() -> None:
    with pytest.raises(GuaranteeKindError, match="must not merge"):
        merge_guarantees(Interval(0.36, 0.38), {"kind": "conformal"})


def test_regularizer_is_required() -> None:
    x = np.linspace(0.0, 1.0, 81)
    y = piecewise_jump_field(x, jump_order=1, jump=2.0, tau=0.4)
    with pytest.raises(InverseRegularizerError):
        locate_interface(x, y, order=3, alpha=25.0, regularizer="")


def test_identifiability_names_a_far_sensor_gap() -> None:
    far = identifiability(
        "location",
        {"tau": 0.37, "alpha": 25.0},
        np.array([0.0, 0.02, 0.04]),
        noise_std=0.2,
        floor=1.0,
    )
    assert far.unidentifiable == ("tau",)
    near = identifiability(
        "location",
        {"tau": 0.37, "alpha": 25.0},
        np.linspace(0.2, 0.6, 21),
        noise_std=0.05,
        floor=1e-3,
    )
    assert near.unidentifiable == ()


def test_place_sensors_beats_uniform() -> None:
    plan = place_sensors(
        "location",
        {"tau": 0.37, "alpha": 25.0},
        np.linspace(0.0, 1.0, 21),
        budget=4,
    )
    assert plan.volume_ratio >= 2.0
    assert plan.guarantee == pytest.approx(1.0 - 1.0 / np.e)


def test_honesty_flags() -> None:
    h = honesty_payload()
    assert h["ill_posedness_removed"] is False
    assert h["conformal_merged"] is False
    assert h["temperature_collapse"] is False
    assert h["not_pde_coefficient_inverse"] is True

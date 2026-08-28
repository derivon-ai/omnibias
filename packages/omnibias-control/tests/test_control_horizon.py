# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified horizon selection from an enclosed discrete closed-loop monodromy (10-03)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.control.horizon import (
    certified_horizon,
    enclose_jacobian_ball,
    honesty_payload,
    horizon_skill,
    monodromy_product,
    worked_example,
)


def test_honesty_payload_all_false():
    assert not any(honesty_payload().values())


def test_enclose_jacobian_ball_basic():
    ball = enclose_jacobian_ball(np.array([[1.0, 0.0], [0.0, 1.0]]), radius=0.1)
    assert ball[0][0].lo == pytest.approx(0.9)
    assert ball[0][0].hi == pytest.approx(1.1)


def test_enclose_jacobian_ball_rejects_negative_radius():
    with pytest.raises(ValueError):
        enclose_jacobian_ball(np.eye(2), radius=-1.0)


def test_enclose_jacobian_ball_rejects_non_square():
    with pytest.raises(ValueError):
        enclose_jacobian_ball(np.ones((2, 3)), radius=0.1)


def test_monodromy_product_identity_chain():
    balls = [enclose_jacobian_ball(np.eye(2), radius=0.0) for _ in range(3)]
    product = monodromy_product(balls)
    for i in range(2):
        for j in range(2):
            expected = 1.0 if i == j else 0.0
            assert product[i][j].lo == pytest.approx(expected)
            assert product[i][j].hi == pytest.approx(expected)


def test_monodromy_product_rejects_empty():
    with pytest.raises(ValueError):
        monodromy_product([])


def test_certified_horizon_finds_finite_window_for_contracting_system():
    m = np.diag([0.5, 0.6])
    result = certified_horizon([m] * 20, radius=0.0, tol=0.05)
    assert result.certified
    assert result.horizon is not None
    assert result.spectral_radius_bound_hi <= 0.05


def test_certified_horizon_reports_uncertified_for_expansive_system():
    m = np.diag([1.5, 1.5])
    result = certified_horizon([m] * 5, radius=0.0, tol=0.05, max_horizon=5)
    assert not result.certified
    assert result.horizon is None
    assert len(result.trace) == 5


def test_certified_horizon_rejects_bad_tol():
    with pytest.raises(ValueError):
        certified_horizon([np.eye(2)], radius=0.0, tol=0.0)


def test_certified_horizon_rejects_empty_sequence():
    with pytest.raises(ValueError):
        certified_horizon([], radius=0.0)


def test_worked_example_gate_g5():
    result = worked_example()
    assert result["g5_earned"] is True
    assert result["certified"] is True


def test_horizon_skill_gate_g5():
    result = horizon_skill(n=40, seed=7)
    assert result["g5_earned"] is True
    assert result["coverage"] == 1.0
    assert math.isfinite(result["median_horizon"])

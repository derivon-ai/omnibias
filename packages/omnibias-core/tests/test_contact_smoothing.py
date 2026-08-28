# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for certified contact-force smoothing (theory 10-04)."""

from __future__ import annotations

import math

import pytest
from omnibias.core.contact_smoothing import (
    contact_force_hard,
    contact_force_smooth,
    contact_force_tower,
    contact_skill,
    contact_smoothing_bias_bound,
    honesty_payload,
    worked_example,
)


def test_honesty_payload_all_false() -> None:
    payload = honesty_payload()
    assert not any(payload.values())


def test_contact_force_smooth_matches_sigmoid() -> None:
    from omnibias.core.ftc import sigmoid

    assert contact_force_smooth(0.3, beta=2.0, f_max=1.5) == pytest.approx(
        1.5 * sigmoid(-2.0 * 0.3)
    )


def test_contact_force_smooth_negative_beta_raises() -> None:
    with pytest.raises(ValueError):
        contact_force_smooth(0.0, beta=-1.0)


def test_contact_force_hard_is_heaviside() -> None:
    assert contact_force_hard(-0.1, f_max=2.0) == 2.0
    assert contact_force_hard(0.0, f_max=2.0) == 2.0
    assert contact_force_hard(0.1, f_max=2.0) == 0.0


def test_contact_force_tower_order_zero_is_value() -> None:
    tower = contact_force_tower(0.4, beta=3.0, order=0, f_max=2.0)
    assert len(tower) == 1
    assert tower[0] == pytest.approx(contact_force_smooth(0.4, beta=3.0, f_max=2.0))


def test_contact_force_tower_matches_finite_difference() -> None:
    z0, beta, f_max = 0.6, 5.0, 2.5
    tower = contact_force_tower(z0, beta, order=3, f_max=f_max)
    h = 1e-4
    fd1 = (
        contact_force_smooth(z0 + h, beta, f_max) - contact_force_smooth(z0 - h, beta, f_max)
    ) / (2.0 * h)
    fd2 = (
        contact_force_smooth(z0 + h, beta, f_max)
        - 2.0 * contact_force_smooth(z0, beta, f_max)
        + contact_force_smooth(z0 - h, beta, f_max)
    ) / (h * h)
    assert tower[1] == pytest.approx(fd1, abs=1e-5)
    assert tower[2] == pytest.approx(fd2, abs=1e-2)


def test_bias_bound_certified_and_covers_grid() -> None:
    report = contact_smoothing_bias_bound(z_min=0.5, beta=6.0, f_max=1.0)
    assert report.certified
    for z in (0.5, 0.7, 1.0, 2.0, -0.5, -0.7, -1.0, -2.0):
        actual = abs(contact_force_smooth(z, 6.0, 1.0) - contact_force_hard(z, 1.0))
        assert actual <= report.bound.hi


def test_bias_bound_inconclusive_at_zero_z_min() -> None:
    report = contact_smoothing_bias_bound(z_min=0.0, beta=1.0)
    assert report.certified is False
    assert "Inconclusive" in report.reason


def test_bias_bound_negative_z_min_inconclusive() -> None:
    report = contact_smoothing_bias_bound(z_min=-0.1, beta=1.0)
    assert not report.certified


def test_bias_bound_negative_beta_raises() -> None:
    with pytest.raises(ValueError):
        contact_smoothing_bias_bound(z_min=0.5, beta=-1.0)


def test_bias_bound_tighter_for_larger_beta() -> None:
    small = contact_smoothing_bias_bound(z_min=0.5, beta=1.0)
    large = contact_smoothing_bias_bound(z_min=0.5, beta=20.0)
    assert large.bound.hi < small.bound.hi


def test_worked_example_g1() -> None:
    result = worked_example()
    assert result["grid_ok"] is True
    assert result["certified"] is True
    assert result["first_derivative_fd_residual"] < 1e-6


def test_contact_skill_g2() -> None:
    result = contact_skill(n=200, seed=3)
    assert result["coverage"] == 1.0
    assert result["inconclusive_at_zero"] is True
    assert result["g2_earned"] is True
    assert math.isfinite(result["median_width"])

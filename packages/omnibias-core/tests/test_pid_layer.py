# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-29: plant PID layer with closed-form I and D."""

from __future__ import annotations

import math

import pytest
from omnibias.core.pid_layer import (
    DISCLAIMER,
    PlantPIDConfig,
    activation_prime,
    error_integral,
    honesty_payload,
    plant_pid,
    plant_pid_skill,
    worked_example,
)


def test_g1_ftc_and_derivative() -> None:
    ex = worked_example()
    assert float(ex["ftc_residual"]) < 1e-8
    assert float(ex["d_residual"]) < 1e-15
    assert ex["measurement"] == pytest.approx(0.5)
    assert ex["derivative"] == pytest.approx(-0.25)


def test_g2_skill() -> None:
    report = plant_pid_skill()
    assert report["g2_earned"] is True
    assert report["unknown_raised"] is True
    assert float(report["median_ftc_residual"]) < 1e-8  # type: ignore[arg-type]


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["trainer_claimed"] is False
    assert payload["cruise_sota_claimed"] is False
    assert payload["discrete_sum_i"] is False
    assert payload["skips_chain_rule"] is False
    assert "not 08-10" in DISCLAIMER


def test_zero_window_has_zero_integral() -> None:
    cfg = PlantPIDConfig(kp=1.0, ki=1.0, kd=0.0, setpoint=0.5, t0=0.25)
    report = plant_pid(0.25, config=cfg)
    assert report.integral == 0.0
    assert report.proportional == pytest.approx(0.5 - report.measurement)
    assert report.control == pytest.approx(cfg.kp * report.error)


def test_zero_alpha_integral_is_rectangle() -> None:
    cfg = PlantPIDConfig(
        kp=0.0, ki=1.0, kd=1.0, setpoint=0.25, alpha=0.0, beta=0.0, t0=0.0
    )
    report = plant_pid(2.0, config=cfg)
    # y = sigmoid(0) = 0.5, e = -0.25, I = -0.5, D = 0
    assert report.measurement == pytest.approx(0.5)
    assert report.integral == pytest.approx(-0.5)
    assert report.derivative == pytest.approx(0.0)


def test_tanh_ftc_matches_log_cosh() -> None:
    cfg = PlantPIDConfig(
        kp=0.0, ki=1.0, kd=0.0, setpoint=0.0, alpha=2.0, beta=0.1, t0=-0.4, family="tanh"
    )
    report = plant_pid(0.3, config=cfg)
    direct = error_integral(
        0.3, -0.4, setpoint=0.0, alpha=2.0, beta=0.1, family="tanh"
    )
    assert report.integral == pytest.approx(direct)
    z = 2.0 * 0.3 + 0.1
    assert report.derivative == pytest.approx(-2.0 * activation_prime(z, "tanh"))


def test_riccati_not_finite_difference() -> None:
    z = 0.3
    closed = activation_prime(z, "sigmoid")
    h = 1e-6
    from omnibias.core.pid_layer import activation_value

    fd = (activation_value(z + h, "sigmoid") - activation_value(z - h, "sigmoid")) / (
        2.0 * h
    )
    assert closed == pytest.approx(fd, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"family": "relu"}, "unknown family"),
        ({"kp": math.inf}, "kp"),
        ({"alpha": math.nan}, "alpha"),
        ({"t0": math.inf}, "t0"),
    ],
)
def test_rejects_illegal_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plant_pid(0.0, config=PlantPIDConfig(**kwargs))  # type: ignore[arg-type]


def test_rejects_non_finite_time() -> None:
    with pytest.raises(ValueError, match="t"):
        plant_pid(math.inf)

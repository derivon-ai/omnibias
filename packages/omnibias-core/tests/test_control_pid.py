# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 08-10: jet-PID optimizer on a directional restriction."""

from __future__ import annotations

import math

import pytest
from omnibias.core.control_pid import (
    DISCLAIMER,
    JetPIDConfig,
    honesty_payload,
    model_definite_integral,
    model_ftc_increment,
    pid_skill,
    pid_step_from_derivatives,
    poly_antiderivative,
    worked_example,
)
from omnibias.core.line_search import poly_eval, taylor_coeffs_from_derivatives


def test_g1_quadratic_bowl() -> None:
    ex = worked_example()
    assert ex["abs_step_err"] < 1e-15
    assert ex["proportional"] == pytest.approx(-2.0)
    assert ex["damped_is_newton"] is True
    assert ex["overshoot_step"] == pytest.approx(2.0)


def test_g2_skill() -> None:
    report = pid_skill()
    assert report["g2_earned"] is True
    assert report["kd_raised"] is True
    assert report["all_finite"] is True
    assert float(report["median_err"]) < 1e-12  # type: ignore[arg-type]


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["global_min_claimed"] is False
    assert payload["plant_pid_claimed"] is False
    assert payload["lqr_claimed"] is False
    assert payload["mpc_claimed"] is False
    assert payload["algebraic_riccati_claimed"] is False
    assert payload["stretch_claim"] is False
    assert payload["skips_chain_rule"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not a plant PID" in DISCLAIMER
    assert "not a global min" in DISCLAIMER


def test_ftc_matches_antiderivative_of_error() -> None:
    # p(s) = 1 - 2s + s^2, e = p' = -2 + 2s, ∫_0^h e = p(h)-p(0)
    coeffs = taylor_coeffs_from_derivatives((1.0, -2.0, 2.0))
    h = 0.4
    ftc = model_ftc_increment(coeffs, 0.0, h)
    assert ftc == pytest.approx(poly_eval(coeffs, h) - poly_eval(coeffs, 0.0))
    # ∫ p also has a closed antiderivative
    anti = poly_antiderivative(coeffs)
    assert model_definite_integral(coeffs, 0.0, h) == pytest.approx(
        poly_eval(anti, h) - poly_eval(anti, 0.0)
    )


def test_model_window_integral_is_exact_ftc() -> None:
    coeffs = taylor_coeffs_from_derivatives((1.0, -2.0, 2.0))
    window = 0.3
    report = pid_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetPIDConfig(kp=0.0, ki=1.0, kd=0.0, window=window, s_max=2.0),
    )
    assert report.integral == pytest.approx(model_ftc_increment(coeffs, 0.0, window))
    assert report.control == pytest.approx(report.integral)


def test_running_integral_accumulates_applied_ftc() -> None:
    derivs = (1.0, -2.0, 2.0)
    first = pid_step_from_derivatives(
        derivs,
        config=JetPIDConfig(
            kp=0.5, ki=1.0, kd=0.0, integral_mode="running", s_max=2.0
        ),
        running_integral=0.0,
    )
    coeffs = taylor_coeffs_from_derivatives(derivs)
    expected = model_ftc_increment(coeffs, 0.0, first.step)
    assert first.running_integral == pytest.approx(expected)
    second = pid_step_from_derivatives(
        derivs,
        config=JetPIDConfig(
            kp=0.0, ki=1.0, kd=0.0, integral_mode="running", s_max=2.0
        ),
        running_integral=first.running_integral,
    )
    assert second.integral == pytest.approx(first.running_integral)


def test_s_max_clips() -> None:
    report = pid_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetPIDConfig(kp=1.0, ki=0.0, kd=0.0, s_max=0.25),
    )
    assert report.step == pytest.approx(0.25)
    assert report.refused is False


def test_uncertified_integral_refuses() -> None:
    # exp(s) jet order 2 at 0; remainder on [0,1] is e/6 > 0.1
    report = pid_step_from_derivatives(
        (1.0, 1.0, 1.0),
        config=JetPIDConfig(
            kp=0.0,
            ki=1.0,
            kd=0.0,
            window=1.0,
            s_max=1.0,
            next_derivative_bound=math.e,
            refuse_uncertified_integral=True,
            remainder_tol=0.1,
        ),
    )
    assert report.refused is True
    assert report.reason == "uncertified_integral"
    assert report.step == 0.0
    assert report.remainder_hi is not None
    assert report.remainder_hi > 0.1


def test_certified_zero_remainder_accepts() -> None:
    report = pid_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetPIDConfig(
            kp=0.5,
            ki=1.0,
            kd=0.0,
            window=0.5,
            s_max=2.0,
            next_derivative_bound=0.0,
            refuse_uncertified_integral=True,
            remainder_tol=1e-12,
        ),
    )
    assert report.refused is False
    assert report.reason == "ok"
    assert report.remainder_hi == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("derivs", "kwargs", "match"),
    [
        ((), {}, "non-empty"),
        ((1.0,), {}, "phi'"),
        ((1.0, -2.0), {"kd": 1.0}, "three derivatives"),
        ((1.0, float("nan")), {}, "finite"),
        ((1.0, -2.0), {"s_max": 0.0}, "s_max"),
        ((1.0, -2.0), {"kp": float("inf")}, "kp"),
        (
            (1.0, -2.0),
            {"refuse_uncertified_integral": True},
            "next_derivative_bound",
        ),
    ],
)
def test_rejects_illegal_inputs(
    derivs: tuple[float, ...], kwargs: dict[str, object], match: str
) -> None:
    config = JetPIDConfig(**kwargs) if kwargs else None  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        pid_step_from_derivatives(derivs, config=config)

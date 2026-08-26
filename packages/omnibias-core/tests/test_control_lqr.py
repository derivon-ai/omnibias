# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 08-11: finite-horizon LQR on a directional jet."""

from __future__ import annotations

import pytest
from omnibias.core.control_lqr import (
    DISCLAIMER,
    JetLQRConfig,
    honesty_payload,
    lqr_skill,
    lqr_step_from_derivatives,
    scalar_finite_horizon_lqr,
    worked_example,
)


def test_g1_newton_recovery() -> None:
    ex = worked_example()
    assert ex["newton_recovered"] is True
    assert ex["abs_step_err"] < 1e-15
    assert ex["damped_smaller"] is True
    assert ex["damped_step"] == pytest.approx(0.5)


def test_g2_skill() -> None:
    report = lqr_skill()
    assert report["g2_earned"] is True
    assert report["singular_raised"] is True
    assert float(report["median_err"]) < 1e-12  # type: ignore[arg-type]


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["global_min_claimed"] is False
    assert payload["plant_lqr_claimed"] is False
    assert payload["algebraic_riccati_claimed"] is False
    assert payload["activation_riccati_as_gain"] is False
    assert payload["mpc_claimed"] is False
    assert payload["skips_chain_rule"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not DARE" in DISCLAIMER
    assert "not activation Riccati" in DISCLAIMER


def test_newton_identity_on_bowl() -> None:
    gain, _p0 = scalar_finite_horizon_lqr(1.0, 2.0, 0.0, 0.0, 1.0, 1)
    assert gain == pytest.approx(0.5)
    report = lqr_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1, s_max=4.0),
    )
    assert report.step == pytest.approx(1.0)
    assert report.gain == pytest.approx(gain)


def test_positive_r_shrinks_newton() -> None:
    newton = lqr_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1),
    )
    damped = lqr_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetLQRConfig(q=0.0, r=4.0, qf=1.0, horizon=1),
    )
    # K = H Qf / (R + H^2 Qf) = 2/8 = 0.25; u = -K e = 0.5
    assert damped.step == pytest.approx(0.5)
    assert abs(damped.step) < abs(newton.step)


def test_s_max_clips() -> None:
    report = lqr_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1, s_max=0.25),
    )
    assert report.step == pytest.approx(0.25)


def test_zero_curvature_with_effort_is_zero_step() -> None:
    report = lqr_step_from_derivatives(
        (1.0, -1.0, 0.0),
        config=JetLQRConfig(q=1.0, r=1.0, qf=1.0, horizon=2),
    )
    assert report.step == pytest.approx(0.0)
    assert report.gain == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("derivs", "kwargs", "match"),
    [
        ((), {}, "non-empty"),
        ((1.0, -2.0), {}, "three derivatives"),
        ((1.0, -2.0, 2.0), {"horizon": 0}, "horizon"),
        ((1.0, -2.0, 2.0), {"r": -1.0}, ">= 0"),
        ((1.0, -2.0, 2.0), {"s_max": 0.0}, "s_max"),
        ((1.0, float("nan"), 2.0), {}, "finite"),
        ((1.0, -1.0, 0.0), {"r": 0.0, "qf": 1.0}, "singular"),
    ],
)
def test_rejects_illegal_inputs(
    derivs: tuple[float, ...], kwargs: dict[str, object], match: str
) -> None:
    config = JetLQRConfig(**kwargs) if kwargs else None  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        lqr_step_from_derivatives(derivs, config=config)

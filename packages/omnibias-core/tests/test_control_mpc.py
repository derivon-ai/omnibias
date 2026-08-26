# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 08-12: receding-horizon jet MPC on a directional restriction."""

from __future__ import annotations

import math

import pytest
from omnibias.core.control_lqr import JetLQRConfig, lqr_step_from_derivatives
from omnibias.core.control_mpc import (
    DISCLAIMER,
    JetMPCConfig,
    honesty_payload,
    mpc_skill,
    mpc_step_from_derivatives,
    worked_example,
)


def test_g1_newton_and_box() -> None:
    ex = worked_example()
    assert ex["newton_recovered"] is True
    assert ex["abs_step_err"] < 1e-15
    assert ex["boxed"] is True
    assert ex["boxed_step"] == pytest.approx(0.25)
    assert ex["receding"] is True


def test_g2_skill() -> None:
    report = mpc_skill()
    assert report["g2_earned"] is True
    assert report["boxed"] is True
    assert report["refused"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["global_min_claimed"] is False
    assert payload["plant_mpc_claimed"] is False
    assert payload["general_qp_claimed"] is False
    assert payload["activation_riccati_as_model"] is False
    assert payload["skips_chain_rule"] is False
    assert "not plant MPC" in DISCLAIMER


def test_unconstrained_matches_lqr() -> None:
    derivs = (1.0, -2.0, 2.0)
    for horizon in (1, 3):
        lqr = lqr_step_from_derivatives(
            derivs,
            config=JetLQRConfig(q=0.0, r=0.25, qf=1.0, horizon=horizon, s_max=4.0),
        )
        mpc = mpc_step_from_derivatives(
            derivs,
            config=JetMPCConfig(q=0.0, r=0.25, qf=1.0, horizon=horizon, u_max=4.0),
        )
        assert mpc.step == pytest.approx(lqr.control)
        assert mpc.unconstrained_step == pytest.approx(lqr.control)
        assert mpc.gain == pytest.approx(lqr.gain)
        assert mpc.receding is True


def test_remainder_refuse() -> None:
    report = mpc_step_from_derivatives(
        (1.0, 1.0, 1.0),
        config=JetMPCConfig(
            q=0.0,
            r=0.0,
            qf=1.0,
            horizon=1,
            u_max=1.0,
            next_derivative_bound=math.e,
            refuse_uncertified=True,
            remainder_tol=0.1,
        ),
    )
    assert report.refused is True
    assert report.reason == "uncertified_model"
    assert report.step == 0.0
    assert report.receding is True


def test_trust_radius_clips() -> None:
    report = mpc_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetMPCConfig(
            q=0.0,
            r=0.0,
            qf=1.0,
            horizon=1,
            u_max=4.0,
            next_derivative_bound=24.0,
            trust_atol=1e-3,
        ),
    )
    assert report.trust_radius is not None
    assert report.trust_radius < 1.0
    assert abs(report.step) <= report.trust_radius + 1e-15
    assert report.reason == "boxed"


@pytest.mark.parametrize(
    ("derivs", "kwargs", "match"),
    [
        ((), {}, "non-empty"),
        ((1.0, -2.0), {}, "three derivatives"),
        ((1.0, -2.0, 2.0), {"horizon": 0}, "horizon"),
        ((1.0, -2.0, 2.0), {"u_max": 0.0}, "u_max"),
        ((1.0, -2.0, 2.0), {"refuse_uncertified": True}, "next_derivative_bound"),
        ((1.0, -2.0, 2.0), {"trust_atol": 1e-3}, "next_derivative_bound"),
    ],
)
def test_rejects_illegal_inputs(
    derivs: tuple[float, ...], kwargs: dict[str, object], match: str
) -> None:
    config = JetMPCConfig(**kwargs) if kwargs else None  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        mpc_step_from_derivatives(derivs, config=config)

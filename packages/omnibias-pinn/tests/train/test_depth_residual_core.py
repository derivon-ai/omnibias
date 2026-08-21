# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free depth-causal residual algebra (theory 08-05)."""

from __future__ import annotations

import math

import pytest
from omnibias.pinn.train import (
    DepthResidualConfig,
    DepthResidualForbidden,
    DepthResidualReport,
    apply_hard_bc_tower,
    hard_bc_mask_tower,
    honesty_payload,
    layer_is_unlocked,
    leibniz_product_tower,
    reject_depth_residual_flood,
)


def test_config_rejects_bad_budget() -> None:
    with pytest.raises(ValueError, match="n_directions"):
        DepthResidualConfig(n_directions=0)
    with pytest.raises(ValueError, match="steps_per_layer"):
        DepthResidualConfig(steps_per_layer=0)
    with pytest.raises(ValueError, match="hard_bc"):
        DepthResidualConfig(hard_bc="periodic")  # type: ignore[arg-type]


def test_leibniz_unit_interval_constant_field() -> None:
    # N = 1, s = x(1-x) => u = x - x^2, u'' = -2
    x = 0.25
    n_tower = (1.0, 0.0, 0.0)
    u = apply_hard_bc_tower(n_tower, x, "unit_interval")
    s = hard_bc_mask_tower(x, "unit_interval", 2)
    assert u[0] == pytest.approx(s[0])
    assert u[2] == pytest.approx(-2.0)


def test_leibniz_matches_product_rule() -> None:
    left = (0.5, -1.0, 2.0)
    right = (1.5, 0.25, -0.5)
    got = leibniz_product_tower(left, right)
    expect = (
        left[0] * right[0],
        left[1] * right[0] + left[0] * right[1],
        left[2] * right[0] + 2.0 * left[1] * right[1] + left[0] * right[2],
    )
    assert got == pytest.approx(expect)


def test_unlock_first_layer_always() -> None:
    assert layer_is_unlocked(0, None, None, 0.1) is True
    assert layer_is_unlocked(0, None, None, None) is True


def test_unlock_later_layer_ratio() -> None:
    assert layer_is_unlocked(1, 0.4, 1.0, 0.5) is True
    assert layer_is_unlocked(1, 0.6, 1.0, 0.5) is False
    assert layer_is_unlocked(1, 10.0, 1.0, None) is True


def test_flood_forbid() -> None:
    with pytest.raises(DepthResidualForbidden, match="allow_full"):
        reject_depth_residual_flood(2, 2, False)
    reject_depth_residual_flood(1, 2, False)
    reject_depth_residual_flood(4, 4, True)


def test_honesty_payload_sealed() -> None:
    payload = honesty_payload()
    assert payload["navier_stokes_proof_claim"] is False
    assert payload["stretch_1e-13_cleared"] is False
    assert payload["hilbert_not_in_scope"] is True
    assert payload["greedy_only_claimed_optimal"] is False
    payload["navier_stokes_proof_claim"] = True
    assert honesty_payload()["navier_stokes_proof_claim"] is False


def test_report_rejects_forged_honesty() -> None:
    kwargs = dict(
        residual_norms=(1.0,),
        residual_norms_before=(2.0,),
        unlocked=(True,),
        n_directions=1,
        n_params=4,
        n_gn_steps=1,
        last_layer_only=False,
    )
    with pytest.raises(ValueError, match="navier_stokes"):
        DepthResidualReport(**kwargs, navier_stokes_proof_claim=True)
    with pytest.raises(ValueError, match="stretch"):
        DepthResidualReport(**kwargs, stretch_cleared=True)
    with pytest.raises(ValueError, match="hilbert"):
        DepthResidualReport(**kwargs, hilbert_not_in_scope=False)
    with pytest.raises(ValueError, match="greedy"):
        DepthResidualReport(**kwargs, greedy_only_claimed_optimal=True)


def test_symmetric_mask_second_derivative() -> None:
    # s = 1 - x^2, s'' = -2
    assert hard_bc_mask_tower(0.5, "symmetric", 2)[2] == pytest.approx(-2.0)
    assert hard_bc_mask_tower(0.0, "none", 2) == (1.0, 0.0, 0.0)

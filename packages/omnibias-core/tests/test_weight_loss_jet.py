# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form one-layer weight-space loss jets."""

from __future__ import annotations

import math

import pytest
from omnibias.core.weight_loss_jet import (
    DISCLAIMER,
    WeightLossJetSpec,
    eval_sigma_derivative,
    finite_difference_jet,
    honesty_payload,
    one_layer_loss,
    one_layer_loss_grad,
    one_layer_loss_hessian,
    one_layer_loss_jet,
    one_layer_newton_direction,
    one_layer_param_count,
    pack_one_layer_params,
    unpack_one_layer_params,
    worked_example,
)

_XS = ((0.5, -0.25), (0.0, 0.75), (-0.4, 0.2))
_YS = (0.15, -0.05, 0.3)
_THETA = pack_one_layer_params(
    0.05,
    (0.8, -0.4),
    (0.1, -0.2),
    ((0.3, -0.1), (0.2, 0.5)),
)
_DIR = pack_one_layer_params(
    0.0,
    (0.2, 0.0),
    (0.0, -0.1),
    ((1.0, 0.0), (0.0, 0.5)),
)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _quad(hess: list[list[float]], direction: list[float]) -> float:
    return sum(
        direction[i] * _dot(row, direction) for i, row in enumerate(hess)
    )


def test_honesty_refuses_rejected_claims() -> None:
    payload = honesty_payload()
    assert payload["closed_form"] is True
    assert payload["one_layer_only"] is True
    assert payload["deep_net_closed_form"] is False
    assert payload["skip_chain_rule"] is False
    assert payload["full_parameter_jacobian"] is False
    assert payload["global_min_claim"] is False
    assert payload["stretch_claim"] is False
    assert payload["theorem_prover_verified"] is False
    assert "chain rule" in DISCLAIMER
    assert "global min" in DISCLAIMER


def test_pack_roundtrip_and_param_count() -> None:
    hidden, dim = 2, 2
    assert one_layer_param_count(hidden, dim) == 9
    bias, readout, hbias, weights = unpack_one_layer_params(_THETA, hidden, dim)
    assert pack_one_layer_params(bias, readout, hbias, weights) == pytest.approx(
        _THETA
    )


def test_worked_example_identities() -> None:
    example = worked_example()
    assert example["jet0_matches_loss"] is True
    assert example["jet1_matches_gdotd"] is True
    assert example["jet2_matches_dHd"] is True


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_jet_matches_grad_hess_and_finite_difference(activation: str) -> None:
    hidden, dim = 2, 2
    jet = one_layer_loss_jet(
        _XS,
        _YS,
        _THETA,
        _DIR,
        order=3,
        hidden=hidden,
        dim=dim,
        activation=activation,
    )
    loss = one_layer_loss(_XS, _YS, _THETA, hidden, dim, activation)
    grad = one_layer_loss_grad(_XS, _YS, _THETA, hidden, dim, activation)
    hess = one_layer_loss_hessian(_XS, _YS, _THETA, hidden, dim, activation)
    assert jet[0] == pytest.approx(loss, abs=1e-14)
    assert jet[1] == pytest.approx(_dot(grad, _DIR), abs=1e-13)
    assert jet[2] == pytest.approx(_quad(hess, _DIR), abs=1e-12)

    fd = finite_difference_jet(
        _XS,
        _YS,
        _THETA,
        _DIR,
        order=3,
        hidden=hidden,
        dim=dim,
        activation=activation,
        step=1e-4,
    )
    assert jet[0] == pytest.approx(fd[0], abs=1e-14)
    assert jet[1] == pytest.approx(fd[1], rel=1e-6, abs=1e-8)
    assert jet[2] == pytest.approx(fd[2], rel=5e-4, abs=1e-6)
    assert jet[3] == pytest.approx(fd[3], rel=2e-2, abs=5e-4)


def test_sigma_tower_matches_polynomials() -> None:
    z = 0.3
    t = math.tanh(z)
    assert eval_sigma_derivative(z, 0, "tanh") == pytest.approx(t)
    # tanh' = 1 - t^2
    assert eval_sigma_derivative(z, 1, "tanh") == pytest.approx(1.0 - t * t)
    s = 1.0 / (1.0 + math.exp(-z))
    assert eval_sigma_derivative(z, 0, "sigmoid") == pytest.approx(s)
    assert eval_sigma_derivative(z, 1, "sigmoid") == pytest.approx(s * (1.0 - s))


def test_newton_direction_is_descent() -> None:
    hidden, dim = 2, 2
    grad = one_layer_loss_grad(_XS, _YS, _THETA, hidden, dim, "tanh")
    hess = one_layer_loss_hessian(_XS, _YS, _THETA, hidden, dim, "tanh")
    direction = one_layer_newton_direction(grad, hess, damping=1e-4)
    assert _dot(grad, direction) < 0.0
    # A raw length-1 Newton step can overshoot; a short restriction must descend.
    baseline = one_layer_loss(_XS, _YS, _THETA, hidden, dim, "tanh")
    trial = [p + 0.05 * d for p, d in zip(_THETA, direction, strict=True)]
    assert one_layer_loss(_XS, _YS, trial, hidden, dim, "tanh") < baseline


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="order"):
        WeightLossJetSpec(order=-1)
    with pytest.raises(ValueError, match="activation"):
        WeightLossJetSpec(activation="relu")
    with pytest.raises(ValueError, match="hidden"):
        one_layer_param_count(0, 2)
    with pytest.raises(ValueError, match="exceeds max_params"):
        one_layer_loss_hessian(
            _XS,
            _YS,
            _THETA,
            hidden=2,
            dim=2,
            max_params=3,
        )
    with pytest.raises(ValueError, match="n must be >= 0"):
        eval_sigma_derivative(0.0, -1, "tanh")


def test_spec_normalizes_activation() -> None:
    spec = WeightLossJetSpec(activation="Tanh")
    assert spec.activation == "tanh"

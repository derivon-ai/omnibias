# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for joint operator-predictor learning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "packages" / "omnibias-core" / "src", ROOT / "packages" / "omnibias-torch" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from omnibias.torch.architectures import JointOperatorRegressor, fit_joint_operator_regressor


def _synthetic(seed: int = 0, n: int = 700) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 4)).astype(np.float32)
    x[:, 3] = rng.uniform(-np.pi, np.pi, size=n)
    y = 3.0 * x[:, 0] ** 2 - 2.0 * x[:, 1] * x[:, 2] + np.sin(x[:, 3])
    y = y + rng.normal(0.0, 0.05, size=n)
    return x, y.astype(np.float32)


def test_joint_operator_bank_contains_scientific_primitives() -> None:
    model = JointOperatorRegressor(4, ombu_channels=2, stochastic_gates=False)
    names = [item.name for item in model.operator_metadata]
    assert "x1^2" in names
    assert "x2*x3" in names
    assert "sin(x4)" in names
    assert "tanh(x1^2)" in names
    assert "ombu_grad_0" in names
    bank = model.operator_bank(torch.randn(5, 4))
    assert bank.shape == (5, model.n_operators)


def test_joint_operator_inv_one_plus_abs_metadata_matches_impl() -> None:
    """The unary reciprocal operator computes ``1/(1+|x|)``; its metadata must
    be named ``inv_one_plus_abs`` (not the misleading ``inv_abs`` == 1/|x|)."""
    model = JointOperatorRegressor(
        2, include_pairwise=False, include_nested=False,
        ombu_channels=0, stochastic_gates=False,
    )
    names = [m.name for m in model.operator_metadata]
    families = {m.family for m in model.operator_metadata}
    assert "inv_one_plus_abs(x1)" in names
    assert "inv_abs(x1)" not in names
    assert "inv_one_plus_abs" in families
    assert "inv_abs" not in families
    # The labelled column actually computes 1/(1+|x|).
    x = torch.randn(7, 2)
    bank = model.operator_bank(x)
    idx = names.index("inv_one_plus_abs(x1)")
    expected = 1.0 / (1.0 + x[:, 0].abs())
    assert torch.allclose(bank[:, idx], expected, atol=1e-6)


def test_joint_operator_fit_recovers_synthetic_law_primitives() -> None:
    x, y = _synthetic()
    x_train, y_train = x[:500], y[:500]
    x_val, y_val = x[500:], y[500:]
    fitted = fit_joint_operator_regressor(
        x_train,
        y_train,
        x_val,
        y_val,
        seed=0,
        epochs=260,
        patience=80,
        sparsity_weight=2e-3,
        model_kwargs={
            "ombu_channels": 0,
            "stochastic_gates": False,
            "initial_gate_logit": 0.0,
        },
    )
    pred = fitted.predict(x_val)
    rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
    selected_names = {row["name"] for row in fitted.selected_operators(top_k=12)}
    assert rmse < 0.25
    assert "x1^2" in selected_names
    assert "x2*x3" in selected_names
    assert "sin(x4)" in selected_names


def test_joint_operator_complexity_loss_is_differentiable() -> None:
    model = JointOperatorRegressor(3, ombu_channels=1, stochastic_gates=True)
    x = torch.randn(16, 3)
    y = torch.randn(16)
    pred = model(x)
    loss = torch.mean((pred - y) ** 2) + 1e-3 * model.complexity_loss()
    loss.backward()
    assert model.gate_logits.grad is not None
    assert torch.isfinite(model.gate_logits.grad).all()


def test_joint_operator_fit_accepts_sample_weights() -> None:
    x, y = _synthetic(n=500)
    weights = np.ones(350, dtype=np.float32)
    weights[y[:350] < np.median(y[:350])] = 2.0
    fitted = fit_joint_operator_regressor(
        x[:350],
        y[:350],
        x[350:],
        y[350:],
        seed=1,
        epochs=80,
        patience=30,
        train_sample_weight=weights,
        val_sample_weight=np.ones(150, dtype=np.float32),
        model_kwargs={"ombu_channels": 0, "stochastic_gates": False},
    )
    pred = fitted.predict(x[350:])
    assert np.isfinite(pred).all()


def test_joint_operator_fit_accepts_asymmetric_loss() -> None:
    x, y = _synthetic(n=420)
    fitted = fit_joint_operator_regressor(
        x[:300],
        y[:300],
        x[300:],
        y[300:],
        seed=2,
        epochs=60,
        patience=25,
        asymmetric_weight=1e-3,
        model_kwargs={"ombu_channels": 0, "stochastic_gates": False},
    )
    assert "val_asymmetric" in fitted.history
    assert np.isfinite(fitted.history["val_asymmetric"]).all()


def test_joint_operator_fit_is_robust_to_float64_default_dtype() -> None:
    """A process-wide ``float64`` default must not desync the float32 model
    from its float32 inputs.

    Regression for the ``addmv ... Float, Double, Float`` crash in the ridge
    polish (and the matching nn.Linear mismatch on the OMBU bank path) that
    triggered when a user ran under ``torch.set_default_dtype(torch.float64)``.
    The autouse conftest fixture restores the default after this test.
    """
    torch.set_default_dtype(torch.float64)
    x, y = _synthetic(n=240)
    fitted = fit_joint_operator_regressor(
        x[:160],
        y[:160],
        x[160:],
        y[160:],
        seed=0,
        epochs=40,
        patience=20,
        model_kwargs={"ombu_channels": 2, "stochastic_gates": False},
    )
    assert fitted.model.readout.weight.dtype == torch.float32
    assert fitted.model.gate_logits.dtype == torch.float32
    pred = fitted.predict(x[160:])
    assert np.isfinite(pred).all()


def test_joint_operator_fit_accepts_validation_selection_metric() -> None:
    x, y = _synthetic(n=420)
    calls = 0

    def endpoint_weighted_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        weight = np.where(y_true < np.median(y_true), 2.0, 1.0)
        return float(np.average(np.abs(y_pred - y_true), weights=weight))

    fitted = fit_joint_operator_regressor(
        x[:300],
        y[:300],
        x[300:],
        y[300:],
        seed=3,
        epochs=50,
        patience=20,
        validation_selection_metric=endpoint_weighted_metric,
        validation_selection_complexity_weight=1e-3,
        model_kwargs={"ombu_channels": 0, "stochastic_gates": False},
    )
    assert calls == len(fitted.history["val_selection_score"])
    assert np.isfinite(fitted.history["val_selection_score"]).all()


def test_fitted_predict_places_input_on_model_device(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``predict`` must build the input tensor on the model's device so it works
    after ``model.to(<gpu>)``. We assert the ``device=`` actually forwarded to
    ``torch.as_tensor`` equals the model's parameter device; this catches the
    missing-device bug even on CPU-only hardware (the old code passed no
    ``device=`` at all)."""
    from omnibias.torch.architectures.joint_operator import FittedJointOperatorRegressor

    model = JointOperatorRegressor(3, ombu_channels=0, stochastic_gates=False)
    fitted = FittedJointOperatorRegressor(
        model=model,
        x_mean=np.zeros(3, dtype=np.float32),
        x_scale=np.ones(3, dtype=np.float32),
        y_mean=0.0,
        y_scale=1.0,
        history={},
    )
    model_device = next(model.parameters()).device

    captured: dict[str, object] = {}
    real_as_tensor = torch.as_tensor

    def spy(data, *args, **kwargs):  # type: ignore[no-untyped-def]
        # The first as_tensor call is the one in predict (before model(xt)).
        if "device" not in captured:
            captured["device"] = kwargs.get("device", "MISSING")
        return real_as_tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", spy)
    pred = fitted.predict(np.zeros((5, 3), dtype=np.float32))
    assert captured["device"] == model_device
    assert pred.shape == (5,)

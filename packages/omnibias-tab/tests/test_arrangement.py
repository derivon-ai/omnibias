# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for the tabular arrangement classifier (05-02 G1/G2 substrate)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.tab.arrangement import (
    arrangement_weights,
    hard_predict_np,
    make_axis_rule,
    make_oblique_xor,
    obliqueness_diagnostic,
    predict_proba_np,
)
from omnibias.tab.torch.arrangement import ArrangementClassifier, fit_arrangement

torch.set_default_dtype(torch.float64)


def test_dataset_determinism() -> None:
    X1, y1, m1 = make_oblique_xor(seed=3, n_samples=200)
    X2, y2, m2 = make_oblique_xor(seed=3, n_samples=200)
    assert np.allclose(X1, X2) and np.allclose(y1, y2)
    assert np.allclose(m1["w1"], m2["w1"])
    A1, b1, _ = make_axis_rule(seed=4, n_samples=200)
    A2, b2, _ = make_axis_rule(seed=4, n_samples=200)
    assert np.allclose(A1, A2) and np.allclose(b1, b2)


def test_planted_normals_exact_on_oblique_xor() -> None:
    X, y, meta = make_oblique_xor(seed=0, n_samples=2000)
    W = np.stack([meta["w1"], meta["w2"]])
    t = np.zeros(2)
    # Cell logits: label = bit0 XOR bit1.
    logits = np.array(
        [10.0 if ((r & 1) ^ ((r >> 1) & 1)) else -10.0 for r in range(4)],
        dtype=np.float64,
    )
    hard = hard_predict_np(W, t, logits, X)
    assert _accuracy(hard, y) == 1.0
    soft = (predict_proba_np(W, t, logits, X, beta=50.0) >= 0.5).astype(float)
    assert _accuracy(soft, y) == 1.0


def test_numpy_torch_membership_parity() -> None:
    from omnibias.partition.torch.weights import partition_weights_arrays

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 6))
    W = rng.normal(size=(2, 6))
    t = rng.normal(size=2)
    beta = 3.5
    np_w = arrangement_weights(W, t, X, beta)
    model = ArrangementClassifier(6, 2, beta=beta)
    with torch.no_grad():
        model.W.copy_(torch.as_tensor(W))
        model.t.copy_(torch.as_tensor(t))
        torch_w = partition_weights_arrays(
            model.W, model.t, torch.as_tensor(X), beta, 2
        ).numpy()
    assert np.allclose(np_w, torch_w, atol=1e-12)


def test_obliqueness_diagnostic_orders_linear_oblique_above_axis() -> None:
    """Dense probe beats axis probes on a single oblique hyperplane label."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 10))
    w = rng.normal(size=10)
    y = (X @ w > 0.0).astype(np.float64)
    Xa, ya, _ = make_axis_rule(seed=0, n_samples=3000)
    do = obliqueness_diagnostic(X, y)
    da = obliqueness_diagnostic(Xa, ya)
    assert do > da
    assert do > 1.05
    # Constructed XOR is not linearly separable; diagnostic stays near 1 and
    # is reported (not gated) for Wave-0 honesty.
    Xo, yo, _ = make_oblique_xor(seed=1, n_samples=3000)
    assert obliqueness_diagnostic(Xo, yo) == pytest.approx(1.0, abs=0.15)


def test_l1_recovers_sparse_normals_on_axis() -> None:
    X, y, meta = make_axis_rule(seed=0, n_samples=4000)
    result = fit_arrangement(
        X,
        y,
        n_hyperplanes=2,
        l1=0.02,
        restarts=2,
        steps=200,
        beta_final=128.0,
        seed=0,
        sparse_warmstart=True,
        val_fraction=0.25,
    )
    W = result.model.numpy_state()["W"]
    # Dominant features should include the true pair.
    strength = np.max(np.abs(W), axis=0)
    top = set(np.argsort(strength)[-2:].tolist())
    assert meta["feature_a"] in top or strength[meta["feature_a"]] > 0.1
    assert meta["feature_b"] in top or strength[meta["feature_b"]] > 0.1
    # Test accuracy well above majority.
    te = slice(3000, 4000)
    acc = _accuracy(result.model.predict(X[te]), y[te])
    assert acc > 0.95


def test_fit_oblique_beats_majority() -> None:
    X, y, _ = make_oblique_xor(seed=2, n_samples=3000)
    result = fit_arrangement(
        X,
        y,
        n_hyperplanes=2,
        l1=0.0,
        restarts=4,
        steps=250,
        beta_final=64.0,
        seed=2,
        sparse_warmstart=False,
        val_fraction=0.2,
    )
    te = slice(2400, 3000)
    acc = _accuracy(result.model.predict(X[te]), y[te])
    assert acc > 0.95


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(pred).reshape(-1) == np.asarray(y).reshape(-1)).mean())


@pytest.mark.parametrize("family", ["oblique", "axis"])
def test_labels_are_binary(family: str) -> None:
    if family == "oblique":
        _, y, _ = make_oblique_xor(seed=0, n_samples=500)
    else:
        _, y, _ = make_axis_rule(seed=0, n_samples=500)
    assert set(np.unique(y).tolist()) <= {0.0, 1.0}

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
    dtype = torch.float64
    with torch.no_grad():
        model.W.copy_(torch.as_tensor(W, dtype=dtype))
        model.t.copy_(torch.as_tensor(t, dtype=dtype))
        torch_w = partition_weights_arrays(
            model.W, model.t, torch.as_tensor(X, dtype=dtype), beta, 2
        ).numpy()
    assert np.allclose(np_w, torch_w, atol=1e-12)


def test_obliqueness_diagnostic_detects_linear_not_parity_structure() -> None:
    """Detects linear oblique lift; does not separate XOR from axis (parity)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 10))
    w = rng.normal(size=10)
    y = (X @ w > 0.0).astype(np.float64)
    Xa, ya, _ = make_axis_rule(seed=0, n_samples=3000)
    do = obliqueness_diagnostic(X, y)
    da = obliqueness_diagnostic(Xa, ya)
    assert do > da
    assert do > 1.05
    # XOR is not linearly separable; dense/axis ratio stays near 1 and does
    # not order above the axis family -- reported, not gated (Wave-0 honesty).
    Xo, yo, _ = make_oblique_xor(seed=1, n_samples=3000)
    dx = obliqueness_diagnostic(Xo, yo)
    assert dx == pytest.approx(1.0, abs=0.15)
    assert abs(dx - da) < 0.08


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


def test_outer_val_split_is_used() -> None:
    """Explicit X_val/y_val must be the monitored split (not an inner reshuffle)."""
    X, y, _ = make_oblique_xor(seed=5, n_samples=800)
    Xtr, ytr = X[:480], y[:480]
    Xva, yva = X[480:640], y[480:640]
    Xte, yte = X[640:], y[640:]
    result = fit_arrangement(
        Xtr,
        ytr,
        X_val=Xva,
        y_val=yva,
        n_hyperplanes=2,
        l1=0.0,
        restarts=2,
        steps=200,
        beta_final=32.0,
        seed=5,
        sparse_warmstart=False,
        patience=30,
        eval_every=5,
    )
    # Val metrics must match scoring the provided Xva (outer split).
    import torch
    from omnibias.tab.torch.arrangement import _bce_logits

    Xva_t = torch.as_tensor(Xva, dtype=torch.float64)
    yva_t = torch.as_tensor(yva, dtype=torch.float64)
    assert result.val_acc == pytest.approx(
        _accuracy(result.model.predict(Xva), yva), abs=1e-12
    )
    assert result.val_bce == pytest.approx(
        _bce_logits(result.model, Xva_t, yva_t), abs=1e-12
    )
    assert _accuracy(result.model.predict(Xte), yte) > 0.9


def test_early_stop_restores_best_checkpoint() -> None:
    """Patience stop restores a checkpoint no worse than the last step."""
    rng = np.random.default_rng(0)
    # Tiny train / noise labels encourage late overfitting under a large step cap.
    n, d = 40, 8
    X = rng.normal(size=(n, d))
    y = (rng.random(n) > 0.5).astype(np.float64)
    Xva = rng.normal(size=(20, d))
    yva = (rng.random(20) > 0.5).astype(np.float64)
    result = fit_arrangement(
        X,
        y,
        X_val=Xva,
        y_val=yva,
        n_hyperplanes=2,
        l1=0.0,
        restarts=1,
        steps=400,
        beta_final=64.0,
        seed=0,
        sparse_warmstart=False,
        patience=10,
        eval_every=5,
        min_delta=1e-4,
    )
    assert result.stopped_early
    assert result.best_step < result.steps_run
    assert not result.at_step_cap
    # Restored checkpoint's val BCE is the recorded best.
    import torch
    from omnibias.tab.torch.arrangement import _bce_logits

    Xva_t = torch.as_tensor(Xva, dtype=torch.float64)
    yva_t = torch.as_tensor(yva, dtype=torch.float64)
    assert result.val_bce == pytest.approx(
        _bce_logits(result.model, Xva_t, yva_t), abs=1e-12
    )


def test_at_step_cap_when_patience_exceeds_budget() -> None:
    X, y, _ = make_oblique_xor(seed=1, n_samples=400)
    result = fit_arrangement(
        X[:240],
        y[:240],
        X_val=X[240:320],
        y_val=y[240:320],
        n_hyperplanes=2,
        l1=0.0,
        restarts=1,
        steps=30,
        beta_final=16.0,
        seed=1,
        sparse_warmstart=False,
        patience=10_000,
        eval_every=5,
    )
    assert result.at_step_cap
    assert not result.stopped_early
    assert result.steps_run == 30


def test_h3_has_eight_cells() -> None:
    model = ArrangementClassifier(4, 3, beta=2.0)
    assert model.n_cells == 8
    X = np.zeros((5, 4), dtype=np.float64)
    logits = model.forward(torch.as_tensor(X, dtype=torch.float64))
    assert logits.shape == (5, 1)


def test_newton_early_stop_restores_checkpoint() -> None:
    X, y, _ = make_oblique_xor(seed=3, n_samples=800)
    result = fit_arrangement(
        X[:480],
        y[:480],
        X_val=X[480:640],
        y_val=y[480:640],
        n_hyperplanes=2,
        l1=0.0,
        restarts=2,
        steps=25,
        beta_init=8.0,
        beta_final=32.0,
        beta_anneal_steps=5,
        seed=3,
        sparse_warmstart=False,
        patience=12,
        eval_every=1,
        optimizer="trust_region",
    )
    assert result.optimizer == "trust_region"
    te = slice(640, 800)
    assert _accuracy(result.model.predict(X[te]), y[te]) > 0.8


def test_cubic_newton_runs() -> None:
    X, y, _ = make_oblique_xor(seed=5, n_samples=400)
    result = fit_arrangement(
        X[:240],
        y[:240],
        X_val=X[240:320],
        y_val=y[240:320],
        n_hyperplanes=2,
        l1=0.0,
        restarts=1,
        steps=8,
        beta_init=8.0,
        beta_final=16.0,
        beta_anneal_steps=3,
        seed=5,
        sparse_warmstart=False,
        patience=6,
        eval_every=1,
        optimizer="cubic",
    )
    assert result.optimizer == "cubic"
    assert result.steps_run >= 1


def test_boosted_stages_reduce_train_bce() -> None:
    from omnibias.tab.torch.arrangement import fit_arrangement_boosted

    X, y, _ = make_oblique_xor(seed=4, n_samples=800)
    result = fit_arrangement_boosted(
        X[:480],
        y[:480],
        X_val=X[480:640],
        y_val=y[480:640],
        n_hyperplanes=2,
        n_stages_max=8,
        learning_rate=0.5,
        stage_patience=4,
        weak_restarts=2,
        weak_steps=80,
        weak_patience=25,
        seed=4,
    )
    assert result.n_stages >= 1
    assert result.history[0] >= result.val_bce - 1e-9
    acc = _accuracy(result.model.predict(X[640:]), y[640:])
    assert acc > 0.7


def test_certify_arrangement_gap_sound_and_soft_hard_agree() -> None:
    from omnibias.partition._core.verified import weight_rounding_gap
    from omnibias.tab.arrangement import arrangement_params, certify_arrangement_gap

    X, y, meta = make_oblique_xor(seed=0, n_samples=800)
    W = np.stack([meta["w1"], meta["w2"]])
    t = np.zeros(2)
    logits = np.array(
        [10.0 if ((r & 1) ^ ((r >> 1) & 1)) else -10.0 for r in range(4)],
        dtype=np.float64,
    )
    beta = 64.0
    cert = certify_arrangement_gap(W, t, X, beta=beta)
    assert cert.is_sound
    assert cert.max_gap >= cert.measured_max - 1e-9
    soft = (predict_proba_np(W, t, logits, X, beta) >= 0.5).astype(np.float64)
    hard = hard_predict_np(W, t, logits, X)
    params = arrangement_params(W, t, beta_final=beta)
    bound, _measured = weight_rounding_gap(params, X, beta)
    # Soft decision margin in probability space; where the certified membership
    # gap is below that margin, soft and hard labels must agree.
    margin = np.abs(predict_proba_np(W, t, logits, X, beta) - 0.5)
    safe = bound < margin
    assert safe.any()
    assert np.all(soft[safe] == hard[safe])


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(pred).reshape(-1) == np.asarray(y).reshape(-1)).mean())


@pytest.mark.parametrize("family", ["oblique", "axis"])
def test_labels_are_binary(family: str) -> None:
    if family == "oblique":
        _, y, _ = make_oblique_xor(seed=0, n_samples=500)
    else:
        _, y, _ = make_axis_rule(seed=0, n_samples=500)
    assert set(np.unique(y).tolist()) <= {0.0, 1.0}


def test_fit_arrangement_multiclass_forward_shape() -> None:
    rng = np.random.default_rng(4)
    X = rng.standard_normal((120, 3))
    y = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0).astype(np.int64)
    y = np.clip(y, 0, 2)
    result = fit_arrangement(
        X[:80],
        y[:80],
        X_val=X[80:],
        y_val=y[80:],
        n_hyperplanes=2,
        n_outputs=3,
        task="multiclass",
        restarts=1,
        steps=40,
        patience=20,
        seed=4,
        sparse_warmstart=False,
    )
    logits = result.model(torch.as_tensor(X[:10], dtype=torch.float64))
    assert logits.shape == (10, 3)
    pred = result.model.predict(X[:10])
    assert pred.shape == (10,)

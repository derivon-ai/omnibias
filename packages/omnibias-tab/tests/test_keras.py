# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Keras 3 tab layers: numpy vs keras.ops parity (float64)."""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

keras = pytest.importorskip("keras")


def _np(z: object) -> np.ndarray:
    if hasattr(z, "detach"):
        return np.asarray(z.detach().cpu().numpy())  # type: ignore[union-attr]
    return np.asarray(keras.ops.convert_to_numpy(z))


def _atol() -> float:
    # keras.ops on the torch backend is float64-clean after prod_last_axis, but
    # matmul / sigmoid are not ULP-identical to numpy; 1e-7 is still tight.
    return 1e-7 if str(keras.config.backend()) == "torch" else 1e-9


def _available_keras_backends() -> tuple[str, ...]:
    """Backend selected by ``KERAS_BACKEND`` at import; extra packages optional."""
    name = str(keras.config.backend())
    if name == "tensorflow":
        pytest.importorskip("tensorflow")
    elif name == "torch":
        pytest.importorskip("torch")
    elif name != "jax":
        pytest.skip(f"unsupported KERAS_BACKEND={name!r}")
    return (name,)


def _beta_abs_grad(layer: object, X: np.ndarray, *, beta: object | None = None) -> float:
    """Abs gradient of ``beta`` through ``sum(layer(X))``; jax / TF / torch tape."""
    backend = str(keras.config.backend())
    x = keras.ops.convert_to_tensor(X, dtype="float64")
    beta_var = layer.beta if beta is None else beta  # type: ignore[attr-defined]
    if backend == "jax":
        import jax

        trainable = list(layer.trainable_variables)  # type: ignore[attr-defined]
        def loss_fn(weights: object) -> object:
            with keras.StatelessScope(list(zip(trainable, weights, strict=True))):
                return keras.ops.sum(layer(x))  # type: ignore[operator]

        grads = jax.grad(loss_fn)([v.value for v in trainable])
        for var, grad in zip(trainable, grads, strict=True):
            if var is beta_var:
                return float(np.abs(np.asarray(grad)))
        raise AssertionError("beta not in trainable_variables")
    if backend == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        with tf.GradientTape() as tape:
            loss = keras.ops.sum(layer(x))  # type: ignore[operator]
        grad = tape.gradient(loss, beta_var)
        return float(np.abs(np.asarray(grad)))
    if backend == "torch":
        pytest.importorskip("torch")
        loss = keras.ops.sum(layer(x))  # type: ignore[operator]
        loss.backward()
        grad = beta_var.value.grad  # type: ignore[union-attr]
        return float(np.abs(np.asarray(grad.detach().cpu())))
    raise AssertionError(f"unsupported KERAS_BACKEND={backend!r}")


def test_keras_arrangement_parity_float64() -> None:
    from omnibias.tab.keras.arrangement import ArrangementClassifier as KerasArr
    from omnibias.tab.torch.arrangement import ArrangementClassifier

    rng = np.random.default_rng(5)
    X = rng.standard_normal((20, 6))
    W = rng.standard_normal((2, 6)) * 0.3
    t = rng.standard_normal((2,)) * 0.1
    cell = rng.standard_normal((4, 1))
    beta = 4.0
    torch_m = ArrangementClassifier(6, 2, beta=beta)
    with __import__("torch").no_grad():
        import torch

        torch_m.W.copy_(torch.as_tensor(W, dtype=torch.float64))
        torch_m.t.copy_(torch.as_tensor(t, dtype=torch.float64))
        torch_m.cell_logits.copy_(torch.as_tensor(cell, dtype=torch.float64))
    torch_z = torch_m(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()

    layer = KerasArr(6, 2, beta=beta, n_outputs=1, dtype="float64")
    layer.build((None, 6))
    layer.W.assign(W)
    layer.t.assign(t)
    layer.cell_logits.assign(cell)
    keras_z = _np(layer(keras.ops.convert_to_tensor(X, dtype="float64")))
    assert keras_z.shape == (20, 1)
    assert np.max(np.abs(torch_z - keras_z)) < _atol()
    X3 = X.reshape(5, 4, 6)
    keras_3 = _np(layer(keras.ops.convert_to_tensor(X3, dtype="float64")))
    assert keras_3.shape == (5, 4, 1)


def test_keras_softtree_parity_float64() -> None:
    from omnibias.tab import SoftTreeConfig, init_params
    from omnibias.tab.keras.model import SoftTreeEnsemble as KerasTree
    from omnibias.tab.torch.model import SoftTreeEnsemble

    cfg = SoftTreeConfig(n_features=5, n_trees=3, depth=2, task="binary", seed=2)
    p = init_params(cfg, 2)
    torch_m = SoftTreeEnsemble(cfg, p)
    X = np.random.default_rng(8).standard_normal((12, 5))
    import torch

    torch_z = torch_m(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    layer = KerasTree(5, n_trees=3, depth=2, n_outputs=1, beta=float(cfg.beta_final), dtype="float64")
    layer.build((None, 5))
    layer.W.assign(p.W)
    layer.t.assign(p.t)
    layer.leaves.assign(p.leaves)
    layer.b0.assign(p.b0)
    keras_z = _np(layer(keras.ops.convert_to_tensor(X, dtype="float64")))
    assert keras_z.shape == (12, 1)
    assert np.max(np.abs(torch_z - keras_z)) < _atol()


def test_keras_boosted_parity_float64() -> None:
    _available_keras_backends()
    from omnibias.tab.keras.arrangement import ArrangementBoosted as KerasBoosted
    from omnibias.tab.keras.arrangement import ArrangementClassifier as KerasArr
    from omnibias.tab.torch.arrangement import ArrangementBoosted, ArrangementClassifier

    rng = np.random.default_rng(6)
    X = rng.standard_normal((20, 6))
    import torch

    torch_members = []
    keras_members = []
    for i in range(2):
        W = rng.standard_normal((2, 6)) * 0.3
        t = rng.standard_normal((2,)) * 0.1
        cell = rng.standard_normal((4, 1))
        beta = 3.0 + 0.5 * i
        tm = ArrangementClassifier(6, 2, beta=beta)
        with torch.no_grad():
            tm.W.copy_(torch.as_tensor(W, dtype=torch.float64))
            tm.t.copy_(torch.as_tensor(t, dtype=torch.float64))
            tm.cell_logits.copy_(torch.as_tensor(cell, dtype=torch.float64))
        torch_members.append(tm)
        km = KerasArr(6, 2, beta=beta, n_outputs=1, dtype="float64")
        km.build((None, 6))
        km.W.assign(W)
        km.t.assign(t)
        km.cell_logits.assign(cell)
        keras_members.append(km)
    torch_b = ArrangementBoosted(torch_members, learning_rate=0.4, base=-0.1)
    keras_b = KerasBoosted(keras_members, learning_rate=0.4, base=-0.1, dtype="float64")
    keras_b.build((None, 6))
    torch_z = torch_b(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    keras_z = _np(keras_b(keras.ops.convert_to_tensor(X, dtype="float64")))
    assert keras_z.shape == (20, 1)
    assert np.max(np.abs(torch_z - keras_z)) < _atol()
    X3 = X.reshape(5, 4, 6)
    keras_3 = _np(keras_b(keras.ops.convert_to_tensor(X3, dtype="float64")))
    assert keras_3.shape == (5, 4, 1)


@pytest.mark.parametrize("kind", ["softtree", "arrangement", "boosted"])
def test_keras_learnable_beta_tape_grad(kind: str) -> None:
    _available_keras_backends()
    rng = np.random.default_rng(8)
    X = rng.standard_normal((6, 4))
    if kind == "softtree":
        from omnibias.tab.keras.model import SoftTreeEnsemble as KerasTree

        frozen = KerasTree(
            4, n_trees=2, depth=1, n_outputs=1, beta=2.5, learnable_beta=False, dtype="float64"
        )
        learn = KerasTree(
            4, n_trees=2, depth=1, n_outputs=1, beta=2.5, learnable_beta=True, dtype="float64"
        )
        learn.build((None, 4))
        learn.leaves.assign(rng.standard_normal(tuple(int(s) for s in learn.leaves.shape)))
        frozen.build((None, 4))
        learn.build((None, 4))
        assert frozen.beta.trainable is False
        assert learn.beta.trainable is True
        assert float(_beta_abs_grad(learn, X)) > 0.0
        return
    if kind == "arrangement":
        from omnibias.tab.keras.arrangement import ArrangementClassifier as KerasArr

        frozen = KerasArr(4, 2, beta=2.5, learnable_beta=False, dtype="float64")
        learn = KerasArr(4, 2, beta=2.5, learnable_beta=True, dtype="float64")
        learn.build((None, 4))
        learn.cell_logits.assign(rng.standard_normal((4, 1)))
        frozen.build((None, 4))
        learn.build((None, 4))
        assert frozen.beta.trainable is False
        assert learn.beta.trainable is True
        assert float(_beta_abs_grad(learn, X)) > 0.0
        return

    from omnibias.tab.keras.arrangement import ArrangementBoosted as KerasBoosted

    frozen = KerasBoosted(
        n_features=4,
        n_members=2,
        n_hyperplanes=2,
        beta=2.5,
        learnable_beta=False,
        dtype="float64",
    )
    learn = KerasBoosted(
        n_features=4,
        n_members=2,
        n_hyperplanes=2,
        beta=2.5,
        learnable_beta=True,
        dtype="float64",
    )
    frozen.build((None, 4))
    learn.build((None, 4))
    for member in frozen.members:
        assert member.beta.trainable is False
    for member in learn.members:
        assert member.beta.trainable is True
        member.cell_logits.assign(
            rng.standard_normal(tuple(int(s) for s in member.cell_logits.shape))
        )
    assert float(_beta_abs_grad(learn, X, beta=learn.members[0].beta)) > 0.0

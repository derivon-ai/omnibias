# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Float64 arrangement / boosted-arrangement parity: numpy vs torch vs jax."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.partition.registry import combine_outputs
from omnibias.tab.arrangement import arrangement_weights
from omnibias.tab.torch.arrangement import ArrangementBoosted, ArrangementClassifier

jnp = pytest.importorskip("jax.numpy")


def _numpy_logits(
    W: np.ndarray, t: np.ndarray, cell_logits: np.ndarray, X: np.ndarray, beta: float
) -> np.ndarray:
    weights = arrangement_weights(W, t, X, beta)
    broadcast = np.broadcast_to(
        np.asarray(cell_logits, dtype=np.float64), weights.shape
    )
    return combine_outputs(weights, broadcast)


def _load_classifier(
    W: np.ndarray, t: np.ndarray, cell: np.ndarray, beta: float
) -> ArrangementClassifier:
    k = 1 if np.asarray(cell).ndim == 1 else int(cell.shape[-1])
    task = "binary" if k == 1 else "multiclass"
    model = ArrangementClassifier(
        W.shape[1], W.shape[0], beta=beta, n_outputs=k, task=task if k > 1 else "binary"
    )
    with torch.no_grad():
        model.W.copy_(torch.as_tensor(W, dtype=torch.float64))
        model.t.copy_(torch.as_tensor(t, dtype=torch.float64))
        src = torch.as_tensor(cell, dtype=torch.float64)
        if src.ndim == 1:
            src = src.unsqueeze(-1)
        model.cell_logits.copy_(src)
    return model


def _squeeze1(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[-1] == 1:
        return arr.reshape(arr.shape[:-1])
    return arr


@pytest.mark.parametrize("beta", [0.5, 3.0, 12.0])
def test_arrangement_logits_numpy_torch_jax(beta: float) -> None:
    from omnibias.tab.jax.arrangement import arrangement_forward

    rng = np.random.default_rng(11)
    X = rng.standard_normal((20, 6))
    W = rng.standard_normal((2, 6)) * 0.4
    t = rng.standard_normal((2,)) * 0.2
    cell = rng.standard_normal((4,))
    np_z = _numpy_logits(W, t, cell, X, beta)
    torch_z = _squeeze1(
        _load_classifier(W, t, cell, beta)(torch.as_tensor(X, dtype=torch.float64))
        .detach()
        .numpy()
    )
    jax_z = _squeeze1(
        np.asarray(
            arrangement_forward(
                jnp.asarray(W), jnp.asarray(t), jnp.asarray(cell), jnp.asarray(X), beta
            )
        )
    )
    assert np.max(np.abs(np_z - torch_z)) < 1e-9
    assert np.max(np.abs(np_z - jax_z)) < 1e-9


def test_boosted_logits_numpy_torch_jax() -> None:
    from omnibias.tab.jax.arrangement import boosted_forward

    rng = np.random.default_rng(13)
    X = rng.standard_normal((16, 5))
    beta = 4.0
    lr = 0.3
    base = 0.15
    members = []
    W_stack, t_stack, cell_stack = [], [], []
    for _ in range(3):
        W = rng.standard_normal((2, 5)) * 0.4
        t = rng.standard_normal((2,)) * 0.2
        cell = rng.standard_normal((4,))
        W_stack.append(W)
        t_stack.append(t)
        cell_stack.append(cell)
        members.append(_load_classifier(W, t, cell, beta))
    np_z = base + lr * sum(
        _numpy_logits(W, t, cell, X, beta)
        for W, t, cell in zip(W_stack, t_stack, cell_stack, strict=True)
    )
    boosted = ArrangementBoosted(members, learning_rate=lr, base=base)
    torch_z = _squeeze1(
        boosted(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    )
    jax_z = _squeeze1(
        np.asarray(
            boosted_forward(
                jnp.asarray(np.stack(W_stack)),
                jnp.asarray(np.stack(t_stack)),
                jnp.asarray(np.stack(cell_stack)),
                jnp.asarray(X),
                beta,
                lr,
                base,
            )
        )
    )
    assert np.max(np.abs(np_z - torch_z)) < 1e-9
    assert np.max(np.abs(np_z - jax_z)) < 1e-9


def test_arrangement_leading_dims_torch_jax() -> None:
    from omnibias.tab.jax.arrangement import arrangement_forward, boosted_forward

    rng = np.random.default_rng(17)
    X = rng.standard_normal((2, 4, 6))
    W = rng.standard_normal((2, 6)) * 0.3
    t = rng.standard_normal((2,)) * 0.1
    cell = rng.standard_normal((4,))
    beta = 5.0
    model = _load_classifier(W, t, cell, beta)
    torch_z = model(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    jax_z = np.asarray(
        arrangement_forward(
            jnp.asarray(W), jnp.asarray(t), jnp.asarray(cell), jnp.asarray(X), beta
        )
    )
    assert torch_z.shape == (2, 4, 1)
    assert jax_z.shape == (2, 4, 1)
    assert np.max(np.abs(torch_z - jax_z)) < 1e-9

    boosted = ArrangementBoosted([model], learning_rate=0.4, base=0.1)
    torch_b = boosted(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    jax_b = np.asarray(
        boosted_forward(
            jnp.asarray(W[None, ...]),
            jnp.asarray(t[None, ...]),
            jnp.asarray(cell[None, ...]),
            jnp.asarray(X),
            beta,
            0.4,
            0.1,
        )
    )
    assert torch_b.shape == (2, 4, 1)
    assert jax_b.shape == (2, 4, 1)
    assert np.max(np.abs(torch_b - jax_b)) < 1e-9


@pytest.mark.parametrize("k", [2, 3])
def test_multiclass_arrangement_numpy_torch_jax(k: int) -> None:
    from omnibias.tab.arrangement import predict_proba_np
    from omnibias.tab.jax.arrangement import arrangement_forward

    rng = np.random.default_rng(19 + k)
    X = rng.standard_normal((18, 5))
    W = rng.standard_normal((2, 5)) * 0.4
    t = rng.standard_normal((2,)) * 0.2
    cell = rng.standard_normal((4, k))
    beta = 3.5
    model = _load_classifier(W, t, cell, beta)
    torch_z = model(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    jax_z = np.asarray(
        arrangement_forward(
            jnp.asarray(W), jnp.asarray(t), jnp.asarray(cell), jnp.asarray(X), beta
        )
    )
    assert torch_z.shape == (18, k)
    assert jax_z.shape == (18, k)
    assert np.max(np.abs(torch_z - jax_z)) < 1e-9
    proba = predict_proba_np(W, t, cell, X, beta)
    assert proba.shape == (18, k)
    assert np.allclose(proba.sum(axis=-1), 1.0, atol=1e-8)

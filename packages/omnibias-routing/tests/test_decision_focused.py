# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Decision-focused routing: regret metric, SPO+ subgradient, ours < two-stage."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.routing import normalized_regret, optimal_tour_costs, spo_plus_gradient

P, HID, EPS = 4, 12, 0.05


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _zero_diag(c: np.ndarray) -> np.ndarray:
    c = np.array(c)
    for b in range(c.shape[0]):
        np.fill_diagonal(c[b], 0.0)
    return c


def _truth(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 500)
    return rng.standard_normal((HID, P)) / np.sqrt(P), rng.standard_normal(HID) / np.sqrt(HID)


def _features(n: int, m: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((m, n, n, P))


def _true_cost(phi: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return _zero_diag(_softplus(1.5 * (np.tanh(phi @ w1.T) @ w2)) + EPS)


def _predict(phi: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _zero_diag(_softplus(phi @ w + b) + EPS)


def test_normalized_regret_sanity() -> None:
    """Predicting true costs -> ~oracle decisions; random costs -> large regret."""
    w1, w2 = _truth(0)
    phi = _features(6, 50, 2)
    c = _true_cost(phi, w1, w2)
    opt = optimal_tour_costs(c)
    rng = np.random.default_rng(9)
    rand = _zero_diag(np.abs(rng.standard_normal(c.shape)) + EPS)
    assert normalized_regret(c, c, opt) < 0.02
    assert normalized_regret(rand, c, opt) > 0.3
    assert normalized_regret(c, c, opt) < normalized_regret(rand, c, opt)


def test_normalized_regret_shape_validation() -> None:
    with pytest.raises(ValueError, match="B, n, n"):
        normalized_regret(np.zeros((6, 6)), np.zeros((6, 6)))


def test_spo_plus_gradient() -> None:
    """The SPO+ subgradient is (B, n, n), finite, and vanishes when pred == true."""
    w1, w2 = _truth(1)
    c = _true_cost(_features(6, 4, 3), w1, w2)
    g = spo_plus_gradient(c, c)
    assert g.shape == c.shape
    assert np.all(np.isfinite(g))
    assert np.allclose(g, 0.0)  # pred == true -> both oracle tours coincide


def _ridge_two_stage(phi: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float]:
    x = phi.reshape(-1, P)
    y = np.log(np.expm1(np.clip(c.reshape(-1) - EPS, 1e-6, None)))
    aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    theta = np.linalg.solve(aug.T @ aug + 1e-2 * np.eye(P + 1), aug.T @ y)
    return theta[:P], float(theta[P])


def test_decision_focused_beats_two_stage() -> None:
    """Backprop through the relaxation yields lower test regret than a two-stage fit."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from omnibias.routing import RelaxationSchedule
    from omnibias.routing.jax import decision_cost

    w1, w2 = _truth(0)
    phi_tr, phi_te = _features(6, 40, 1), _features(6, 60, 2)
    c_tr, c_te = _true_cost(phi_tr, w1, w2), _true_cost(phi_te, w1, w2)
    opt_te = optimal_tour_costs(c_te)

    w0, b0 = _ridge_two_stage(phi_tr, c_tr)
    r_two_stage = normalized_regret(_predict(phi_te, w0, b0), c_te, opt_te)

    phij, cj = jnp.asarray(phi_tr), jnp.asarray(c_tr)
    sched = RelaxationSchedule.fast()

    def loss(params: tuple) -> jnp.ndarray:
        w, b = params
        cpred = jax.nn.softplus(phij @ w + b) + EPS
        return decision_cost(cpred, cj, kind="assignment", schedule=sched)

    vg = jax.jit(jax.value_and_grad(loss))
    w, b, lr = jnp.asarray(w0), jnp.asarray(float(b0)), 0.1
    for _ in range(25):  # plain SGD (no optax dependency in tests)
        _, (gw, gb) = vg((w, b))
        w, b = w - lr * gw, b - lr * gb
    r_ours = normalized_regret(_predict(phi_te, np.asarray(w), float(b)), c_te, opt_te)

    assert r_ours <= r_two_stage + 1e-9  # decision-focused is at least as good
    assert r_ours < 0.5 * r_two_stage + 0.02  # and here clearly better (~2x)


def test_decision_cost_parity_jax_torch() -> None:
    """decision_cost is bit-identical across backends on a fixed input."""
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    import jax
    import torch

    jax.config.update("jax_enable_x64", True)
    from omnibias.routing.jax import decision_cost as dc_jax
    from omnibias.routing.torch import decision_cost as dc_torch

    w1, w2 = _truth(2)
    phi = _features(6, 5, 7)
    c_pred = _true_cost(phi, w1, w2)
    c_true = _true_cost(_features(6, 5, 8), w1, w2)
    lj = float(dc_jax(c_pred, c_true, kind="flow"))
    lt = float(dc_torch(c_pred, c_true, kind="flow"))
    assert abs(lj - lt) < 1e-10

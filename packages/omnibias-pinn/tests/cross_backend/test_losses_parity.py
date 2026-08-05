# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity tests for ``omnibias.pinn.{torch,jax}.losses``.

Both backends must produce bit-identical outputs (to ``rtol=1e-12,
atol=1e-12``) for every loss helper, given the same input array.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax import losses as jax_losses
from omnibias.pinn.torch import losses as torch_losses


def _allclose(a: torch.Tensor, b, rtol=1e-12, atol=1e-12) -> bool:
    a_np = a.detach().cpu().numpy()
    b_np = np.asarray(b)
    return np.allclose(a_np, b_np, rtol=rtol, atol=atol)


def _residual_pair(shape: tuple[int, ...], seed: int):
    np_R = np.random.default_rng(seed).standard_normal(shape).astype(np.float64)
    return torch.from_numpy(np_R), jnp.asarray(np_R)


@pytest.mark.parametrize(
    "shape", [(4, 32), (4, 16, 16), (2, 8, 8, 8)],
)
def test_sobolev_residual_loss_parity(shape):
    Rt, Rj = _residual_pair(shape, seed=11)
    L = 2.0 * math.pi
    for p in [0.0, 0.5, 1.0, 2.0]:
        loss_t = torch_losses.sobolev_residual_loss(Rt, L=L, sobolev_p=p)
        loss_j = jax_losses.sobolev_residual_loss(Rj, L=L, sobolev_p=p)
        assert _allclose(loss_t, loss_j), f"sobolev p={p} shape={shape}"


@pytest.mark.parametrize(
    "shape", [(4, 16), (4, 16, 16), (2, 8, 8, 8)],
)
def test_sobolev_weight_parity(shape):
    Rt, Rj = _residual_pair(shape, seed=12)
    L = 2.0 * math.pi
    for p in [0.5, 1.0, 2.0]:
        wt = torch_losses.sobolev_weight(Rt, L=L, sobolev_p=p)
        wj = jax_losses.sobolev_weight(Rj, L=L, sobolev_p=p)
        assert _allclose(wt, wj), f"sobolev_weight p={p} shape={shape}"


def test_causal_weights_parity():
    L_per_bin_np = np.linspace(0.1, 1.0, 32, dtype=np.float64)
    Lt = torch.from_numpy(L_per_bin_np)
    Lj = jnp.asarray(L_per_bin_np)
    wt = torch_losses.causal_weights_from_per_bin(Lt, epsilon=2.0)
    wj = jax_losses.causal_weights_from_per_bin(Lj, epsilon=2.0)
    assert _allclose(wt, wj)


@pytest.mark.parametrize(
    "shape", [(8, 16), (8, 16, 16), (4, 8, 8, 8)],
)
def test_causal_residual_loss_plain_parity(shape):
    Rt, Rj = _residual_pair(shape, seed=13)
    for eps in [0.5, 1.0, 2.0]:
        loss_t, w_t = torch_losses.causal_residual_loss(
            Rt, epsilon=eps, return_weights=True,
        )
        loss_j, w_j = jax_losses.causal_residual_loss(
            Rj, epsilon=eps, return_weights=True,
        )
        assert _allclose(loss_t, loss_j), f"plain causal eps={eps} shape={shape}"
        assert _allclose(w_t, w_j), f"plain causal weights eps={eps} shape={shape}"


@pytest.mark.parametrize(
    "shape", [(8, 16, 16), (4, 8, 8, 8)],
)
def test_causal_residual_loss_sobolev_parity(shape):
    Rt, Rj = _residual_pair(shape, seed=14)
    L = 2.0 * math.pi
    for p in [0.5, 1.0, 2.0]:
        loss_t = torch_losses.causal_residual_loss(
            Rt, epsilon=1.0, L=L, sobolev_p=p,
        )
        loss_j = jax_losses.causal_residual_loss(
            Rj, epsilon=1.0, L=L, sobolev_p=p,
        )
        assert _allclose(loss_t, loss_j), f"sobolev causal p={p} shape={shape}"


def test_entropy_consistent_residual_parity():
    Rt, Rj = _residual_pair((4, 8, 8), seed=15)
    loss_t = torch_losses.entropy_consistent_residual(Rt)
    loss_j = jax_losses.entropy_consistent_residual(Rj)
    assert _allclose(loss_t, loss_j)

    Ut, Uj = _residual_pair((4, 8, 8), seed=16)

    def torch_w(u):
        return u * u + 1.0

    def jax_w(u):
        return u * u + 1.0

    loss_t = torch_losses.entropy_consistent_residual(
        Rt, entropy_weight=torch_w, state_for_weight=Ut,
    )
    loss_j = jax_losses.entropy_consistent_residual(
        Rj, entropy_weight=jax_w, state_for_weight=Uj,
    )
    assert _allclose(loss_t, loss_j)


def test_ntk_balanced_loss_parity():
    losses_np = {"pde": 1.5, "bc": 0.7, "ic": 2.1}
    traces_np = {"pde": 12.0, "bc": 4.5, "ic": 1.0}
    losses_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in losses_np.items()}
    losses_j = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in losses_np.items()}
    traces_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in traces_np.items()}
    traces_j = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in traces_np.items()}

    total_t, w_t = torch_losses.ntk_balanced_loss(losses_t, ntk_traces=traces_t)
    total_j, w_j = jax_losses.ntk_balanced_loss(losses_j, ntk_traces=traces_j)
    assert _allclose(total_t, total_j)
    for k in losses_np:
        assert math.isclose(w_t[k], w_j[k], rel_tol=1e-12, abs_tol=1e-12)


def test_mse_residual_loss_parity():
    Rt, Rj = _residual_pair((4, 8, 8), seed=17)
    assert _allclose(torch_losses.mse_residual_loss(Rt),
                     jax_losses.mse_residual_loss(Rj))

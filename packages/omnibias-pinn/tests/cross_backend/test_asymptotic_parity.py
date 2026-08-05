# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity for ``losses.asymptotic`` (torch <-> jax).

Both backends build the same exact Taylor jet and take the same L'Hopital limit,
so every helper must agree to ``rtol=atol=1e-12`` in float64 on identical inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax.losses import asymptotic as jax_asym  # noqa: E402
from omnibias.pinn.torch.losses import asymptotic as torch_asym  # noqa: E402


def _np_mlp(seed: int, dims=(3, 6, 5, 2), act: str = "tanh"):
    rng = np.random.default_rng(seed)
    raw = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        W = rng.normal(scale=0.5, size=(dout, din)).astype(np.float64)
        b = rng.normal(scale=0.3, size=(dout,)).astype(np.float64)
        spec = None if i == len(dims) - 2 else act
        raw.append((W, b, spec))
    x0 = rng.normal(size=(dims[0],)).astype(np.float64)
    v = rng.normal(size=(dims[0],)).astype(np.float64)
    return raw, x0, v


def _to_torch(raw, x0, v):
    layers = [(torch.from_numpy(W), torch.from_numpy(b), s) for W, b, s in raw]
    return layers, torch.from_numpy(x0), torch.from_numpy(v)


def _to_jax(raw, x0, v):
    layers = [(jnp.asarray(W), jnp.asarray(b), s) for W, b, s in raw]
    return layers, jnp.asarray(x0), jnp.asarray(v)


def _agree(t: torch.Tensor, j) -> bool:
    return np.allclose(t.detach().cpu().numpy(), np.asarray(j), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rate", [0, 1, 2])
@pytest.mark.parametrize("out_index", [0, 1])
def test_asymptotic_ratio_parity(rate: int, out_index: int) -> None:
    raw, x0, v = _np_mlp(seed=10 + rate)
    lt, xt, vt = _to_torch(raw, x0, v)
    lj, xj, vj = _to_jax(raw, x0, v)
    rt = torch_asym.asymptotic_ratio(lt, xt, vt, rate=rate, order=3, out_index=out_index)
    rj = jax_asym.asymptotic_ratio(lj, xj, vj, rate=rate, order=3, out_index=out_index)
    assert _agree(rt, rj)


def test_network_ray_jet_parity() -> None:
    raw, x0, v = _np_mlp(seed=21)
    lt, xt, vt = _to_torch(raw, x0, v)
    lj, xj, vj = _to_jax(raw, x0, v)
    jt = torch_asym.network_ray_jet(lt, xt, vt, order=4)
    jj = jax_asym.network_ray_jet(lj, xj, vj, order=4)
    assert _agree(jt, jj)


def test_asymptotic_bc_loss_parity() -> None:
    raw, x0, v = _np_mlp(seed=22)
    lt, xt, vt = _to_torch(raw, x0, v)
    lj, xj, vj = _to_jax(raw, x0, v)
    lt_loss = torch_asym.asymptotic_bc_loss(lt, xt, vt, target=0.2, rate=1, weight=2.5)
    lj_loss = jax_asym.asymptotic_bc_loss(lj, xj, vj, target=0.2, rate=1, weight=2.5)
    assert _agree(lt_loss, lj_loss)


def test_far_field_decay_loss_parity() -> None:
    raw, x0, v = _np_mlp(seed=23)
    lt, xt, vt = _to_torch(raw, x0, v)
    lj, xj, vj = _to_jax(raw, x0, v)
    lt_loss = torch_asym.far_field_decay_loss(lt, xt, vt, order=3, weight=0.75)
    lj_loss = jax_asym.far_field_decay_loss(lj, xj, vj, order=3, weight=0.75)
    assert _agree(lt_loss, lj_loss)

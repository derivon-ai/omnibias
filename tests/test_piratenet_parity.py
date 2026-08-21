# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend apply parity for PirateNet given shared weights."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

import jax.numpy as jnp  # noqa: E402
from omnibias.jax.architectures.piratenet import (  # noqa: E402
    PirateNetConfig as JaxCfg,
)
from omnibias.jax.architectures.piratenet import init_pirate_params as jax_init
from omnibias.jax.architectures.piratenet import pirate_apply as jax_apply
from omnibias.torch.architectures.piratenet import pirate_apply as torch_apply


def _to_torch(tree: object) -> object:
    if isinstance(tree, dict):
        return {k: _to_torch(v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_to_torch(v) for v in tree)
    if isinstance(tree, list):
        return [_to_torch(v) for v in tree]
    return torch.tensor(np.asarray(tree), dtype=torch.float64)


def test_pirate_apply_parity_shared_weights() -> None:
    cfg = JaxCfg(in_dim=2, hidden=5, n_layers=2, seed=3)
    jparams = jax_init(cfg)
    tparams = _to_torch(jparams)
    x = np.array([[0.1, 0.2], [-0.3, 0.4], [0.0, 0.0]], dtype=np.float64)
    yj = np.asarray(jax_apply(jparams, jnp.asarray(x)))
    yt = torch_apply(tparams, torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(yj, yt, atol=1e-14, rtol=0.0)

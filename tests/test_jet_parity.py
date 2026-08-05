# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity and golden regression for the Faà di Bruno jet kernel.

The jax and torch jet kernels share the pure-Python Bell combinatorics and the
identical shifted-power algorithm, so a deep-MLP directional tower must agree to
float64 precision across backends and against a pinned, oracle-derived golden
file (``tests/data/faa_di_bruno_mlp_golden.npz``, computed by nested
``jax.jacfwd``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.jet import jet_to_tower as jax_jet_to_tower  # noqa: E402
from omnibias.jax.jet import mlp_jet as jax_mlp_jet  # noqa: E402
from omnibias.torch.jet import jet_to_tower as torch_jet_to_tower  # noqa: E402
from omnibias.torch.jet import mlp_jet as torch_mlp_jet  # noqa: E402

_GOLDEN = Path(__file__).resolve().parent / "data" / "faa_di_bruno_mlp_golden.npz"


def _load_golden():
    data = np.load(_GOLDEN)
    order = int(data["meta"][0])
    n_layers = sum(1 for k in data.files if k.startswith("W"))
    raw = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)]
    return raw, data["x0"], data["v"], data["tower"], order


def test_cross_backend_and_golden() -> None:
    raw, x0, v, golden_tower, order = _load_golden()

    jax_layers = [
        (jnp.asarray(W), jnp.asarray(b), ("tanh" if i < len(raw) - 1 else None))
        for i, (W, b) in enumerate(raw)
    ]
    torch_layers = [
        (torch.as_tensor(W), torch.as_tensor(b), ("tanh" if i < len(raw) - 1 else None))
        for i, (W, b) in enumerate(raw)
    ]

    jax_tower = np.asarray(
        jax_jet_to_tower(jax_mlp_jet(jnp.asarray(x0), jnp.asarray(v), jax_layers, order))
    )
    torch_tower = (
        torch_jet_to_tower(
            torch_mlp_jet(
                torch.as_tensor(x0), torch.as_tensor(v), torch_layers, order
            )
        )
        .double()
        .numpy()
    )

    # jax vs torch
    assert np.allclose(jax_tower, torch_tower, rtol=1e-12, atol=1e-12)
    # both vs pinned oracle-derived golden
    assert np.allclose(jax_tower, golden_tower, rtol=1e-12, atol=1e-12)
    assert np.allclose(torch_tower, golden_tower, rtol=1e-12, atol=1e-12)

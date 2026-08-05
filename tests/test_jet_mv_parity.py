# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity and golden regression for the multivariate jet kernel.

The jax and torch multivariate kernels share the pure-Python multi-index
combinatorics and the identical shifted-power / truncated-polynomial algorithm,
so the full Taylor-coefficient array of a deep MLP must agree to float64
precision across backends and against a pinned, oracle-derived golden file
(``tests/data/faa_di_bruno_mv_golden.npz``, computed by nested ``jax.jacfwd``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.jet_mv import mlp_jet_mv as jax_mlp_jet_mv  # noqa: E402
from omnibias.torch.jet_mv import mlp_jet_mv as torch_mlp_jet_mv  # noqa: E402

_GOLDEN = Path(__file__).resolve().parent / "data" / "faa_di_bruno_mv_golden.npz"


def _load_golden():
    data = np.load(_GOLDEN)
    order, _dim, _c = (int(v) for v in data["meta"])
    n_layers = sum(1 for k in data.files if k.startswith("W"))
    raw = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)]
    return raw, data["x0"], data["coeffs"], order


def test_cross_backend_and_golden_mv() -> None:
    raw, x0, golden_coeffs, order = _load_golden()

    jax_layers = [
        (jnp.asarray(W), jnp.asarray(b), ("tanh" if i < len(raw) - 1 else None))
        for i, (W, b) in enumerate(raw)
    ]
    torch_layers = [
        (torch.as_tensor(W), torch.as_tensor(b), ("tanh" if i < len(raw) - 1 else None))
        for i, (W, b) in enumerate(raw)
    ]

    jax_coeffs = np.asarray(jax_mlp_jet_mv(jnp.asarray(x0), jax_layers, order))
    torch_coeffs = (
        torch_mlp_jet_mv(torch.as_tensor(x0), torch_layers, order).double().numpy()
    )

    # jax vs torch
    assert np.allclose(jax_coeffs, torch_coeffs, rtol=1e-12, atol=1e-12)
    # both vs pinned oracle-derived golden
    assert np.allclose(jax_coeffs, golden_coeffs, rtol=1e-12, atol=1e-12)
    assert np.allclose(torch_coeffs, golden_coeffs, rtol=1e-12, atol=1e-12)

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
from omnibias.jax.jet import compose_jet as jax_compose_jet  # noqa: E402
from omnibias.jax.jet import compose_jet_riccati as jax_compose_riccati  # noqa: E402
from omnibias.jax.jet import jet_to_tower as jax_jet_to_tower  # noqa: E402
from omnibias.jax.jet import mlp_jet as jax_mlp_jet  # noqa: E402
from omnibias.torch.jet import compose_jet as torch_compose_jet  # noqa: E402
from omnibias.torch.jet import compose_jet_riccati as torch_compose_riccati  # noqa: E402
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


def test_compose_jet_arbitrary_tower_cross_backend() -> None:
    """The valuation-reduced composition kernel agrees across the twins."""
    rng = np.random.default_rng(101)
    for order in (0, 1, 5, 9):
        u = rng.normal(size=(order + 1, 4))
        tower = rng.normal(size=(order + 1, 4))
        j = np.asarray(jax_compose_jet(jnp.asarray(u), jnp.asarray(tower)))
        t = torch_compose_jet(
            torch.as_tensor(u, dtype=torch.float64),
            torch.as_tensor(tower, dtype=torch.float64),
        ).numpy()
        assert np.allclose(j, t, rtol=1e-14, atol=1e-15)


def test_compose_jet_riccati_cross_backend_and_vs_generic() -> None:
    """The O(N^2) Riccati fastpath: twin agreement, and agreement with the tower."""
    rng = np.random.default_rng(103)
    order = 8
    u = np.zeros((order + 1, 3))
    u[0] = 0.3 + rng.normal(scale=0.05, size=3)
    u[1] = rng.normal(scale=0.3, size=3)
    u[2] = rng.normal(scale=0.2, size=3)
    for poly in ((0.0, 1.0, -1.0), (1.0, 0.0, -1.0), (0.0, 1.0)):
        s0 = {
            (0.0, 1.0, -1.0): 1.0 / (1.0 + np.exp(-u[0])),
            (1.0, 0.0, -1.0): np.tanh(u[0]),
            (0.0, 1.0): np.exp(u[0]),
        }[poly]
        j = np.asarray(
            jax_compose_riccati(jnp.asarray(u), jnp.asarray(s0), poly)
        )
        t = torch_compose_riccati(
            torch.as_tensor(u, dtype=torch.float64),
            torch.as_tensor(s0, dtype=torch.float64),
            poly,
        ).numpy()
        assert np.allclose(j, t, rtol=1e-14, atol=1e-15)


def test_mlp_jet_riccati_flag_matches_pinned_golden() -> None:
    """``riccati=True`` reproduces the pinned tanh-MLP tower to rounding."""
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
        jax_jet_to_tower(
            jax_mlp_jet(
                jnp.asarray(x0), jnp.asarray(v), jax_layers, order, riccati=True
            )
        )
    )
    torch_tower = (
        torch_jet_to_tower(
            torch_mlp_jet(
                torch.as_tensor(x0),
                torch.as_tensor(v),
                torch_layers,
                order,
                riccati=True,
            )
        )
        .double()
        .numpy()
    )
    assert np.allclose(jax_tower, torch_tower, rtol=1e-12, atol=1e-13)
    assert np.allclose(jax_tower, golden_tower, rtol=1e-10, atol=1e-11)
    assert np.allclose(torch_tower, golden_tower, rtol=1e-10, atol=1e-11)

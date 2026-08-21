# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-12 G6: torch / jax jet line search bit-identical on float64."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.core.line_search import JetLineSearchConfig


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_g6_quartic_step_bit_identical() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax.line_search import jet_line_search as jax_search
    from omnibias.torch.line_search import jet_line_search as torch_search

    torch.set_default_dtype(torch.float64)

    def torch_loss(p: torch.Tensor) -> torch.Tensor:
        s = p[0]
        return 1.0 - 2.0 * s + 3.0 * s**2 - 2.0 * s**3 + 2.0 * s**4

    def jax_loss(p: jax.Array) -> jax.Array:
        s = p[0]
        return 1.0 - 2.0 * s + 3.0 * s**2 - 2.0 * s**3 + 2.0 * s**4

    cfg = JetLineSearchConfig(order=4, trust_radius=1.0, verify=True)
    t_res = torch_search(
        torch_loss,
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        config=cfg,
        next_derivative_bound=0.0,
    )
    j_res = jax_search(
        jax_loss,
        jnp.zeros(2),
        jnp.array([1.0, 0.0]),
        config=cfg,
        next_derivative_bound=0.0,
    )
    assert _ulp_error(t_res.step, j_res.step) <= 4.0
    assert _ulp_error(t_res.model_value, j_res.model_value) <= 4.0
    assert t_res.fell_back == j_res.fell_back
    assert t_res.truncation_certified == j_res.truncation_certified


def test_g6_directional_derivatives_exp() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax.line_search import directional_derivatives as jax_d
    from omnibias.torch.line_search import directional_derivatives as torch_d

    torch.set_default_dtype(torch.float64)
    p_np = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    d_np = np.array([0.3, 0.4, -0.1], dtype=np.float64)

    def t_loss(p: torch.Tensor) -> torch.Tensor:
        return torch.exp((p**2).sum())

    def j_loss(p: jax.Array) -> jax.Array:
        return jnp.exp(jnp.sum(p**2))

    t_der = torch_d(t_loss, torch.as_tensor(p_np), torch.as_tensor(d_np), 5)
    j_der = jax_d(j_loss, jnp.asarray(p_np), jnp.asarray(d_np), 5)
    worst = max(_ulp_error(a, b) for a, b in zip(t_der, j_der, strict=True))
    assert worst <= 16.0, f"derivative parity worst_ulp={worst}"

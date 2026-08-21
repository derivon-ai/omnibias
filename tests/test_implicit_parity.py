# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX parity for the section-5 scalar DEQ (theory 08-08)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import torch
from omnibias.core.implicit import DEQConfig
from omnibias.jax.implicit import deq_solve as jax_solve
from omnibias.jax.implicit import deq_vjp as jax_vjp
from omnibias.torch.implicit import deq_solve as torch_solve
from omnibias.torch.implicit import deq_vjp as torch_vjp


def test_g3_scalar_parity() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = DEQConfig(solver="newton", tol=1e-14, max_iter=30)
    t_res = torch_solve(torch.tensor([[0.2]]), torch.tensor([0.5]), "tanh", config=cfg)
    j_res = jax_solve(jnp.asarray([[0.2]]), jnp.asarray([0.5]), "tanh", config=cfg)
    t_u = float(t_res.u.reshape(-1)[0])
    j_u = float(j_res.u.reshape(-1)[0])
    t_g = float(
        torch_vjp(
            torch.tensor([[0.2]]),
            torch.tensor([0.5]),
            "tanh",
            torch.tensor([1.0]),
            config=cfg,
        ).reshape(-1)[0]
    )
    j_g = float(
        jax_vjp(
            jnp.asarray([[0.2]]),
            jnp.asarray([0.5]),
            "tanh",
            jnp.asarray([1.0]),
            config=cfg,
        ).reshape(-1)[0]
    )
    assert abs(t_u - j_u) <= 1e-12
    assert abs(t_g - j_g) <= 1e-12
    assert abs(t_res.residual - j_res.residual) <= 1e-12
    assert t_res.n_iter == j_res.n_iter


def test_g3_width2_parity() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = DEQConfig(solver="newton", tol=1e-14, max_iter=40)
    w = [[0.15, -0.05], [0.08, 0.12]]
    x = [0.4, -0.2]
    g = [1.0, 0.0]
    t_res = torch_solve(torch.tensor(w), torch.tensor(x), "tanh", config=cfg)
    j_res = jax_solve(jnp.asarray(w), jnp.asarray(x), "tanh", config=cfg)
    t_g = torch_vjp(
        torch.tensor(w), torch.tensor(x), "tanh", torch.tensor(g), config=cfg
    )
    j_g = jax_vjp(jnp.asarray(w), jnp.asarray(x), "tanh", jnp.asarray(g), config=cfg)
    assert abs(float(t_res.u[0]) - float(j_res.u.reshape(-1)[0])) <= 1e-12
    assert abs(float(t_res.u[1]) - float(j_res.u.reshape(-1)[1])) <= 1e-12
    assert float((t_g - torch.tensor(j_g.tolist())).abs().max()) <= 1e-12

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Implicit DEQ Newton, JAX (theory 08-08)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
from omnibias.core.implicit import DEQConfig, DEQNotContractive, DEQSolverUnknown
from omnibias.jax.implicit import deq_solve, deq_vjp


def test_section5_scalar_fixed_point() -> None:
    cfg = DEQConfig(solver="newton", tol=1e-12, max_iter=20)
    result = deq_solve(jnp.asarray([[0.2]]), jnp.asarray([0.5]), "tanh", config=cfg)
    u = float(result.u.reshape(-1)[0])
    z = 0.2 * u + 0.5
    assert abs(u - float(jnp.tanh(z))) <= 1e-12
    assert result.residual <= 1e-10
    assert 0.542 < u < 0.544


def test_g1_ift_matches_central_fd() -> None:
    cfg = DEQConfig(solver="newton", tol=1e-14, max_iter=30)
    w = jnp.asarray([[0.2]])
    x = jnp.asarray([0.5])
    vjp = float(deq_vjp(w, x, "tanh", jnp.asarray([1.0]), config=cfg).reshape(-1)[0])
    eps = 1e-6
    up = float(deq_solve(w + eps, x, "tanh", config=cfg).u.reshape(-1)[0])
    um = float(deq_solve(w - eps, x, "tanh", config=cfg).u.reshape(-1)[0])
    fd = (up - um) / (2.0 * eps)
    rel = abs(vjp - fd) / max(abs(fd), 1e-12)
    assert rel <= 1e-8
    assert deq_solve(w, x, "tanh", config=cfg).residual <= 1e-10


def test_contraction_raises() -> None:
    cfg = DEQConfig(require_contraction=True, max_iter=2)
    with pytest.raises(DEQNotContractive, match="unroll"):
        deq_solve(jnp.asarray([[3.0]]), jnp.asarray([0.1]), "tanh", config=cfg)


def test_anderson_raises() -> None:
    with pytest.raises(DEQSolverUnknown, match="newton"):
        deq_solve(
            jnp.asarray([[0.2]]),
            jnp.asarray([0.5]),
            "tanh",
            config=DEQConfig(solver="anderson"),
        )

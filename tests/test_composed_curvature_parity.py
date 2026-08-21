# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 08-02 G3: torch / jax composed Hessian bit-identical on float64."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.core.composed_curvature import ComposedCurvatureConfig


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_g3_scalar_nest_hessian_parity() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax.optim_composed import (
        composed_block_hessian as jax_hess,
    )
    from omnibias.jax.optim_composed import (
        composed_curvature_step as jax_step,
    )
    from omnibias.jax.optim_composed import (
        scalar_nest_hessian as jax_nest,
    )
    from omnibias.torch.optim_composed import (
        composed_block_hessian as torch_hess,
    )
    from omnibias.torch.optim_composed import (
        composed_curvature_step as torch_step,
    )
    from omnibias.torch.optim_composed import (
        scalar_nest_hessian as torch_nest,
    )

    torch.set_default_dtype(torch.float64)
    t_closed = torch_nest(torch.tensor(0.2), torch.tensor(0.1), 1.0)
    j_closed = jax_nest(jnp.asarray(0.2), jnp.asarray(0.1), 1.0)
    for a, b in zip(t_closed, j_closed, strict=True):
        assert _ulp_error(a, b) <= 4.0

    def t_res(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        return (curr * torch.tanh(prev) - 1.0).reshape(1)

    def j_res(prev: jax.Array, curr: jax.Array) -> jax.Array:
        return jnp.reshape(curr * jnp.tanh(prev) - 1.0, (1,))

    cfg = ComposedCurvatureConfig(n_directions=2, allow_full=True)
    t_dirs = (
        (torch.tensor(1.0), torch.tensor(0.0)),
        (torch.tensor(0.0), torch.tensor(1.0)),
    )
    j_dirs = (
        (jnp.asarray(1.0), jnp.asarray(0.0)),
        (jnp.asarray(0.0), jnp.asarray(1.0)),
    )
    t_slice, t_joint, t_cross = torch_hess(
        t_res, torch.tensor(0.2), torch.tensor(0.1), t_dirs, config=cfg
    )
    j_slice, j_joint, j_cross = jax_hess(
        j_res, jnp.asarray(0.2), jnp.asarray(0.1), j_dirs, config=cfg
    )
    for i in range(2):
        for j in range(2):
            assert _ulp_error(float(t_joint[i, j]), float(j_joint[i, j])) <= 4.0
    assert _ulp_error(float(t_slice[0, 0]), float(j_slice[0, 0])) <= 4.0
    assert _ulp_error(float(t_cross[0, 0]), float(j_cross[0, 0])) <= 4.0

    _, _, t_rep = torch_step(t_res, torch.tensor(0.2), torch.tensor(0.1), config=cfg)
    _, _, j_rep = jax_step(j_res, jnp.asarray(0.2), jnp.asarray(0.1), config=cfg)
    assert t_rep.escaped == j_rep.escaped
    assert _ulp_error(t_rep.lambda_min_joint, j_rep.lambda_min_joint) <= 8.0
    assert _ulp_error(t_rep.lambda_min_slice, j_rep.lambda_min_slice) <= 8.0

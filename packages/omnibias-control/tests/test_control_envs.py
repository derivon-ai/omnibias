# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""DifferentiableEnvironment implementations: double gyre + advection-diffusion grid."""

from __future__ import annotations

import numpy as np


def test_double_gyre_satisfies_protocol_jax():
    import jax.numpy as jnp
    from omnibias.control.jax.envs import DoubleGyrePointMass
    from omnibias.control.ocp import DifferentiableEnvironment

    env = DoubleGyrePointMass()
    assert isinstance(env, DifferentiableEnvironment)
    y = jnp.array([0.3, 0.4])
    u = jnp.zeros(2)
    y1 = env.step(y, u)
    assert y1.shape == (2,)
    assert np.all(np.isfinite(np.asarray(y1)))
    assert float(env.cost(y, u)) > 0.0
    assert float(env.terminal_cost(y1)) > 0.0


def test_double_gyre_satisfies_protocol_torch():
    import torch
    from omnibias.control.ocp import DifferentiableEnvironment
    from omnibias.control.torch.envs import DoubleGyrePointMass

    torch.set_default_dtype(torch.float64)
    env = DoubleGyrePointMass()
    assert isinstance(env, DifferentiableEnvironment)
    y = torch.tensor([0.3, 0.4])
    u = torch.zeros(2)
    y1 = env.step(y, u)
    assert tuple(y1.shape) == (2,)
    assert torch.all(torch.isfinite(y1))
    assert float(env.cost(y, u)) > 0.0
    assert float(env.terminal_cost(y1)) > 0.0


def test_double_gyre_torch_jax_parity():
    import jax
    import jax.numpy as jnp
    import pytest
    import torch
    from omnibias.control.jax.envs import DoubleGyrePointMass as JaxEnv
    from omnibias.control.torch.envs import DoubleGyrePointMass as TorchEnv

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    jenv, tenv = JaxEnv(), TorchEnv()
    y0 = [0.35, 0.42]
    u0 = [0.01, -0.02]
    jy = jenv.step(jnp.array(y0), jnp.array(u0))
    ty = tenv.step(torch.tensor(y0), torch.tensor(u0))
    assert np.allclose(np.asarray(jy), ty.numpy(), atol=1e-10)
    j_cost = float(jenv.cost(jnp.array(y0), jnp.array(u0)))
    t_cost = float(tenv.cost(torch.tensor(y0), torch.tensor(u0)))
    assert j_cost == pytest.approx(t_cost, rel=1e-9)


def test_advection_diffusion_grid_satisfies_protocol_jax():
    import jax.numpy as jnp
    from omnibias.control.jax.envs import AdvectionDiffusionGrid
    from omnibias.control.ocp import DifferentiableEnvironment

    env = AdvectionDiffusionGrid(n_grid=8)
    assert isinstance(env, DifferentiableEnvironment)
    assert env.state_dim == 8
    assert env.action_dim == 1
    y = jnp.zeros(8)
    u = jnp.array([0.5])
    y1 = env.step(y, u)
    assert y1.shape == (8,)
    assert np.all(np.isfinite(np.asarray(y1)))
    assert float(env.cost(y, u)) >= 0.0


def test_advection_diffusion_grid_rejects_odd_n():
    import pytest
    from omnibias.control.jax.envs import AdvectionDiffusionGrid

    with pytest.raises(ValueError):
        AdvectionDiffusionGrid(n_grid=7)


def test_advection_diffusion_grid_torch_jax_parity():
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.control.jax.envs import AdvectionDiffusionGrid as JaxGrid
    from omnibias.control.torch.envs import AdvectionDiffusionGrid as TorchGrid

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    jenv, tenv = JaxGrid(n_grid=8), TorchGrid(n_grid=8)
    y0 = np.linspace(-1.0, 1.0, 8)
    u0 = [0.3]
    jy = jenv.step(jnp.asarray(y0), jnp.array(u0))
    ty = tenv.step(torch.as_tensor(y0), torch.tensor(u0))
    assert np.allclose(np.asarray(jy), ty.numpy(), atol=1e-8)

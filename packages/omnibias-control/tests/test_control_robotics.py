# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""DifferentiableEnvironment adapters for wrapped robotics engines (theory 10-02, Phase 5).

Gate G8 (cost parity vs SHAC on one hardware class) needs a real Brax or
MuJoCo-MJX installation and is **not** run in this suite; see
``omnibias.control.robotics`` module docstring. These tests instead check
(1) the adapters degrade to a clear, actionable ``ImportError`` when the
engine is absent, and (2) the adapter *seam* itself -- a flat-array
``DifferentiableEnvironment`` around an externally-owned stepping rule --
is exercised end to end via the dependency-free fake fixture, including
training through :mod:`omnibias.control.jax.policy`.
"""

from __future__ import annotations

import pytest
from omnibias.control.ocp import DifferentiableEnvironment
from omnibias.control.robotics import (
    BraxEnvironmentAdapter,
    FakeArticulatedEnvironment,
    MjxEnvironmentAdapter,
    brax_available,
    mjx_available,
)


def test_fake_articulated_environment_satisfies_protocol():
    env = FakeArticulatedEnvironment()
    assert isinstance(env, DifferentiableEnvironment)


def test_fake_articulated_environment_step_is_finite():
    import jax.numpy as jnp

    env = FakeArticulatedEnvironment()
    y = jnp.array([0.1, -0.1, 0.0, 0.0])
    u = jnp.array([0.05, -0.02])
    y1 = env.step(y, u)
    assert y1.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(y1)))
    assert float(env.cost(y, u)) >= 0.0
    assert float(env.terminal_cost(y1)) >= 0.0


def test_fake_articulated_environment_trains_via_exact_adjoint():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.policy import actor_adjoint_step

    env = FakeArticulatedEnvironment()
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    w1 = 0.1 * jax.random.normal(k1, (8, 4))
    b1 = jnp.zeros(8)
    w2 = 0.1 * jax.random.normal(k2, (2, 8))
    b2 = jnp.zeros(2)
    layers = [(w1, b1, "tanh"), (w2, b2, None)]
    y0 = jnp.array([0.2, -0.1, 0.0, 0.0])
    result = actor_adjoint_step(env, layers, y0, horizon=4, lr=0.01)
    import numpy as np

    assert np.all(np.isfinite(result.diagnostics["grad_theta"]))


def test_brax_available_and_mjx_available_return_bool():
    assert isinstance(brax_available(), bool)
    assert isinstance(mjx_available(), bool)


def test_brax_adapter_raises_import_error_without_brax():
    if brax_available():
        pytest.skip("brax is installed in this environment; degradation path not exercised")
    with pytest.raises(ImportError, match="brax"):
        BraxEnvironmentAdapter(
            env=None,
            cost_fn=lambda y, u: 0.0,
            terminal_cost_fn=lambda y: 0.0,
            state_dim=4,
            action_dim=2,
        )


def test_mjx_adapter_raises_import_error_without_mujoco():
    if mjx_available():
        pytest.skip("mujoco.mjx is installed in this environment; degradation path not exercised")
    with pytest.raises(ImportError, match="mujoco"):
        MjxEnvironmentAdapter(model=None, cost_fn=lambda y, u: 0.0, terminal_cost_fn=lambda y: 0.0)


def test_brax_state_environment_raises_without_brax():
    if brax_available():
        pytest.skip("brax is installed in this environment")
    from omnibias.control.robotics import brax_state_environment

    with pytest.raises(ImportError, match="brax"):
        brax_state_environment(None, cost_fn=lambda y, u: 0.0, terminal_cost_fn=lambda y: 0.0)

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified contact-force smoothing environments (jax/torch twins, theory 10-04)."""

from __future__ import annotations

import numpy as np


def test_contact_point_mass_satisfies_protocol_jax():
    import jax.numpy as jnp
    from omnibias.control.jax.contact import ContactPointMass1D
    from omnibias.control.ocp import DifferentiableEnvironment

    env = ContactPointMass1D()
    assert isinstance(env, DifferentiableEnvironment)
    y = jnp.array([1.0, 0.0])
    u = jnp.array([0.0])
    y1 = env.step(y, u)
    assert np.all(np.isfinite(np.asarray(y1)))
    assert float(env.cost(y, u)) >= 0.0
    assert float(env.terminal_cost(y1)) >= 0.0


def test_contact_point_mass_satisfies_protocol_torch():
    import torch
    from omnibias.control.ocp import DifferentiableEnvironment
    from omnibias.control.torch.contact import ContactPointMass1D

    torch.set_default_dtype(torch.float64)
    env = ContactPointMass1D()
    assert isinstance(env, DifferentiableEnvironment)
    y = torch.tensor([1.0, 0.0])
    u = torch.tensor([0.0])
    y1 = env.step(y, u)
    assert torch.all(torch.isfinite(y1))


def test_contact_force_smooth_matches_core_forward_value_jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.contact import contact_force_smooth
    from omnibias.core.contact_smoothing import contact_force_smooth as core_smooth

    for z in (-0.5, 0.0, 0.3, 1.2):
        tensor_val = float(contact_force_smooth(jnp.array(z, dtype=jnp.float64), beta=5.0, f_max=2.0))
        core_val = core_smooth(z, beta=5.0, f_max=2.0)
        assert abs(tensor_val - core_val) < 1e-10


def test_contact_force_smooth_matches_core_forward_value_torch():
    import torch
    from omnibias.control.torch.contact import contact_force_smooth
    from omnibias.core.contact_smoothing import contact_force_smooth as core_smooth

    torch.set_default_dtype(torch.float64)
    for z in (-0.5, 0.0, 0.3, 1.2):
        tensor_val = float(contact_force_smooth(torch.tensor(z), beta=5.0, f_max=2.0))
        core_val = core_smooth(z, beta=5.0, f_max=2.0)
        assert abs(tensor_val - core_val) < 1e-10


def test_contact_certificate_matches_core_bias_bound_jax():
    from omnibias.control.jax.contact import contact_certificate
    from omnibias.core.contact_smoothing import contact_smoothing_bias_bound

    report = contact_certificate(z_min=0.1, beta=10.0, f_max=1.0)
    core_report = contact_smoothing_bias_bound(z_min=0.1, beta=10.0, f_max=1.0)
    assert report.bound.hi == core_report.bound.hi
    assert report.certified == core_report.certified


def test_contact_point_mass_torch_jax_parity():
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.control.jax.contact import ContactPointMass1D as JaxEnv
    from omnibias.control.torch.contact import ContactPointMass1D as TorchEnv

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    jenv, tenv = JaxEnv(), TorchEnv()
    y0, u0 = [0.8, -0.2], [0.5]
    jy = jenv.step(jnp.array(y0), jnp.array(u0))
    ty = tenv.step(torch.tensor(y0), torch.tensor(u0))
    assert np.allclose(np.asarray(jy), ty.numpy(), atol=1e-10)


def test_contact_environment_trains_via_exact_adjoint_jax():
    """Full end-to-end smoke: the contact env is differentiable through env_step_jacobians."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.contact import ContactPointMass1D
    from omnibias.control.jax.policy import actor_adjoint_step

    env = ContactPointMass1D()
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    w1 = 0.1 * jax.random.normal(k1, (8, 2))
    b1 = jnp.zeros(8)
    w2 = 0.1 * jax.random.normal(k2, (1, 8))
    b2 = jnp.zeros(1)
    layers = [(w1, b1, "tanh"), (w2, b2, None)]
    y0 = jnp.array([1.0, 0.0])
    result = actor_adjoint_step(env, layers, y0, horizon=5, lr=0.01)
    assert np.all(np.isfinite(result.diagnostics["grad_theta"]))

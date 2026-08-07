# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""FNO smoke + DeepONet data/loss against the MOL heat reference (torch + jax)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.operator.jax import (
    data_loss as jax_data_loss,
)
from omnibias.pinn.operator.jax import (
    heat_residual_loss as jax_heat_residual_loss,
)
from omnibias.pinn.operator.jax import (
    heat_residual_loss_fd as jax_heat_residual_loss_fd,
)
from omnibias.pinn.operator.jax import (
    make_deeponet,
    make_fno1d,
)
from omnibias.pinn.operator.jax import (
    make_heat_slab as jax_make_heat_slab,
)
from omnibias.pinn.operator.torch import (
    build_deeponet,
    build_fno1d,
    data_loss,
    heat_residual_loss,
    heat_residual_loss_fd,
    make_heat_slab,
)
from omnibias.pinn.torch import ops as tops

jax.config.update("jax_enable_x64", True)

DTYPE = torch.float64
DIFFUSIVITY = 0.1
RTOL = 1e-11


def test_fno1d_forward_shape_torch() -> None:
    fno = build_fno1d(modes=4, width=8, n_layers=2, dtype=DTYPE)
    u0 = torch.randn(3, 32, dtype=DTYPE)
    out = fno(u0)
    assert out.shape == (3, 32, 1)


def test_fno1d_forward_shape_jax() -> None:
    fno = make_fno1d(modes=4, width=8, n_layers=2, seed=0)
    u0 = jax.random.normal(jax.random.PRNGKey(0), (3, 32), dtype=jnp.float64)
    out = fno(u0)
    assert out.shape == (3, 32, 1)


def test_heat_slab_mol_matches_exact_fourier_evolution_torch() -> None:
    """MOL heat snapshots match the exact Fourier mode decay of the IC."""
    D = DIFFUSIVITY
    t_final = 0.2
    n_times = 21
    slab = make_heat_slab(
        n_samples=2,
        n_grid=32,
        n_sensors=16,
        n_modes=2,
        diffusivity=D,
        t_final=t_final,
        n_times=n_times,
        seed=0,
        dtype=DTYPE,
    )
    assert slab.sensors.shape == (2, 16)
    assert slab.values.shape[0] == 2
    assert slab.coords.shape[1] == 2
    n_x = slab.grid.n
    u = slab.values[..., 0].reshape(2, n_times, n_x)
    k = slab.grid.k
    for f in range(2):
        u0_hat = torch.fft.fft(u[f, 0])
        exact = torch.fft.ifft(u0_hat * torch.exp(-D * k**2 * t_final)).real
        rel = float(torch.linalg.norm(u[f, -1] - exact) / torch.linalg.norm(exact))
        assert rel < 1e-8, (f, rel)


def test_heat_residual_loss_matches_hand_assembled_ops_torch() -> None:
    torch.manual_seed(1)
    slab = make_heat_slab(
        n_samples=2,
        n_grid=32,
        n_sensors=16,
        n_modes=2,
        diffusivity=DIFFUSIVITY,
        t_final=0.2,
        n_times=5,
        seed=1,
        dtype=DTYPE,
    )
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t")),
        components=ComponentSpec(("u",)),
        n_sensors=16,
        trunk_width=8,
        trunk_hidden=12,
        trunk_depth=2,
        branch_hidden=12,
        branch_depth=2,
        jet_order=2,
        dtype=DTYPE,
    )
    loss = heat_residual_loss(
        op, slab.sensors, slab.coords, diffusivity=DIFFUSIVITY
    )
    field = op.condition(slab.sensors)
    state = field.on_grid(slab.coords)
    u_t = tops.derivative(state, "u", axis=1, order=1)
    u_xx = tops.derivative(state, "u", axis=0, order=2)
    hand = torch.mean((u_t - DIFFUSIVITY * u_xx) ** 2)
    assert torch.allclose(loss, hand, atol=0.0, rtol=0.0)
    assert torch.isfinite(data_loss(op, slab))
    # FD helper must also be finite.
    dt = float(torch.unique(slab.coords[:, 1], sorted=True).diff().min())
    assert torch.isfinite(
        heat_residual_loss_fd(
            op, slab.sensors, slab.coords, diffusivity=DIFFUSIVITY, dt=dt
        )
    )


def test_heat_slab_and_losses_jax() -> None:
    slab = jax_make_heat_slab(
        n_samples=2,
        n_grid=32,
        n_sensors=16,
        n_modes=2,
        diffusivity=DIFFUSIVITY,
        t_final=0.2,
        n_times=5,
        seed=0,
    )
    assert slab.sensors.shape == (2, 16)
    op = make_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t")),
        components=ComponentSpec(("u",)),
        n_sensors=16,
        trunk_width=8,
        trunk_hidden=12,
        trunk_depth=2,
        branch_hidden=12,
        branch_depth=2,
        jet_order=2,
        seed=0,
    )
    loss_d = jax_data_loss(op, slab)
    loss_r = jax_heat_residual_loss(
        op, slab.sensors, slab.coords, diffusivity=DIFFUSIVITY
    )
    assert math.isfinite(float(loss_d))
    assert math.isfinite(float(loss_r))
    times = jnp.unique(slab.coords[:, 1])
    dt = float(times[1] - times[0])
    assert math.isfinite(
        float(
            jax_heat_residual_loss_fd(
                op, slab.sensors, slab.coords, diffusivity=DIFFUSIVITY, dt=dt
            )
        )
    )


def test_heat_slab_parity_torch_jax() -> None:
    t_slab = make_heat_slab(
        n_samples=3,
        n_grid=32,
        n_sensors=16,
        n_modes=2,
        diffusivity=DIFFUSIVITY,
        t_final=0.2,
        n_times=5,
        seed=7,
        dtype=DTYPE,
    )
    j_slab = jax_make_heat_slab(
        n_samples=3,
        n_grid=32,
        n_sensors=16,
        n_modes=2,
        diffusivity=DIFFUSIVITY,
        t_final=0.2,
        n_times=5,
        seed=7,
    )
    np.testing.assert_allclose(
        t_slab.sensors.detach().numpy(),
        np.asarray(j_slab.sensors),
        rtol=RTOL,
        atol=0.0,
    )
    np.testing.assert_allclose(
        t_slab.values.detach().numpy(),
        np.asarray(j_slab.values),
        rtol=RTOL,
        atol=0.0,
    )
    np.testing.assert_allclose(
        t_slab.coords.detach().numpy(),
        np.asarray(j_slab.coords),
        rtol=RTOL,
        atol=0.0,
    )

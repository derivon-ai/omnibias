# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity tests for TISE / TDSE / NLS residuals.

Asserts torch and jax produce numerically identical residual tensors
(rtol=1e-9, atol=1e-12) when fed matching weights / coords through a
:class:`OneLayerVectorField` carrying a ``(psi_re, psi_im)`` group.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField as JOne
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField as TOne
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.jax import equations as jeq
from omnibias.qpinn.torch import equations as teq

from .conftest import _allclose


def _build_torch_psi_field(shared, axes, riccati):
    coord = CoordinateSpec(axes=axes)
    components = make_psi_components(name="psi")
    field = TOne(
        coordinate_spec=coord, components=components,
        hidden=shared["H"], base=riccati, dtype=torch.float64,
    )
    with torch.no_grad():
        field.W.weight.copy_(torch.from_numpy(shared["W"]))
        field.W.bias.copy_(torch.from_numpy(shared["beta"]))
        field.c.weight.copy_(torch.from_numpy(shared["c"]))
        field.c.bias.copy_(torch.from_numpy(shared["b"]))
    coords = torch.from_numpy(shared["coords"]).clone()
    return field, coords


def _build_jax_psi_field(shared, axes, riccati):
    coord = CoordinateSpec(axes=axes)
    components = make_psi_components(name="psi")
    spec = jax_get_activation(riccati)
    field = JOne(
        coordinate_spec=coord, components=components, spec=spec,
        W=jnp.asarray(shared["W"]),
        beta=jnp.asarray(shared["beta"]),
        c=jnp.asarray(shared["c"]),
        b=jnp.asarray(shared["b"]),
        hidden=shared["H"],
    )
    coords = jnp.asarray(shared["coords"])
    return field, coords


def _harmonic_V(s):
    """The 1D harmonic potential ``V(x) = 1/2 x^2`` for both backends."""
    return 0.5 * s.coords[..., 0] ** 2


def test_tise_residual_parity(riccati, shared_psi_params_1d):
    t_field, t_coords = _build_torch_psi_field(shared_psi_params_1d, ("x",), riccati)
    j_field, j_coords = _build_jax_psi_field(shared_psi_params_1d, ("x",), riccati)
    t_state = t_field(t_coords)
    j_state = j_field(j_coords)
    t_res = teq.tise(t_state, energy=0.5, potential=_harmonic_V).residual
    j_res = jeq.tise(j_state, energy=0.5, potential=_harmonic_V).residual
    assert _allclose(t_res, j_res)


def test_tdse_residual_parity(riccati, shared_psi_params):
    t_field, t_coords = _build_torch_psi_field(shared_psi_params, ("x", "t"), riccati)
    j_field, j_coords = _build_jax_psi_field(shared_psi_params, ("x", "t"), riccati)
    t_state = t_field(t_coords)
    j_state = j_field(j_coords)
    t_res = teq.tdse(t_state, hbar=1.0, mass=1.0, potential=_harmonic_V).residual
    j_res = jeq.tdse(j_state, hbar=1.0, mass=1.0, potential=_harmonic_V).residual
    assert _allclose(t_res, j_res)


def test_nls_residual_parity(riccati, shared_psi_params):
    t_field, t_coords = _build_torch_psi_field(shared_psi_params, ("x", "t"), riccati)
    j_field, j_coords = _build_jax_psi_field(shared_psi_params, ("x", "t"), riccati)
    t_state = t_field(t_coords)
    j_state = j_field(j_coords)
    t_res = teq.nls(t_state, g=2.0, hbar=1.0, mass=1.0, potential=_harmonic_V).residual
    j_res = jeq.nls(j_state, g=2.0, hbar=1.0, mass=1.0, potential=_harmonic_V).residual
    assert _allclose(t_res, j_res)


def test_tise_with_quadrature_parity(riccati, shared_psi_params_1d):
    """When quadrature weights are passed, the energy_estimate must
    match too."""
    t_field, t_coords = _build_torch_psi_field(shared_psi_params_1d, ("x",), riccati)
    j_field, j_coords = _build_jax_psi_field(shared_psi_params_1d, ("x",), riccati)
    t_state = t_field(t_coords)
    j_state = j_field(j_coords)
    B = shared_psi_params_1d["coords"].shape[0]
    w_np = np.full((B,), 4.0 / B, dtype=np.float64)
    t_res = teq.tise(
        t_state, energy=0.0, potential=_harmonic_V,
        quadrature_weights=torch.from_numpy(w_np),
    )
    j_res = jeq.tise(
        j_state, energy=0.0, potential=_harmonic_V,
        quadrature_weights=jnp.asarray(w_np),
    )
    assert _allclose(t_res.residual, j_res.residual)
    assert _allclose(t_res.energy_estimate, j_res.energy_estimate)


def _trap_2d(s):
    """2D isotropic harmonic trap ``V(x, y) = 1/2 (x^2 + y^2)`` for both backends."""
    return 0.5 * (s.coords[..., 0] ** 2 + s.coords[..., 1] ** 2)


def test_rotating_nls_residual_parity(riccati, shared_psi_params_2d):
    """omnibias.qpinn.{torch,jax}.equations.RotatingNLS must produce
    bit-identical residuals at ``rtol=1e-9, atol=1e-12``.
    """
    t_field, t_coords = _build_torch_psi_field(shared_psi_params_2d, ("x", "y"), riccati)
    j_field, j_coords = _build_jax_psi_field(shared_psi_params_2d, ("x", "y"), riccati)
    t_state = t_field(t_coords)
    j_state = j_field(j_coords)
    t_out = teq.rotating_nls(
        t_state, g=1.5, omega_rot=0.7, mu=2.1,
        hbar=1.0, mass=1.0, potential=_trap_2d,
    )
    j_out = jeq.rotating_nls(
        j_state, g=1.5, omega_rot=0.7, mu=2.1,
        hbar=1.0, mass=1.0, potential=_trap_2d,
    )
    assert _allclose(t_out.residual, j_out.residual)
    assert _allclose(t_out.density, j_out.density)


def test_rotating_nls_zero_rotation_collapses_to_static_gpe(riccati, shared_psi_params_2d):
    """Setting ``omega_rot = 0`` must reproduce the stationary NLS
    residual ``H psi + g|psi|^2 psi - mu psi`` (built by hand).
    """
    t_field, t_coords = _build_torch_psi_field(shared_psi_params_2d, ("x", "y"), riccati)
    state = t_field(t_coords)
    out = teq.rotating_nls(
        state, g=2.0, omega_rot=0.0, mu=1.0, potential=_trap_2d,
    )
    # Hand build the residual: H psi + g rho psi - mu psi (no rotation).
    psi_re = state.ops.value(state, "psi_re")
    psi_im = state.ops.value(state, "psi_im")
    lap_re = state.ops.laplacian(state, "psi_re")
    lap_im = state.ops.laplacian(state, "psi_im")
    V = _trap_2d(state)
    rho = psi_re * psi_re + psi_im * psi_im
    expected_re = -0.5 * lap_re + V * psi_re + 2.0 * rho * psi_re - 1.0 * psi_re
    expected_im = -0.5 * lap_im + V * psi_im + 2.0 * rho * psi_im - 1.0 * psi_im
    assert torch.allclose(out.residual[..., 0], expected_re, rtol=1e-12, atol=1e-14)
    assert torch.allclose(out.residual[..., 1], expected_im, rtol=1e-12, atol=1e-14)

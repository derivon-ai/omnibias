# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for conservation cages on the JAX backend."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.cage import (
    energy_conserving_advection,
    enstrophy_conserving_advection,
    make_hard_boundary_field,
    make_mass_flux_potential_field,
    make_streamfunction_field,
)
from omnibias.pinn.jax.fields.spectral import make_spectral_vector_field


def _randomise_spectral(field, rng):
    leaves, td = jax.tree_util.tree_flatten(field)
    new_leaves = list(leaves)
    new_leaves[0] = jnp.asarray(rng.normal(size=field.W_t.shape))
    new_leaves[1] = jnp.asarray(rng.normal(scale=0.1, size=field.beta_t.shape))
    new_leaves[-2] = jnp.asarray(rng.normal(scale=0.1, size=field.V.shape))
    new_leaves[-1] = jnp.asarray(rng.normal(scale=0.1, size=field.b_t.shape))
    return jax.tree_util.tree_unflatten(td, new_leaves)


def _make_div_free_state():
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("psi", "p"))
    base = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=6, time_depth=1, activation="tanh", seed=0,
    )
    base = _randomise_spectral(base, np.random.default_rng(123))
    cage = make_streamfunction_field(
        base=base, psi="psi", velocity_names=("u", "v"),
        passthrough_names=("p",),
    )
    coords = jnp.array([
        [0.0, 0.1, 0.2],
        [0.5, 0.7, 0.5],
        [1.0, 1.3, 1.0],
    ], dtype=jnp.float64)
    return cage(coords)


def _make_compressible_state():
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")})
    field = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=6, time_depth=1, activation="tanh", seed=1,
    )
    field = _randomise_spectral(field, np.random.default_rng(456))
    coords = jnp.array([
        [0.0, 0.1, 0.2],
        [0.5, 0.7, 0.5],
        [1.0, 1.3, 1.0],
    ], dtype=jnp.float64)
    return field(coords)


def test_skew_advection_div_free_equals_standard():
    state = _make_div_free_state()
    standard = jops.advection(state, velocity=("u", "v"))
    skew = energy_conserving_advection(state, velocity=("u", "v"))
    assert np.allclose(np.asarray(standard), np.asarray(skew),
                       rtol=1e-12, atol=1e-12)


def test_skew_advection_general_formula():
    state = _make_compressible_state()
    standard = jops.advection(state, velocity=("u", "v"))
    skew = energy_conserving_advection(state, velocity=("u", "v"))
    div_u = jops.divergence(state, ("u", "v"))
    u_v = jops.stack_components(state, ("u", "v"))
    expected_correction = 0.5 * div_u[..., None] * u_v
    assert np.allclose(
        np.asarray(skew - standard), np.asarray(expected_correction),
        rtol=1e-12, atol=1e-12,
    )


def test_enstrophy_advection_div_free_equals_standard():
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("psi", "omega"))
    base = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=6, time_depth=1, activation="tanh", seed=0,
    )
    base = _randomise_spectral(base, np.random.default_rng(789))
    cage = make_streamfunction_field(
        base=base, psi="psi", velocity_names=("u", "v"),
        passthrough_names=("omega",),
    )
    coords = jnp.array([
        [0.0, 0.1, 0.2],
        [0.5, 0.7, 0.5],
        [1.0, 1.3, 1.0],
    ], dtype=jnp.float64)
    state = cage(coords)
    std = jops.advection(state, velocity=("u", "v"), scalar="omega")
    skew = enstrophy_conserving_advection(
        state, velocity=("u", "v"), vorticity="omega",
    )
    assert np.allclose(np.asarray(std), np.asarray(skew),
                       rtol=1e-12, atol=1e-12)


# --- HardBoundary ---------------------------------------------------


def test_hard_boundary_zero_value_on_boundary():
    cspec = CoordinateSpec(
        axes=("t", "x", "y"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    )
    mspec = ComponentSpec(("u",))
    base = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=4, time_hidden=4, time_depth=1, activation="tanh", seed=0,
    )
    base = _randomise_spectral(base, np.random.default_rng(11))

    def distance(coords):
        x, y = coords[..., 1], coords[..., 2]
        return (1.0 - x ** 2) * (1.0 - y ** 2)

    cage = make_hard_boundary_field(
        base=base, distance_fn=distance,
        boundary_value_fn=None,
        bounded_names=("u",),
    )
    boundary_coords = jnp.array([
        [0.5, 1.0, 0.3],
        [0.5, -1.0, 0.7],
        [0.5, 0.2, 1.0],
        [0.5, -0.7, -1.0],
    ], dtype=jnp.float64)
    state = cage(boundary_coords)
    val = state.u.value
    assert float(jnp.max(jnp.abs(val))) < 1e-13


def test_hard_boundary_g_recovered():
    cspec = CoordinateSpec(
        axes=("t", "x"),
        time_axis="t",
        domain=((0.0, 1.0), (-1.0, 1.0)),
    )
    mspec = ComponentSpec(("u",))
    base = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=3, time_hidden=4, time_depth=1, activation="tanh", seed=0,
    )
    base = _randomise_spectral(base, np.random.default_rng(22))

    def distance(coords):
        return 1.0 - coords[..., 1] ** 2

    def g(coords):
        return {"u": jnp.sin(coords[..., 0]) * coords[..., 1]}

    cage = make_hard_boundary_field(
        base=base, distance_fn=distance,
        boundary_value_fn=g, bounded_names=("u",),
    )
    boundary_coords = jnp.array([
        [0.0, 1.0],
        [0.5, 1.0],
        [0.5, -1.0],
    ], dtype=jnp.float64)
    state = cage(boundary_coords)
    expected = jnp.array([
        jnp.sin(jnp.float64(0.0)) * 1.0,
        jnp.sin(jnp.float64(0.5)) * 1.0,
        jnp.sin(jnp.float64(0.5)) * (-1.0),
    ])
    assert np.allclose(np.asarray(state.u.value), np.asarray(expected),
                       atol=1e-13)


# --- MassFlux ----------------------------------------------------


def test_mass_flux_div_zero():
    cspec = CoordinateSpec(
        axes=("t", "x", "y", "z"),
        periodicity=(False, True, True, True),
        time_axis="t",
    )
    mspec = ComponentSpec(("Psi1", "Psi2", "Psi3", "rho"))
    base = make_spectral_vector_field(
        coordinate_spec=cspec, components=mspec,
        K=3, time_hidden=4, time_depth=1, activation="tanh", seed=0,
    )
    base = _randomise_spectral(base, np.random.default_rng(33))
    cage = make_mass_flux_potential_field(base=base)
    coords = jnp.array([
        [0.0, 0.1, 0.2, -0.5],
        [0.5, 0.7, 0.5, 0.0],
        [1.0, 1.3, 1.0, 1.0],
    ], dtype=jnp.float64)
    state = cage(coords)
    div = jops.divergence(state, ("rhou", "rhov", "rhow"))
    assert float(jnp.max(jnp.abs(div))) < 1e-12

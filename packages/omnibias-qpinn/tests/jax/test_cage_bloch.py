# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the Bloch-periodic cage (jax)."""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import make_one_layer_vector_field
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.jax.cage import BlochPeriodicField, make_bloch_periodic_field


def _build_base():
    coord = CoordinateSpec(axes=("x",))
    spec = make_psi_components(name="u")
    return make_one_layer_vector_field(
        coordinate_spec=coord, components=spec,
        hidden=8, base="gaussian", dtype=jnp.float64, seed=0,
    )


class TestBlochConstructor:
    def test_builds(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[1.5])
        assert isinstance(cage, BlochPeriodicField)
        assert cage.components.names == ("psi_re", "psi_im")

    def test_rejects_wrong_k_shape(self):
        base = _build_base()
        with pytest.raises(ValueError, match="k must have shape"):
            make_bloch_periodic_field(base=base, k=[1.5, 2.0])


class TestBlochValueAtZeroK:
    def test_value_matches_base_at_k0(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[0.0])
        coords = jnp.linspace(-2.0, 2.0, 11, dtype=jnp.float64)[:, None]
        base_state = base(coords)
        cage_state = cage(coords)
        assert jnp.allclose(
            cage_state.ops.value(cage_state, "psi_re"),
            base_state.ops.value(base_state, "u_re"),
        )

    def test_derivative_matches_base_at_k0(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[0.0])
        coords = jnp.linspace(-2.0, 2.0, 11, dtype=jnp.float64)[:, None]
        base_state = base(coords)
        cage_state = cage(coords)
        assert jnp.allclose(
            cage_state.ops.derivative(cage_state, "psi_re", axis=0, order=2),
            base_state.ops.derivative(base_state, "u_re", axis=0, order=2),
        )


class TestBlochDensity:
    def test_density_independent_of_k(self):
        base = _build_base()
        coords = jnp.linspace(-2.0, 2.0, 11, dtype=jnp.float64)[:, None]
        cage_a = make_bloch_periodic_field(base=base, k=[0.0])
        cage_b = make_bloch_periodic_field(base=base, k=[2.5])
        a = cage_a(coords)
        b = cage_b(coords)
        d_a = (a.ops.value(a, "psi_re") ** 2 + a.ops.value(a, "psi_im") ** 2)
        d_b = (b.ops.value(b, "psi_re") ** 2 + b.ops.value(b, "psi_im") ** 2)
        assert jnp.allclose(d_a, d_b)


class TestBlochPytree:
    def test_round_trip(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[1.5])
        leaves, treedef = jax.tree_util.tree_flatten(cage)
        cage2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(cage2, BlochPeriodicField)
        assert cage2.components.names == cage.components.names

    def test_jit(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[1.5])

        @jax.jit
        def value(cage_, x):
            state = cage_(x)
            return state.ops.value(state, "psi_re")

        coords = jnp.linspace(-2.0, 2.0, 11, dtype=jnp.float64)[:, None]
        out = value(cage, coords)
        assert jnp.all(jnp.isfinite(out))

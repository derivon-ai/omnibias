# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the L^2-norm-conservation cage (jax backend)."""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import make_one_layer_vector_field
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.jax.cage import (
    NormConservationField,
    make_norm_conservation_field,
    norm_loss,
)
from omnibias.qpinn.jax.diagnostics import norm_squared


def _build_base_1d():
    coord = CoordinateSpec(axes=("x",))
    spec = make_psi_components(name="psi")
    return make_one_layer_vector_field(
        coordinate_spec=coord, components=spec, hidden=16,
        base="gaussian", dtype=jnp.float64, seed=0,
    ), coord


class TestNormCageConstructor:
    def test_builds_with_uniform_quadrature(self):
        base, _ = _build_base_1d()
        x = jnp.linspace(-3.0, 3.0, 401, dtype=jnp.float64)[:, None]
        w = jnp.full((401,), 6.0 / 401, dtype=jnp.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x, quadrature_weights=w,
        )
        assert isinstance(cage, NormConservationField)
        assert cage.components.names == ("psi_re", "psi_im")

    def test_rejects_bad_weights_shape(self):
        base, _ = _build_base_1d()
        x = jnp.zeros((5, 1), dtype=jnp.float64)
        w = jnp.ones((5, 1), dtype=jnp.float64)
        with pytest.raises(ValueError, match="1D"):
            make_norm_conservation_field(
                base=base, quadrature_coords=x, quadrature_weights=w,
            )

    def test_rejects_negative_weights(self):
        base, _ = _build_base_1d()
        x = jnp.zeros((5, 1), dtype=jnp.float64)
        w = jnp.array([1.0, -0.1, 1.0, 1.0, 1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            make_norm_conservation_field(
                base=base, quadrature_coords=x, quadrature_weights=w,
            )


class TestNormCageNumerics:
    def test_caged_field_has_unit_norm(self):
        base, _ = _build_base_1d()
        x = jnp.linspace(-3.0, 3.0, 401, dtype=jnp.float64)[:, None]
        w = jnp.full((401,), 6.0 / 401, dtype=jnp.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x, quadrature_weights=w,
        )
        state = cage(x)
        nrm = norm_squared(state, quadrature_weights=w)
        assert math.isclose(float(nrm), 1.0, rel_tol=1e-12, abs_tol=1e-12)

    def test_pytree_round_trip(self):
        """The cage must survive jax.tree_util pytree flatten/unflatten."""
        base, _ = _build_base_1d()
        x = jnp.linspace(-3.0, 3.0, 401, dtype=jnp.float64)[:, None]
        w = jnp.full((401,), 6.0 / 401, dtype=jnp.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x, quadrature_weights=w,
        )
        leaves, treedef = jax.tree_util.tree_flatten(cage)
        cage2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(cage2, NormConservationField)
        assert cage2.components.names == cage.components.names

    def test_jit_compatible(self):
        """The cage forward pass must be JIT-able."""
        base, _ = _build_base_1d()
        x = jnp.linspace(-3.0, 3.0, 401, dtype=jnp.float64)[:, None]
        w = jnp.full((401,), 6.0 / 401, dtype=jnp.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x, quadrature_weights=w,
        )

        @jax.jit
        def loss(cage_):
            state = cage_(x)
            return norm_squared(state, quadrature_weights=w)

        nrm = loss(cage)
        assert math.isclose(float(nrm), 1.0, rel_tol=1e-12, abs_tol=1e-12)


class TestSoftNormLoss:
    def test_zero_when_normalised(self):
        base, _ = _build_base_1d()
        x = jnp.linspace(-3.0, 3.0, 401, dtype=jnp.float64)[:, None]
        w = jnp.full((401,), 6.0 / 401, dtype=jnp.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x, quadrature_weights=w,
        )
        state = cage(x)
        loss = norm_loss(state, quadrature_weights=w)
        assert float(loss) < 1e-20

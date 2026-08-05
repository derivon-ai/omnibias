# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for Helmholtz / Klein-Gordon / Dirac residuals (jax)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import make_one_layer_vector_field
from omnibias.qpinn import (
    make_psi_components,
    make_spinor_components,
)
from omnibias.qpinn.jax.equations import (
    Dirac,
    DiracOutput,
    Helmholtz,
    HelmholtzOutput,
    KleinGordon,
    KleinGordonOutput,
    dirac,
    helmholtz,
    klein_gordon,
)


class TestHelmholtz:
    def _build_field(self):
        coord = CoordinateSpec(axes=("x", "y"))
        spec = make_psi_components(name="psi")
        return make_one_layer_vector_field(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=jnp.float64, seed=0,
        )

    def test_shape(self):
        field = self._build_field()
        coords = jax.random.normal(jax.random.PRNGKey(1), (20, 2), dtype=jnp.float64)
        state = field(coords)
        out = Helmholtz(k=2.0)(state)
        assert isinstance(out, HelmholtzOutput)
        assert out.residual.shape == (20, 2)

    def test_function_form(self):
        field = self._build_field()
        coords = jax.random.normal(jax.random.PRNGKey(1), (20, 2), dtype=jnp.float64)
        state = field(coords)
        out_cls = Helmholtz(k=2.0)(state)
        out_fn = helmholtz(state, k=2.0)
        assert jnp.allclose(out_cls.residual, out_fn.residual)


class TestKleinGordon:
    def _build_field(self):
        coord = CoordinateSpec(axes=("x", "t"))
        spec = ComponentSpec(("phi",))
        return make_one_layer_vector_field(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=jnp.float64, seed=0,
        )

    def test_shape(self):
        field = self._build_field()
        coords = jax.random.normal(jax.random.PRNGKey(1), (20, 2), dtype=jnp.float64)
        state = field(coords)
        out = KleinGordon(mass=1.0)(state)
        assert isinstance(out, KleinGordonOutput)
        assert out.residual.shape == (20,)

    def test_phi4_changes_residual(self):
        field = self._build_field()
        coords = jax.random.normal(jax.random.PRNGKey(1), (20, 2), dtype=jnp.float64)
        state = field(coords)
        out_a = KleinGordon(mass=1.0, lambda_phi4=0.0)(state)
        out_b = KleinGordon(mass=1.0, lambda_phi4=0.5)(state)
        assert not jnp.allclose(out_a.residual, out_b.residual)

    def test_requires_time(self):
        coord = CoordinateSpec(axes=("x",))
        spec = ComponentSpec(("phi",))
        field = make_one_layer_vector_field(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=jnp.float64, seed=0,
        )
        coords = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float64)[:, None]
        state = field(coords)
        with pytest.raises(ValueError, match="time axis"):
            KleinGordon()(state)


class TestDirac:
    def _build_field(self, axes=("x", "y", "z", "t")):
        coord = CoordinateSpec(axes=axes)
        spec = make_spinor_components(name="spinor", n_components=4)
        return make_one_layer_vector_field(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=jnp.float64, seed=0,
        )

    def test_shape(self):
        field = self._build_field()
        coords = jax.random.normal(jax.random.PRNGKey(1), (16, 4), dtype=jnp.float64)
        state = field(coords)
        out = Dirac(mass=1.0)(state)
        assert isinstance(out, DiracOutput)
        assert out.residual.shape == (16, 8)

    def test_representation_changes_residual(self):
        field = self._build_field()
        coords = jax.random.normal(jax.random.PRNGKey(1), (16, 4), dtype=jnp.float64)
        state = field(coords)
        out_d = Dirac(mass=1.0, representation="dirac")(state)
        out_w = Dirac(mass=1.0, representation="weyl")(state)
        assert not jnp.allclose(out_d.residual, out_w.residual)

    def test_lower_dim(self):
        field = self._build_field(axes=("x", "t"))
        coords = jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float64)
        state = field(coords)
        out = Dirac(mass=1.0)(state)
        assert out.residual.shape == (16, 8)

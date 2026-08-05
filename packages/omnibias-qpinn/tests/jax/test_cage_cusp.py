# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the nuclear-cusp cage (jax backend).

Three independent correctness oracles:

1. **Autodiff cross-check**: the cage's closed-form value / gradient /
   Laplacian / mixed-partial of ``C(r) psi_base(r)`` match ``jax.grad`` /
   ``jax.hessian`` of the same product.
2. **Analytic H-atom**: with a constant base and ``rates=0`` the caged
   wavefunction is exactly ``exp(-Z s)``, whose local energy is ``-Z^2/2``
   everywhere -- the hydrogenic ground-state eigenvalue.
3. **Kato cusp**: the realised electron-nucleus slope is ``u'(0) = -Z``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import make_one_layer_vector_field
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.jax.cage.cusp import (
    NuclearCuspField,
    make_nuclear_cusp_field,
    nuclear_cusp_slope,
)


def _build_base_3d(seed: int = 0):
    coord = CoordinateSpec(axes=("x", "y", "z"))
    spec = make_psi_components(name="psi")
    return make_one_layer_vector_field(
        coordinate_spec=coord, components=spec, hidden=8,
        base="tanh", dtype=jnp.float64, seed=seed,
    )


def _cusp_C(r, nuclei, charges, rates):
    tot = jnp.asarray(0.0)
    for a in range(nuclei.shape[0]):
        s = jnp.sqrt(jnp.sum((r - nuclei[a]) ** 2) + 1e-30)
        tot = tot + (-charges[a] * s / (1.0 + rates[a] * s))
    return jnp.exp(tot)


class TestCuspCageConstructor:
    def test_builds_and_keeps_component_names(self):
        base = _build_base_3d()
        cage = make_nuclear_cusp_field(
            base=base,
            nuclei=jnp.zeros((1, 3)),
            charges=jnp.array([1.0]),
        )
        assert isinstance(cage, NuclearCuspField)
        assert cage.components.names == ("psi_re", "psi_im")

    def test_rejects_wrong_nuclei_dim(self):
        base = _build_base_3d()
        with pytest.raises(ValueError, match="nuclei must be"):
            make_nuclear_cusp_field(
                base=base, nuclei=jnp.zeros((1, 2)), charges=jnp.array([1.0])
            )

    def test_rejects_charge_count_mismatch(self):
        base = _build_base_3d()
        with pytest.raises(ValueError, match="charges must be"):
            make_nuclear_cusp_field(
                base=base, nuclei=jnp.zeros((2, 3)), charges=jnp.array([1.0])
            )


class TestCuspCageVsAutograd:
    def test_value_grad_laplacian_mixed_parity(self):
        base = _build_base_3d(seed=1)
        nuclei = jnp.array([[0.0, 0.0, 0.0], [1.2, -0.3, 0.5]])
        charges = jnp.array([1.0, 3.0])
        rates = jnp.array([0.6, 0.9])
        cage = make_nuclear_cusp_field(
            base=base, nuclei=nuclei, charges=charges, rates=rates
        )
        rng = np.random.default_rng(0)
        coords = jnp.asarray(rng.normal(size=(5, 3)))
        state = cage(coords)

        def base_val(r, name):
            st = base.evaluate(r[None, :])
            return st.ops.value(st, name)[0]

        for name in ("psi_re", "psi_im"):
            v = np.asarray(state.ops.value(state, name))
            v_ref = np.array([
                float(_cusp_C(coords[b], nuclei, charges, rates) * base_val(coords[b], name))
                for b in range(5)
            ])
            np.testing.assert_allclose(v, v_ref, atol=1e-11)

            def f(r, name=name):
                return _cusp_C(r, nuclei, charges, rates) * base_val(r, name)

            for ax in range(3):
                d = np.asarray(state.ops.derivative(state, name, axis=ax, order=1))
                d_ref = np.array([float(jax.grad(f)(coords[b])[ax]) for b in range(5)])
                np.testing.assert_allclose(d, d_ref, atol=1e-9)

            lap = np.asarray(state.ops.laplacian(state, name))
            lap_ref = np.array(
                [float(jnp.trace(jax.hessian(f)(coords[b]))) for b in range(5)]
            )
            np.testing.assert_allclose(lap, lap_ref, atol=1e-8)

            mp = np.asarray(state.ops.mixed_partial(state, name, (0, 1), (1, 1)))
            mp_ref = np.array(
                [float(jax.hessian(f)(coords[b])[0, 1]) for b in range(5)]
            )
            np.testing.assert_allclose(mp, mp_ref, atol=1e-8)


class TestHydrogenicOracle:
    @pytest.mark.parametrize("Z", [1.0, 2.0, 3.0])
    def test_constant_base_gives_exact_hydrogenic_energy(self, Z):
        base = _build_base_3d()
        # Constant base psi_re = 1, psi_im = 0 (zero readout weights).
        const_base = dataclasses.replace(
            base, c=jnp.zeros_like(base.c), b=jnp.array([1.0, 0.0])
        )
        cage = make_nuclear_cusp_field(
            base=const_base,
            nuclei=jnp.zeros((1, 3)),
            charges=jnp.array([Z]),
            rates=0.0,  # exact exp(-Z s) Slater factor
        )
        rng = np.random.default_rng(int(Z))
        coords = jnp.asarray(rng.normal(size=(6, 3)) * 0.7)
        state = cage(coords)
        psi = state.ops.value(state, "psi_re")
        lap = state.ops.laplacian(state, "psi_re")
        s = jnp.linalg.norm(coords, axis=-1)
        # exact 1s: psi = exp(-Z s)
        np.testing.assert_allclose(np.asarray(psi), np.asarray(jnp.exp(-Z * s)), atol=1e-12)
        T_L = -0.5 * lap / psi
        V = -Z / s
        E_L = T_L + V
        np.testing.assert_allclose(np.asarray(E_L), -0.5 * Z * Z, atol=1e-9)


class TestCuspCondition:
    def test_slope_equals_minus_charge(self):
        base = _build_base_3d()
        cage = make_nuclear_cusp_field(
            base=base, nuclei=jnp.zeros((2, 3)), charges=jnp.array([1.0, 6.0])
        )
        assert float(nuclear_cusp_slope(cage, 0)) == -1.0
        assert float(nuclear_cusp_slope(cage, 1)) == -6.0

    @pytest.mark.parametrize("Z", [1.0, 4.0])
    def test_numerical_radial_slope_at_nucleus(self, Z):
        """d/ds log C -> -Z as s -> 0 (finite-difference on the factor)."""
        rate = 0.7

        def log_c(s):
            return -Z * s / (1.0 + rate * s)

        h = 1e-7
        slope = (log_c(h) - log_c(0.0)) / h
        assert abs(slope - (-Z)) < 1e-5


class TestCagePytreeJit:
    def test_pytree_round_trip(self):
        base = _build_base_3d()
        cage = make_nuclear_cusp_field(
            base=base, nuclei=jnp.zeros((1, 3)), charges=jnp.array([2.0])
        )
        leaves, treedef = jax.tree_util.tree_flatten(cage)
        cage2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(cage2, NuclearCuspField)
        assert cage2.components.names == cage.components.names

    def test_jit_forward(self):
        base = _build_base_3d()
        cage = make_nuclear_cusp_field(
            base=base, nuclei=jnp.zeros((1, 3)), charges=jnp.array([1.0]), rates=0.0
        )
        coords = jnp.asarray(np.random.default_rng(3).normal(size=(4, 3)))

        @jax.jit
        def lap_sum(cage_):
            st = cage_(coords)
            return st.ops.laplacian(st, "psi_re").sum()

        val = lap_sum(cage)
        assert np.isfinite(float(val))

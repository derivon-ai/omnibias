# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form Jastrow correlation factor: parity + cusp tests.

Three independent correctness oracles for the Pade-Jastrow factor
``exp(J)`` with ``J = sum_{i<j} u_ee(r_ij) + sum_{i,a} u_en(r_ia)``:

1. **Independent code path**: the closed-form value equals a pure-numpy
   double loop (:func:`jastrow_log_value_np`).
2. **Autodiff cross-check**: closed-form ``(grad, lap) J`` and the combined
   Jastrow-Slater local kinetic energy match ``jax.grad`` / ``jax.hessian``.
3. **Analytic cusp**: the coalescence slopes equal the Kato cusp values
   ``u_ee'(0) = 1/2`` and ``u_en'(0) = -Z_a`` by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.ferminet.jastrow import (  # noqa: E402
    JastrowParams,
    electron_electron_cusp,
    electron_nucleus_cusp,
    jastrow_init_params,
    jastrow_log_value,
    jastrow_log_value_np,
    jastrow_slater_local_kinetic_energy,
    jastrow_value_grad_laplacian,
)
from omnibias.ferminet.restricted import (  # noqa: E402
    tier2_grad_laplacian_log_psi,
    tier2_init_params,
    tier2_local_kinetic_energy,
    tier2_log_abs_psi,
)


def _rand_walker(n_e: int, atoms: jnp.ndarray, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    n_atoms = atoms.shape[0]
    out = []
    for _ in range(n_e):
        a = rng.integers(n_atoms)
        out.append(np.asarray(atoms[a]) + 0.4 * rng.normal(size=3))
    return jnp.asarray(np.concatenate(out), dtype=jnp.float64)


def _make_params(n_atoms: int, seed: int) -> tuple[JastrowParams, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    atoms = jnp.asarray(rng.normal(size=(n_atoms, 3)), dtype=jnp.float64)
    charges = jnp.asarray(rng.uniform(1.0, 8.0, size=(n_atoms,)), dtype=jnp.float64)
    jp = jastrow_init_params(atoms, charges, a_ee=0.5, b_ee=0.8, b_en=1.3)
    return jp, charges


class TestJastrowValue:
    @pytest.mark.parametrize("n_e", [1, 2, 3, 4])
    def test_value_matches_numpy_reference(self, n_e):
        jp, _ = _make_params(n_atoms=2, seed=n_e)
        r = _rand_walker(n_e, jp.atoms, seed=100 + n_e)
        v_jet, _, _ = jastrow_value_grad_laplacian(jp, r, n_e)
        v_np = jastrow_log_value_np(jp, r, n_e)
        assert abs(float(v_jet) - v_np) < 1e-11

    def test_two_value_paths_agree(self):
        jp, _ = _make_params(n_atoms=3, seed=1)
        r = _rand_walker(4, jp.atoms, seed=2)
        v_a, _, _ = jastrow_value_grad_laplacian(jp, r, n_e=4)
        v_b = jastrow_log_value(jp, r, n_e=4)
        assert abs(float(v_a) - float(v_b)) < 1e-12

    def test_permutation_symmetry(self):
        """Swapping two electrons leaves the symmetric Jastrow value fixed."""
        jp, _ = _make_params(n_atoms=2, seed=5)
        n_e = 3
        r = _rand_walker(n_e, jp.atoms, seed=6).reshape(n_e, 3)
        v0 = jastrow_log_value(jp, r.reshape(-1), n_e)
        swapped = r.at[jnp.array([0, 1])].set(r[jnp.array([1, 0])])
        v1 = jastrow_log_value(jp, swapped.reshape(-1), n_e)
        assert abs(float(v0) - float(v1)) < 1e-13

    def test_single_electron_has_no_ee_term(self):
        """One electron: only the e-n sum survives (no pairs)."""
        jp, _ = _make_params(n_atoms=2, seed=7)
        r = _rand_walker(1, jp.atoms, seed=8)
        v, grad, _ = jastrow_value_grad_laplacian(jp, r, n_e=1)
        # e-n only: compare against the numpy reference (also e-n only here).
        assert abs(float(v) - jastrow_log_value_np(jp, r, 1)) < 1e-12
        assert grad.shape == (1, 3)


class TestJastrowDerivativesVsAutograd:
    @pytest.mark.parametrize("n_e", [2, 3, 4])
    def test_grad_and_laplacian_parity(self, n_e):
        jp, _ = _make_params(n_atoms=2, seed=n_e + 10)
        r = _rand_walker(n_e, jp.atoms, seed=n_e + 20)
        _, grad, lap = jastrow_value_grad_laplacian(jp, r, n_e)

        def jfn(x):
            return jastrow_log_value(jp, x, n_e)

        g_ref = jax.grad(jfn)(r).reshape(n_e, 3)
        lap_ref = jnp.trace(jax.hessian(jfn)(r))
        np.testing.assert_allclose(
            np.asarray(grad), np.asarray(g_ref), atol=1e-9, rtol=1e-9
        )
        np.testing.assert_allclose(float(lap), float(lap_ref), atol=1e-8, rtol=1e-8)


class TestJastrowSlaterKinetic:
    @pytest.mark.parametrize("n_e", [2, 3, 4])
    def test_combined_local_kinetic_vs_autograd(self, n_e):
        """The marquee claim: closed-form ``T_loc`` of ``e^J det M`` == autograd."""
        n_atoms = 2
        jp, _ = _make_params(n_atoms=n_atoms, seed=n_e + 30)
        t2 = tier2_init_params(
            n_atoms=n_atoms, n_orb=n_e, hidden=6, seed=n_e + 3, atoms=jp.atoms
        )
        r = _rand_walker(n_e, jp.atoms, seed=n_e + 40)

        T_cf = jastrow_slater_local_kinetic_energy(t2, jp, r, n_e)

        def log_abs(x):
            return tier2_log_abs_psi(t2, x, n_e) + jastrow_log_value(jp, x, n_e)

        g = jax.grad(log_abs)(r)
        lap = jnp.trace(jax.hessian(log_abs)(r))
        T_ref = -0.5 * (lap + jnp.sum(g * g))
        np.testing.assert_allclose(float(T_cf), float(T_ref), atol=5e-9, rtol=5e-9)

    def test_zero_jastrow_recovers_bare_slater(self):
        """With all cusp slopes zero the Jastrow is flat and ``T_loc`` reduces
        to the bare determinant kinetic energy bit-for-bit."""
        n_atoms, n_e = 2, 3
        rng = np.random.default_rng(77)
        atoms = jnp.asarray(rng.normal(size=(n_atoms, 3)), dtype=jnp.float64)
        jp = JastrowParams(
            atoms=atoms,
            a_ee=jnp.asarray(0.0),
            b_ee=jnp.asarray(1.0),
            a_en=jnp.zeros((n_atoms,)),
            b_en=jnp.ones((n_atoms,)),
        )
        t2 = tier2_init_params(
            n_atoms=n_atoms, n_orb=n_e, hidden=6, seed=9, atoms=atoms
        )
        r = _rand_walker(n_e, atoms, seed=10)
        T_jas = jastrow_slater_local_kinetic_energy(t2, jp, r, n_e)
        T_bare = tier2_local_kinetic_energy(t2, r, n_e)
        np.testing.assert_allclose(float(T_jas), float(T_bare), atol=1e-12, rtol=1e-12)


class TestCuspConditions:
    def test_electron_electron_cusp_is_one_half(self):
        jp, _ = _make_params(n_atoms=2, seed=1)
        assert abs(float(electron_electron_cusp(jp)) - 0.5) < 1e-14

    def test_electron_nucleus_cusp_is_minus_charge(self):
        jp, charges = _make_params(n_atoms=3, seed=2)
        np.testing.assert_allclose(
            np.asarray(electron_nucleus_cusp(jp)), -np.asarray(charges), atol=1e-14
        )

    def test_numerical_ee_slope_at_coalescence(self):
        """Finite-difference radial slope of ``u_ee`` at ``r -> 0`` equals ``a_ee``."""
        a_ee, b_ee = 0.5, 0.8
        h = 1e-7

        def u_ee(r):
            return a_ee * r / (1.0 + b_ee * r)

        slope = (u_ee(h) - u_ee(0.0)) / h
        assert abs(slope - a_ee) < 1e-6

    def test_numerical_en_slope_at_coalescence(self):
        """Finite-difference radial slope of ``u_en`` at ``r -> 0`` equals ``-Z``."""
        Z, b_en = 6.0, 1.3
        a_en = -Z
        h = 1e-7

        def u_en(r):
            return a_en * r / (1.0 + b_en * r)

        slope = (u_en(h) - u_en(0.0)) / h
        assert abs(slope - a_en) < 1e-5


class TestRefactorSafety:
    """The tier2 split must leave the public kinetic energy untouched."""

    @pytest.mark.parametrize("n_e", [2, 3])
    def test_grad_split_matches_autograd(self, n_e):
        t2 = tier2_init_params(n_atoms=2, n_orb=n_e, hidden=6, seed=n_e)
        r = _rand_walker(n_e, t2.atoms, seed=n_e + 1)
        grad, _ = tier2_grad_laplacian_log_psi(t2, r, n_e)

        def log_abs(x):
            return tier2_log_abs_psi(t2, x, n_e)

        g_ref = jax.grad(log_abs)(r).reshape(n_e, 3)
        np.testing.assert_allclose(
            np.asarray(grad), np.asarray(g_ref), atol=1e-9, rtol=1e-9
        )

    @pytest.mark.parametrize("n_e", [2, 3, 4])
    def test_kinetic_still_matches_autograd(self, n_e):
        t2 = tier2_init_params(n_atoms=2, n_orb=n_e, hidden=6, seed=n_e + 5)
        r = _rand_walker(n_e, t2.atoms, seed=n_e + 6)
        T = tier2_local_kinetic_energy(t2, r, n_e)

        def log_abs(x):
            return tier2_log_abs_psi(t2, x, n_e)

        g = jax.grad(log_abs)(r)
        lap = jnp.trace(jax.hessian(log_abs)(r))
        T_ref = -0.5 * (lap + jnp.sum(g * g))
        np.testing.assert_allclose(float(T), float(T_ref), atol=5e-10, rtol=5e-10)

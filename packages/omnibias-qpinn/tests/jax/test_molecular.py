# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Molecular electronic-structure local energy (jax backend).

Three independent correctness oracles:

1. **Analytic**: the hydrogenic 1s orbital ``log|psi| = -Z r`` gives a local
   energy that is *constant* and equal to ``-Z^2/2`` at every electron
   position (the exact ground-state eigenvalue); the isotropic harmonic
   oscillator ``log|psi| = -w r^2/2`` gives ``E_L = 3w/2``.
2. **Independent code path**: ``coulomb_potential`` reproduces
   :func:`omnibias.jax.coulomb_potential` bit-for-bit.
3. **Autodiff cross-check**: the closed-form jet ``(grad, lap) log|psi|``
   matches :func:`jax.grad` / :func:`jax.hessian` of the same MLP.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.jax import coulomb_potential as ref_coulomb
from omnibias.qpinn.jax import molecular as M


def _hydrogenic_derivatives(Z: float, r_vec):
    """Closed-form ``(grad, lap) log|psi|`` for ``log|psi| = -Z |r|``."""
    r_vec = jnp.asarray(r_vec)
    r = jnp.linalg.norm(r_vec)
    grad = -Z * r_vec / r
    lap = -2.0 * Z / r  # nabla^2 (-Z r) = -2 Z / r in 3D
    return grad, lap


class TestHydrogenicLocalEnergy:
    """The 1s hydrogenic orbital is the marquee exact oracle."""

    @pytest.mark.parametrize("Z", [1.0, 2.0, 3.0, 8.0])
    @pytest.mark.parametrize(
        "r_vec",
        [[0.3, 0.0, 0.0], [0.5, -0.2, 0.7], [1.3, 0.4, -0.9], [0.05, 0.05, 0.05]],
    )
    def test_local_energy_is_minus_half_z_squared(self, Z, r_vec):
        grad, lap = _hydrogenic_derivatives(Z, r_vec)
        R = jnp.zeros(3)
        charges = jnp.array([float(Z)])
        e = M.molecular_local_energy(grad, lap, R, jnp.asarray(r_vec), charges, n_e=1)
        assert abs(float(e) - (-0.5 * Z * Z)) < 1e-9

    def test_hamiltonian_dataclass_matches_free_function(self):
        Z, r_vec = 2.0, [0.5, -0.2, 0.7]
        grad, lap = _hydrogenic_derivatives(Z, r_vec)
        R = jnp.zeros(3)
        ham = M.MolecularHamiltonian(charges=jnp.array([Z]), n_e=1)
        e_cls = ham.local_energy(grad, lap, R, jnp.asarray(r_vec))
        e_fn = M.molecular_local_energy(
            grad, lap, R, jnp.asarray(r_vec), jnp.array([Z]), n_e=1
        )
        assert float(e_cls) == float(e_fn)

    def test_local_energy_is_position_independent(self):
        """The eigenvalue is constant across the domain (variance = 0)."""
        Z = 1.0
        energies = []
        for r_vec in [[0.2, 0, 0], [0, 0.9, 0], [0.4, -0.4, 0.4], [1.7, 0, 0]]:
            grad, lap = _hydrogenic_derivatives(Z, r_vec)
            energies.append(
                float(
                    M.molecular_local_energy(
                        grad, lap, jnp.zeros(3), jnp.asarray(r_vec),
                        jnp.array([Z]), n_e=1,
                    )
                )
            )
        assert np.std(energies) < 1e-9


class TestHarmonicOscillator:
    """Isotropic 3D SHO ground state ``log|psi| = -w r^2/2`` -> ``E = 3w/2``."""

    @pytest.mark.parametrize("w", [0.5, 1.0, 2.3])
    @pytest.mark.parametrize("r_vec", [[0.3, 0.0, 0.0], [0.5, -0.2, 0.7]])
    def test_ground_state_energy(self, w, r_vec):
        r_vec = jnp.asarray(r_vec)
        grad = -w * r_vec
        lap = -3.0 * w
        potential = 0.5 * w * w * jnp.sum(r_vec**2)
        e = M.local_energy(grad, lap, potential)
        assert abs(float(e) - 1.5 * w) < 1e-9


class TestCoulombPotential:
    """The Coulomb twin must reproduce ``omnibias.jax`` bit-for-bit."""

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_matches_reference_h2_like(self, seed):
        rng = np.random.default_rng(seed)
        R = jnp.asarray(rng.normal(size=6))  # 2 nuclei
        r = jnp.asarray(rng.normal(size=9))  # 3 electrons
        charges = jnp.array([1.0, 8.0])
        a = M.coulomb_potential(R, r, charges, n_e=3)
        b = ref_coulomb(R, r, charges, n_e=3)
        assert abs(float(a) - float(b)) < 1e-12

    def test_single_electron_single_atom_has_no_ee_nn(self):
        R = jnp.zeros(3)
        r = jnp.array([0.0, 0.0, 0.5])
        v = M.coulomb_potential(R, r, jnp.array([1.0]), n_e=1)
        assert abs(float(v) - (-1.0 / 0.5)) < 1e-12

    def test_nuclear_repulsion_only(self):
        """Two protons 2 bohr apart, one electron far away contributes e-n."""
        R = jnp.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        r = jnp.array([1.0, 1000.0, 0.0])  # electron very far
        v = M.coulomb_potential(R, r, jnp.array([1.0, 1.0]), n_e=1)
        # n-n term is 1/2 exactly; e-n terms are negligible (~1e-3).
        assert abs(float(v) - 0.5) < 1e-2


class TestClosedFormKineticVsAutograd:
    """Closed-form jet Laplacian of an MLP ``log|psi|`` == autograd."""

    @pytest.mark.parametrize("seed", [0, 3, 11])
    def test_grad_and_laplacian_parity(self, seed):
        rng = np.random.default_rng(seed)
        D, hidden = 6, 5
        W1 = jnp.asarray(rng.normal(size=(hidden, D)))
        b1 = jnp.asarray(rng.normal(size=hidden))
        W2 = jnp.asarray(rng.normal(size=(1, hidden)))
        b2 = jnp.asarray(rng.normal(size=1))
        layers = [(W1, b1, "tanh"), (W2, b2, None)]

        def log_psi(x):
            return (W2 @ jnp.tanh(W1 @ x + b1) + b2)[0]

        x0 = jnp.asarray(rng.normal(size=D))
        grad, lap = M.log_psi_derivatives(x0, layers, order=2)

        g_ref = jax.grad(log_psi)(x0)
        lap_ref = jnp.trace(jax.hessian(log_psi)(x0))
        assert np.allclose(np.asarray(grad), np.asarray(g_ref), atol=1e-10)
        assert abs(float(lap) - float(lap_ref)) < 1e-10

    def test_kinetic_energy_from_jet_matches_autograd(self):
        rng = np.random.default_rng(2026)
        D, hidden = 3, 6
        W1 = jnp.asarray(rng.normal(size=(hidden, D)))
        b1 = jnp.asarray(rng.normal(size=hidden))
        W2 = jnp.asarray(rng.normal(size=(1, hidden)))
        b2 = jnp.asarray(rng.normal(size=1))
        layers = [(W1, b1, "tanh"), (W2, b2, None)]

        def log_psi(x):
            return (W2 @ jnp.tanh(W1 @ x + b1) + b2)[0]

        x0 = jnp.asarray(rng.normal(size=D))
        grad, lap = M.log_psi_derivatives(x0, layers, order=2)
        t_closed = M.local_kinetic_energy(grad, lap)

        g_ref = jax.grad(log_psi)(x0)
        lap_ref = jnp.trace(jax.hessian(log_psi)(x0))
        t_ref = -0.5 * (lap_ref + jnp.sum(g_ref * g_ref))
        assert abs(float(t_closed) - float(t_ref)) < 1e-10

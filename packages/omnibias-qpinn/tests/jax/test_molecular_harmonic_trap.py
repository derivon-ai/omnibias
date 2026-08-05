# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Harmonic-trap oracle for the jet-Laplacian local energy (jax backend).

The isotropic quantum harmonic oscillator is the analytic oracle shared with
the torch Galerkin cross-check (see ``tests/torch/test_molecular_harmonic_trap``).
For a ground-state log-amplitude ``log|psi| = -w r^2 / 2`` in ``D`` dimensions
the closed-form drift kinetic energy plus the trap potential collapses to the
exact eigenvalue ``E_L = D w / 2`` at every point:

.. math::

    T_L = -\tfrac12(\nabla^2 \log|\psi| + \lVert\nabla\log|\psi|\rVert^2)
        = -\tfrac12(-wD + w^2 r^2), \qquad V = \tfrac12 w^2 r^2,
    \qquad E_L = T_L + V = \tfrac{D w}{2}.

This module also carries the closed-form-scope enforcement test for the jax
molecular surface (mirroring the torch enforcement).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.qpinn.jax import molecular as M


class TestMolecularHarmonicOracle:
    """``local_energy`` on the isotropic SHO ground state -> ``D w / 2``."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("w", [0.5, 1.0, 2.3])
    def test_ground_state_local_energy(self, dim: int, w: float) -> None:
        rng = np.random.default_rng(dim * 100 + int(w * 10))
        for _ in range(4):
            r_vec = jnp.asarray(rng.normal(size=dim))
            grad = -w * r_vec
            lap = jnp.asarray(-w * dim)
            potential = 0.5 * w * w * jnp.sum(r_vec**2)
            e = M.local_energy(grad, lap, potential)
            assert abs(float(e) - 0.5 * dim * w) < 1e-10

    def test_local_energy_is_position_independent_1d(self) -> None:
        """The eigenvalue is constant (zero variance) across the trap."""
        energies = []
        for x in (-1.7, -0.3, 0.0, 0.4, 2.1):
            r_vec = jnp.asarray([x])
            e = M.local_energy(-1.0 * r_vec, jnp.asarray(-1.0), 0.5 * r_vec[0] ** 2)
            energies.append(float(e))
        assert np.std(energies) < 1e-12
        assert abs(float(np.mean(energies)) - 0.5) < 1e-12


class TestQuantumChemistryHonesty:
    """Enforcement: only the closed-form molecular slice is exported (jax twin)."""

    def test_no_iterative_or_stochastic_surface_exported(self) -> None:
        exported = set(M.__all__)
        forbidden = {
            "vmc_sample",
            "vmc_energy",
            "metropolis_step",
            "scf",
            "hartree_fock",
            "roothaan",
            "ci_solve",
            "fci",
            "coupled_cluster",
            "ccsd",
            "eri",
            "electron_repulsion_integrals",
            "gaussian_basis_integrals",
        }
        assert exported.isdisjoint(forbidden)

    def test_module_docstring_records_out_of_scope(self) -> None:
        doc = (M.__doc__ or "").lower()
        assert "closed-form" in doc
        assert "vmc" in doc
        assert "scf" in doc
        assert "not" in doc

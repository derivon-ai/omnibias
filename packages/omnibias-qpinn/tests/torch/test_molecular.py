# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Molecular electronic-structure local energy (torch backend).

Bit-identical twin of :mod:`tests.jax.test_molecular`; same three oracles
(hydrogenic ``-Z^2/2``, harmonic ``3w/2``, closed-form-vs-autograd kinetic).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

torch.set_default_dtype(torch.float64)

from omnibias.qpinn.torch import molecular as M


def _hydrogenic_derivatives(Z: float, r_vec):
    r_vec = torch.as_tensor(r_vec, dtype=torch.float64)
    r = torch.linalg.norm(r_vec)
    grad = -Z * r_vec / r
    lap = -2.0 * Z / r
    return grad, lap


class TestHydrogenicLocalEnergy:
    @pytest.mark.parametrize("Z", [1.0, 2.0, 3.0, 8.0])
    @pytest.mark.parametrize(
        "r_vec",
        [[0.3, 0.0, 0.0], [0.5, -0.2, 0.7], [1.3, 0.4, -0.9], [0.05, 0.05, 0.05]],
    )
    def test_local_energy_is_minus_half_z_squared(self, Z, r_vec):
        grad, lap = _hydrogenic_derivatives(Z, r_vec)
        R = torch.zeros(3)
        charges = torch.tensor([float(Z)])
        e = M.molecular_local_energy(
            grad, lap, R, torch.as_tensor(r_vec, dtype=torch.float64), charges, n_e=1
        )
        assert abs(float(e) - (-0.5 * Z * Z)) < 1e-9

    def test_hamiltonian_dataclass_matches_free_function(self):
        Z, r_vec = 2.0, [0.5, -0.2, 0.7]
        grad, lap = _hydrogenic_derivatives(Z, r_vec)
        R = torch.zeros(3)
        rt = torch.as_tensor(r_vec, dtype=torch.float64)
        ham = M.MolecularHamiltonian(charges=torch.tensor([Z]), n_e=1)
        e_cls = ham.local_energy(grad, lap, R, rt)
        e_fn = M.molecular_local_energy(grad, lap, R, rt, torch.tensor([Z]), n_e=1)
        assert float(e_cls) == float(e_fn)

    def test_local_energy_is_position_independent(self):
        Z = 1.0
        energies = []
        for r_vec in [[0.2, 0, 0], [0, 0.9, 0], [0.4, -0.4, 0.4], [1.7, 0, 0]]:
            grad, lap = _hydrogenic_derivatives(Z, r_vec)
            energies.append(
                float(
                    M.molecular_local_energy(
                        grad, lap, torch.zeros(3),
                        torch.as_tensor(r_vec, dtype=torch.float64),
                        torch.tensor([Z]), n_e=1,
                    )
                )
            )
        assert np.std(energies) < 1e-9


class TestHarmonicOscillator:
    @pytest.mark.parametrize("w", [0.5, 1.0, 2.3])
    @pytest.mark.parametrize("r_vec", [[0.3, 0.0, 0.0], [0.5, -0.2, 0.7]])
    def test_ground_state_energy(self, w, r_vec):
        r_vec = torch.as_tensor(r_vec, dtype=torch.float64)
        grad = -w * r_vec
        lap = torch.tensor(-3.0 * w)
        potential = 0.5 * w * w * torch.sum(r_vec**2)
        e = M.local_energy(grad, lap, potential)
        assert abs(float(e) - 1.5 * w) < 1e-9


class TestCoulombPotential:
    def test_single_electron_single_atom_has_no_ee_nn(self):
        R = torch.zeros(3)
        r = torch.tensor([0.0, 0.0, 0.5])
        v = M.coulomb_potential(R, r, torch.tensor([1.0]), n_e=1)
        assert abs(float(v) - (-1.0 / 0.5)) < 1e-12

    def test_electron_electron_repulsion(self):
        """Two electrons 1 bohr apart, single unit nucleus at the midpoint."""
        R = torch.tensor([0.5, 0.0, 0.0])
        r = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        v = M.coulomb_potential(R, r, torch.tensor([1.0]), n_e=2)
        # e-e = 1/1 = 1; e-n = -(1/0.5 + 1/0.5) = -4; total = -3.
        assert abs(float(v) - (-3.0)) < 1e-12


class TestClosedFormKineticVsAutograd:
    @pytest.mark.parametrize("seed", [0, 3, 11])
    def test_grad_and_laplacian_parity(self, seed):
        rng = np.random.default_rng(seed)
        D, hidden = 6, 5
        W1 = torch.as_tensor(rng.normal(size=(hidden, D)))
        b1 = torch.as_tensor(rng.normal(size=hidden))
        W2 = torch.as_tensor(rng.normal(size=(1, hidden)))
        b2 = torch.as_tensor(rng.normal(size=1))
        layers = [(W1, b1, "tanh"), (W2, b2, None)]

        def log_psi(x):
            return (W2 @ torch.tanh(W1 @ x + b1) + b2)[0]

        x0 = torch.as_tensor(rng.normal(size=D))
        grad, lap = M.log_psi_derivatives(x0, layers, order=2)

        g_ref = torch.autograd.functional.jacobian(log_psi, x0)
        lap_ref = torch.trace(torch.autograd.functional.hessian(log_psi, x0))
        assert torch.allclose(grad, g_ref, atol=1e-10)
        assert abs(float(lap) - float(lap_ref)) < 1e-10

    def test_kinetic_energy_from_jet_matches_autograd(self):
        rng = np.random.default_rng(2026)
        D, hidden = 3, 6
        W1 = torch.as_tensor(rng.normal(size=(hidden, D)))
        b1 = torch.as_tensor(rng.normal(size=hidden))
        W2 = torch.as_tensor(rng.normal(size=(1, hidden)))
        b2 = torch.as_tensor(rng.normal(size=1))
        layers = [(W1, b1, "tanh"), (W2, b2, None)]

        def log_psi(x):
            return (W2 @ torch.tanh(W1 @ x + b1) + b2)[0]

        x0 = torch.as_tensor(rng.normal(size=D))
        grad, lap = M.log_psi_derivatives(x0, layers, order=2)
        t_closed = M.local_kinetic_energy(grad, lap)

        g_ref = torch.autograd.functional.jacobian(log_psi, x0)
        lap_ref = torch.trace(torch.autograd.functional.hessian(log_psi, x0))
        t_ref = -0.5 * (lap_ref + torch.sum(g_ref * g_ref))
        assert abs(float(t_closed) - float(t_ref)) < 1e-10

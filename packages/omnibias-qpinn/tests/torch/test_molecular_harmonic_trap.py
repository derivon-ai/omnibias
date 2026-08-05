# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Harmonic-trap cross-checks tying the two closed-form Laplacian paths (torch).

omnibias exposes the bound-state kinetic operator through *two independent*
closed-form code paths:

1. the field-substrate ``state.ops.laplacian`` that the direct Galerkin
   eigensolver (:mod:`omnibias.qpinn.torch.eigensolvers.galerkin`) assembles
   into the reduced Hamiltonian, and
2. the multivariate jet Laplacian of ``log|psi|`` that the molecular local
   energy (:mod:`omnibias.qpinn.torch.molecular`) uses.

The 1-D quantum harmonic oscillator ``H = -1/2 d^2/dx^2 + 1/2 x^2`` is the
shared analytic oracle: its ground state is the *single Gaussian*
``phi_0(x) = exp(-x^2/2)`` with ``E_0 = 1/2``. Because the ``gaussian``
activation is exactly ``exp(-z^2/2)``, a one-channel ``OneLayerVectorField``
with unit weights **is** ``phi_0`` -- so a ``K = 1`` Galerkin solve returns the
eigenvalue with no training and no fitting error, exercising the closed-form
Laplacian inside the eigensolver against the exact spectrum.

Three oracle classes:

* :class:`TestGalerkinHarmonicTrap` -- ``ops.laplacian`` path via Galerkin.
* :class:`TestMolecularHarmonicOracle` -- jet-Laplacian path via ``local_energy``.
* :class:`TestQuantumChemistryHonesty` -- enforcement that the iterative /
  stochastic quantum-chemistry surface (VMC / SCF / CI / CC / ERI) is *never*
  exported as closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

torch.set_default_dtype(torch.float64)

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn.torch import molecular as M
from omnibias.qpinn.torch.eigensolvers import galerkin_eigh, galerkin_matrices


def _sho_ground_state_field() -> OneLayerVectorField:
    """One-channel Gaussian field equal to ``phi_0(x) = exp(-x^2/2)``.

    ``gaussian(z) = exp(-z^2/2)``; with ``W = 1``, ``beta = 0``, ``c = 1``,
    ``b = 0`` the field value is exactly the SHO ground state. No training.
    """
    coord = CoordinateSpec(axes=("x",))
    spec = ComponentSpec(("phi_0",), groups={"phi": ("phi_0",)})
    field = OneLayerVectorField(
        coordinate_spec=coord, components=spec, hidden=1, base="gaussian"
    )
    with torch.no_grad():
        field.W.weight.copy_(torch.tensor([[1.0]]))
        field.W.bias.zero_()
        field.c.weight.copy_(torch.tensor([[1.0]]))
        field.c.bias.zero_()
    return field


def _sho_potential(coords: torch.Tensor) -> torch.Tensor:
    """``V(x) = 1/2 x^2`` (unit-frequency harmonic trap)."""
    return 0.5 * coords[..., 0] ** 2


def _trapezoid_grid(a: float, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Composite-trapezoid nodes/weights on ``[-a, a]`` -> ``(coords, weights)``."""
    x = torch.linspace(-a, a, n, dtype=torch.float64)
    dx = float(x[1] - x[0])
    w = torch.full((n,), dx, dtype=torch.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    return x.unsqueeze(-1), w


class TestGalerkinHarmonicTrap:
    """The ``ops.laplacian`` path recovers the exact SHO spectrum."""

    def test_field_value_is_the_gaussian_ground_state(self) -> None:
        field = _sho_ground_state_field()
        xs = torch.linspace(-3.0, 3.0, 13, dtype=torch.float64).unsqueeze(-1)
        state = field(xs)
        value = state.ops.value(state, "phi_0")
        assert torch.allclose(value, torch.exp(-0.5 * xs[:, 0] ** 2), atol=1e-12)

    def test_k1_ground_state_energy_is_exactly_half(self) -> None:
        """``K = 1`` Rayleigh quotient == 0.5 to quadrature precision.

        The single basis function is the exact eigenstate, so the reduced
        Hamiltonian is ``H = 0.5 * S`` and ``E_0 = H/S = 0.5`` regardless of
        the (sufficiently fine) grid -- a bit-stable closed-form-Laplacian
        oracle inside the eigensolver.
        """
        field = _sho_ground_state_field()
        coords, weights = _trapezoid_grid(8.0, 801)
        S, H = galerkin_matrices(
            field=field,
            quadrature_coords=coords,
            quadrature_weights=weights,
            basis_names=("phi_0",),
            potential_fn=_sho_potential,
            kinetic_prefactor=0.5,
        )
        rayleigh = float((H[0, 0] / S[0, 0]).detach())
        assert abs(rayleigh - 0.5) < 1e-10

        result = galerkin_eigh(
            field=field,
            quadrature_coords=coords,
            quadrature_weights=weights,
            basis_names=("phi_0",),
            potential_fn=_sho_potential,
            kinetic_prefactor=0.5,
            n_states=1,
        )
        assert abs(float(result.eigenvalues[0]) - 0.5) < 1e-10
        assert np.isfinite(result.cond_S)

    def test_lowest_eigenvalue_respects_variational_bound(self) -> None:
        """Any Gaussian basis gives a lowest eigenvalue ``>= E_0 = 0.5``.

        Two *off-centre* Gaussians (centres ``+-1``) do not contain ``phi_0``
        exactly, so the Rayleigh-Ritz eigenvalue is a genuine upper bound on
        the true ground-state energy: it must sit at or above ``0.5``. This
        certifies the closed-form kinetic term is not silently under-counted.
        """
        coord = CoordinateSpec(axes=("x",))
        spec = ComponentSpec(
            ("g_minus", "g_plus"),
            groups={"phi": ("g_minus", "g_plus")},
        )
        field = OneLayerVectorField(
            coordinate_spec=coord, components=spec, hidden=2, base="gaussian"
        )
        with torch.no_grad():
            # Two unit-width Gaussians centred at x = -1 and x = +1:
            # gaussian(x - c) = exp(-(x-c)^2/2), so W = 1, beta = -c.
            field.W.weight.copy_(torch.tensor([[1.0], [1.0]]))
            field.W.bias.copy_(torch.tensor([1.0, -1.0]))  # beta = -c
            field.c.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            field.c.bias.zero_()
        coords, weights = _trapezoid_grid(9.0, 1201)
        result = galerkin_eigh(
            field=field,
            quadrature_coords=coords,
            quadrature_weights=weights,
            basis_names=("g_minus", "g_plus"),
            potential_fn=_sho_potential,
            kinetic_prefactor=0.5,
            n_states=2,
        )
        # Variational: lowest eigenvalue is an upper bound on 0.5.
        assert float(result.eigenvalues[0]) >= 0.5 - 1e-9
        # And a reasonable basis stays close to the exact value.
        assert float(result.eigenvalues[0]) < 0.75


class TestMolecularHarmonicOracle:
    r"""The jet-Laplacian ``local_energy`` path on the isotropic SHO.

    For ``log|psi| = -w r^2 / 2`` in ``D`` dimensions the closed-form drift
    kinetic energy plus the trap potential ``V = w^2 r^2 / 2`` collapses to the
    exact ground-state eigenvalue ``E_L = D w / 2`` at *every* point.
    """

    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("w", [0.5, 1.0, 2.3])
    def test_ground_state_local_energy(self, dim: int, w: float) -> None:
        rng = np.random.default_rng(dim * 100 + int(w * 10))
        for _ in range(4):
            r_vec = torch.as_tensor(rng.normal(size=dim), dtype=torch.float64)
            grad = -w * r_vec  # grad (-w r^2/2) = -w r
            lap = torch.tensor(-w * dim)  # nabla^2 (-w r^2/2) = -w D
            potential = 0.5 * w * w * torch.sum(r_vec**2)
            e = M.local_energy(grad, lap, potential)
            assert abs(float(e) - 0.5 * dim * w) < 1e-10

    def test_matches_galerkin_ground_state_1d(self) -> None:
        """1-D unit trap: jet-path ``E_L`` == Galerkin ``E_0`` == 0.5, everywhere."""
        for x in (-1.7, -0.3, 0.0, 0.4, 2.1):
            r_vec = torch.tensor([x])
            grad = -1.0 * r_vec
            lap = torch.tensor(-1.0)
            potential = 0.5 * r_vec[0] ** 2
            e = M.local_energy(grad, lap, potential)
            assert abs(float(e) - 0.5) < 1e-12


class TestQuantumChemistryHonesty:
    """Enforcement: only the closed-form molecular slice is exported.

    VMC Monte-Carlo sampling, SCF / Hartree-Fock / CI / coupled-cluster
    self-consistency, and Gaussian-basis electron-repulsion integrals are
    iterative or stochastic numerics (see ``docs/scope-and-guarantees.md`` s6).
    These tests fail if such a surface is silently added to the closed-form
    molecular module or if the honesty note is deleted from its docstring.
    """

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
        # Names the iterative/stochastic pieces that are *not* claimed.
        assert "vmc" in doc
        assert "scf" in doc
        assert "not" in doc

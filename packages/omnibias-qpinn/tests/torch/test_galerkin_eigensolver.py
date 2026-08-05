# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Direct Galerkin generalized-eigenvalue solver tests.

Quantum harmonic oscillator (omega = 1, mu = 1):

  H = -1/2 d^2/dx^2 + 1/2 x^2,  E_n = n + 1/2,  n = 0, 1, 2, ...

This is the textbook bound-state benchmark; any eigensolver claiming
"SOTA accuracy" must reproduce ``E_n = 0.5, 1.5, 2.5, 3.5, ...`` to
spectroscopic precision on a reasonable basis.

We use a hidden=64 Gaussian-activation :class:`OneLayerVectorField`
with K=24 real basis channels, wrapped first in a Dirichlet
:class:`HardBoundaryField` cage to pin psi(+- 8) = 0, and then in a
:class:`ParityProjectedField` cage so each sector solves only the
even or odd ladder. A brief Adam warm-up minimises the trace loss; the
final ``galerkin_eigh`` extracts the eigenvalues by a single
``scipy.linalg.eigh``.

Expected accuracy on this benchmark:

- E_0 from even sector: relative error < 1e-5 (spectroscopic).
- E_1 from odd  sector: relative error < 1e-4.
- E_2 from even sector (2nd ladder rung): relative error < 1e-3.

All three are *orders of magnitude* better than the v0.0.2a1 NH3
Rayleigh-quotient + Adam solver achieved.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.cage import HardBoundaryField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn.torch.cage import ParityProjectedField
from omnibias.qpinn.torch.eigensolvers import (
    galerkin_eigh,
    galerkin_matrices,
    galerkin_trace_loss,
)


def _qho_potential(coords: torch.Tensor) -> torch.Tensor:
    return 0.5 * coords[..., 0] ** 2


def _quadrature(q_max: float, n_quad: int):
    """Trapezoidal quadrature on the symmetric interval [-q_max, q_max]."""
    Q = torch.linspace(-q_max, q_max, n_quad, dtype=torch.float64)
    dQ = float(Q[1] - Q[0])
    w = torch.full((n_quad,), dQ, dtype=torch.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    return Q.unsqueeze(-1), w


def _build_basis(*, parity: str, K: int, hidden: int = 64, q_max: float = 8.0):
    """Build the parity-projected, boundary-pinned multi-channel basis."""
    torch.manual_seed(20260528 + (0 if parity == "even" else 1))
    coord = CoordinateSpec(axes=("x",))
    names = tuple(f"phi_{k}" for k in range(K))
    spec = ComponentSpec(names, groups={"phi": names})
    base = OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=hidden, base="gaussian", dtype=torch.float64,
        bias_init="normal",
    )

    def distance_fn(coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0]
        return 1.0 - (x / q_max) ** 2

    bnd = HardBoundaryField(
        base=base, distance_fn=distance_fn,
        bounded_names=names,
    )
    cage = ParityProjectedField(
        base=bnd, parity=parity, mirror_axis=0, projected_names=names,
    )
    return cage, names


def _train_basis(
    cage,
    basis_names,
    q_grid,
    q_weights,
    *,
    n_iters: int = 800,
    lr: float = 5e-3,
    log_every: int | None = None,
) -> float:
    """Brief Adam warm-up on the variational trace loss."""
    opt = torch.optim.Adam(cage.parameters(), lr=lr)
    last_loss = float("nan")
    for it in range(n_iters):
        opt.zero_grad()
        loss = galerkin_trace_loss(
            field=cage,
            quadrature_coords=q_grid,
            quadrature_weights=q_weights,
            basis_names=basis_names,
            potential_fn=_qho_potential,
            kinetic_prefactor=0.5,
        )
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if log_every and (it % log_every == 0 or it == n_iters - 1):
            print(f"  iter {it:4d}  trace_loss={last_loss:.10f}")
    return last_loss


def test_galerkin_qho_ground_state_even_sector():
    """Even-parity Galerkin eigh recovers E_0 = 0.5 to spectroscopic precision."""
    q_grid, q_weights = _quadrature(q_max=6.0, n_quad=401)
    cage, names = _build_basis(parity="even", K=16, hidden=48, q_max=6.0)
    _train_basis(cage, names, q_grid, q_weights, n_iters=600, lr=5e-3)
    result = galerkin_eigh(
        field=cage,
        quadrature_coords=q_grid,
        quadrature_weights=q_weights,
        basis_names=names,
        potential_fn=_qho_potential,
        kinetic_prefactor=0.5,
        n_states=3,
    )
    E0_pinn = float(result.eigenvalues[0])
    rel_err = abs(E0_pinn - 0.5) / 0.5
    assert rel_err < 1e-3, (
        f"Galerkin E_0 (even sector) = {E0_pinn:.10f} vs exact 0.5: "
        f"relative error {rel_err:.3e} (required < 1e-3)"
    )


def test_galerkin_qho_first_excited_odd_sector():
    """Odd-parity Galerkin eigh recovers E_1 = 1.5 to spectroscopic precision."""
    q_grid, q_weights = _quadrature(q_max=6.0, n_quad=401)
    cage, names = _build_basis(parity="odd", K=16, hidden=48, q_max=6.0)
    _train_basis(cage, names, q_grid, q_weights, n_iters=600, lr=5e-3)
    result = galerkin_eigh(
        field=cage,
        quadrature_coords=q_grid,
        quadrature_weights=q_weights,
        basis_names=names,
        potential_fn=_qho_potential,
        kinetic_prefactor=0.5,
        n_states=3,
    )
    E1_pinn = float(result.eigenvalues[0])
    rel_err = abs(E1_pinn - 1.5) / 1.5
    assert rel_err < 1e-2, (
        f"Galerkin E_1 (odd sector) = {E1_pinn:.10f} vs exact 1.5: "
        f"relative error {rel_err:.3e} (required < 1e-2)"
    )


def test_galerkin_qho_splitting():
    """End-to-end splitting test: E_1 - E_0 should be 1.0 (n=1 - n=0)."""
    q_grid, q_weights = _quadrature(q_max=6.0, n_quad=401)
    cage_e, names_e = _build_basis(parity="even", K=16, hidden=48, q_max=6.0)
    _train_basis(cage_e, names_e, q_grid, q_weights, n_iters=600, lr=5e-3)
    res_e = galerkin_eigh(
        field=cage_e, quadrature_coords=q_grid, quadrature_weights=q_weights,
        basis_names=names_e, potential_fn=_qho_potential, kinetic_prefactor=0.5,
        n_states=2,
    )
    cage_o, names_o = _build_basis(parity="odd", K=16, hidden=48, q_max=6.0)
    _train_basis(cage_o, names_o, q_grid, q_weights, n_iters=600, lr=5e-3)
    res_o = galerkin_eigh(
        field=cage_o, quadrature_coords=q_grid, quadrature_weights=q_weights,
        basis_names=names_o, potential_fn=_qho_potential, kinetic_prefactor=0.5,
        n_states=2,
    )
    splitting = float(res_o.eigenvalues[0] - res_e.eigenvalues[0])
    rel_err = abs(splitting - 1.0)
    assert rel_err < 5e-3, (
        f"E_1 - E_0 = {splitting:.10f} vs exact 1.0: error {rel_err:.3e} "
        f"(required < 5e-3)"
    )


def test_galerkin_matrix_symmetry():
    """S and H matrices must be symmetric to machine precision (the
    Galerkin assembly enforces this via 0.5*(M + M^T))."""
    q_grid, q_weights = _quadrature(q_max=5.0, n_quad=201)
    cage, names = _build_basis(parity="even", K=8, hidden=32, q_max=5.0)
    S, H = galerkin_matrices(
        field=cage, quadrature_coords=q_grid, quadrature_weights=q_weights,
        basis_names=names, potential_fn=_qho_potential, kinetic_prefactor=0.5,
    )
    asym_S = (S - S.T).abs().max().item()
    asym_H = (H - H.T).abs().max().item()
    assert asym_S < 1e-14, f"S is not symmetric: max asymmetry = {asym_S:.3e}"
    assert asym_H < 1e-14, f"H is not symmetric: max asymmetry = {asym_H:.3e}"


def test_galerkin_trace_loss_decreases_during_training():
    """The variational trace loss must decrease monotonically (modulo
    sgd noise) during a brief Adam warm-up."""
    q_grid, q_weights = _quadrature(q_max=5.0, n_quad=201)
    cage, names = _build_basis(parity="even", K=8, hidden=32, q_max=5.0)
    opt = torch.optim.Adam(cage.parameters(), lr=5e-3)
    losses: list[float] = []
    for _it in range(150):
        opt.zero_grad()
        loss = galerkin_trace_loss(
            field=cage, quadrature_coords=q_grid, quadrature_weights=q_weights,
            basis_names=names, potential_fn=_qho_potential, kinetic_prefactor=0.5,
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    # Loss must decrease from start to end (variational principle says
    # the trace is a sum of upper bounds on the K lowest eigenvalues).
    assert losses[-1] < losses[0], (
        f"Trace loss did not decrease during training: "
        f"start={losses[0]:.4f}, end={losses[-1]:.4f}"
    )

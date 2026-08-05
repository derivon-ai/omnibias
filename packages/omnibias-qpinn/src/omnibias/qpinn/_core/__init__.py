# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic helpers for quantum-physics PINN residuals.

The submodules here carry pure-Python, dtype-agnostic helpers that the
torch and jax backends consume in the same way:

- :mod:`complex` -- split-real encoding of a complex wavefunction
  ``psi = psi_re + i * psi_im`` via two real :class:`ComponentSpec`
  components grouped under a single name. The encoding is the canonical
  v0.0.1 choice; native complex-valued ``ComponentSpec`` is a future
  v0.0.2 enhancement.
- :mod:`units` -- atomic-units convenience constants kept in lockstep
  with :mod:`omnibias.jax.bo_derivatives`. Re-defined here (not
  re-exported) so the torch-only install never imports jax.
"""

from __future__ import annotations

from omnibias.qpinn._core.complex import (
    apply_hamiltonian,
    apply_kinetic,
    apply_potential,
    is_psi_group,
    make_psi_components,
    psi_density,
    psi_phase,
    psi_value,
)
from omnibias.qpinn._core.parity import (
    project_parity_even_derivative,
    project_parity_odd_derivative,
    project_parity_value,
)
from omnibias.qpinn._core.spinor import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    apply_gamma_matrix,
    gamma5,
    gamma_matrices,
    gamma_partial_psi,
    make_spinor_components,
    pauli_dot,
    pauli_matrices,
    spinor_value,
)
from omnibias.qpinn._core.units import (
    AMU_TO_ME,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_CM,
    HARTREE_TO_EV,
)
from omnibias.qpinn._core.vortex import (
    VortexDetection,
    detect_vortices,
    detect_vortices_full,
    feynman_vortex_count,
    thomas_fermi_density_2d,
    thomas_fermi_mu_2d,
    thomas_fermi_radius_2d,
)

__all__ = [
    "AMU_TO_ME",
    "BOHR_TO_ANGSTROM",
    "HARTREE_TO_CM",
    "HARTREE_TO_EV",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "VortexDetection",
    "apply_gamma_matrix",
    "apply_hamiltonian",
    "apply_kinetic",
    "apply_potential",
    "detect_vortices",
    "detect_vortices_full",
    "feynman_vortex_count",
    "gamma5",
    "gamma_matrices",
    "gamma_partial_psi",
    "is_psi_group",
    "make_psi_components",
    "make_spinor_components",
    "pauli_dot",
    "pauli_matrices",
    "project_parity_even_derivative",
    "project_parity_odd_derivative",
    "project_parity_value",
    "psi_density",
    "psi_phase",
    "psi_value",
    "spinor_value",
    "thomas_fermi_density_2d",
    "thomas_fermi_mu_2d",
    "thomas_fermi_radius_2d",
]

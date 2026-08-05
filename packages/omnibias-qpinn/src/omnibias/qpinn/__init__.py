# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Quantum-physics-informed neural networks built on omnibias-pinn.

``omnibias-qpinn`` ships closed-form differential-operator residuals,
hard-conservation cages, and diagnostics for the partial differential
equations of quantum physics (Schrodinger / Gross-Pitaevskii / Helmholtz
/ Klein-Gordon / Dirac).

Public API split:

- :mod:`omnibias.qpinn` (this module) -- backend-agnostic helpers
  (complex encoding, spinor DSL, atomic-units convenience constants).
- :mod:`omnibias.qpinn.torch` and :mod:`omnibias.qpinn.jax` -- the two
  backends, imported on demand. Each carries ``equations``, ``cage``,
  and ``diagnostics`` submodules with bit-identical numerics. The direct
  Galerkin ``eigensolvers`` submodule is currently **torch only** (alpha; a
  JAX twin is on the roadmap).

The package rides on top of :mod:`omnibias.pinn`. We deliberately do
**not** ship our own ``fields`` / ``ops`` modules: every public residual
consumes a ``FieldState`` produced by an
``omnibias.pinn.{torch,jax}.fields.*`` field.

Backend selection
-----------------

Importing ``omnibias.qpinn.torch`` requires ``omnibias-qpinn[torch]``
(pulls in ``omnibias-torch`` + ``torch>=2.0``). Importing
``omnibias.qpinn.jax`` requires ``omnibias-qpinn[jax]``. The backend
subpackages are *not* imported eagerly here, so installing only one
extra is sufficient.

Example
-------

.. code-block:: python

    from omnibias.qpinn import make_psi_components
    components = make_psi_components(name="psi")

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-qpinn")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

from omnibias.qpinn._core import (
    AMU_TO_ME,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_CM,
    HARTREE_TO_EV,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    apply_gamma_matrix,
    apply_hamiltonian,
    apply_kinetic,
    apply_potential,
    detect_vortices,
    detect_vortices_full,
    feynman_vortex_count,
    gamma5,
    gamma_matrices,
    gamma_partial_psi,
    is_psi_group,
    make_psi_components,
    make_spinor_components,
    pauli_dot,
    pauli_matrices,
    project_parity_even_derivative,
    project_parity_odd_derivative,
    project_parity_value,
    psi_density,
    psi_phase,
    psi_value,
    spinor_value,
    thomas_fermi_density_2d,
    thomas_fermi_mu_2d,
    thomas_fermi_radius_2d,
)

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "AMU_TO_ME",
    "BOHR_TO_ANGSTROM",
    "HARTREE_TO_CM",
    "HARTREE_TO_EV",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "__lineage__",
    "__version__",
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

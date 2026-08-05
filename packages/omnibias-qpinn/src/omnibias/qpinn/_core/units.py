# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Atomic-units convenience constants.

These are intentionally a verbatim copy of the constants in
:mod:`omnibias.jax.bo_derivatives` so that quantum-PINN code can
inter-operate with Born-Oppenheimer / vibrational-frequency calculations
without a hard import of the jax backend. A cross-package test in
``tests/_core/test_complex_encoding.py`` asserts the values match to
machine precision when the jax backend is importable.

Values come from CODATA 2018:

- ``HARTREE_TO_CM``  -- 1 Hartree (E_h) in inverse centimetres,
  ``E_h / (h c) * 1e-2``.
- ``HARTREE_TO_EV``  -- 1 Hartree in electron-volts.
- ``BOHR_TO_ANGSTROM`` -- 1 Bohr radius (a_0) in angstrom.
- ``AMU_TO_ME``      -- 1 atomic mass unit in electron masses.
"""

from __future__ import annotations

#: Hartree -> wavenumber (cm^-1). Matches :mod:`omnibias.jax.bo_derivatives`.
HARTREE_TO_CM: float = 219474.6313632

#: Hartree -> electron-volt. CODATA 2018.
HARTREE_TO_EV: float = 27.211386245988

#: Bohr radius -> angstrom. CODATA 2018.
BOHR_TO_ANGSTROM: float = 0.52917721067

#: Atomic mass unit -> electron-mass units. Matches
#: :mod:`omnibias.jax.bo_derivatives`.
AMU_TO_ME: float = 1822.888486209


__all__ = [
    "AMU_TO_ME",
    "BOHR_TO_ANGSTROM",
    "HARTREE_TO_CM",
    "HARTREE_TO_EV",
]

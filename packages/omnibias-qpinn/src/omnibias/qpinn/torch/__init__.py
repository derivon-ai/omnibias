# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch backend for omnibias-qpinn.

Public submodules:

- :mod:`equations` -- prebuilt PDE residuals for the quantum-physics
  PDE family (``TISE``, ``TDSE``, ``NLS``, ``Helmholtz``, ``KleinGordon``,
  ``Dirac``).
- :mod:`cage` -- strict-conservation layer wrappers
  (``NormConservationField``, ``BlochPeriodicField``,
  ``HermitianOperatorField``).
- :mod:`diagnostics` -- expectation values, probability currents, norm
  drift, energy variance.
- :mod:`molecular` -- Born-Oppenheimer electronic-structure local energy
  (closed-form kinetic + Coulomb potential).

Fields and the operator surface live in :mod:`omnibias.pinn.torch` --
``omnibias.qpinn`` deliberately does not re-export them.
"""

from __future__ import annotations

from omnibias.qpinn.torch import (
    cage,
    diagnostics,
    eigensolvers,
    equations,
    molecular,
)

__all__ = ["cage", "diagnostics", "eigensolvers", "equations", "molecular"]

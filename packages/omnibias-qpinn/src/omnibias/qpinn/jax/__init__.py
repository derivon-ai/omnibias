# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX backend for omnibias-qpinn.

Public submodules:

- :mod:`equations` -- prebuilt PDE residuals (``TISE``, ``TDSE``,
  ``NLS``, ...).
- :mod:`cage` -- strict-conservation cage layers
  (``NormConservationField``, ...).
- :mod:`diagnostics` -- expectation values, probability currents,
  norm drift.
- :mod:`molecular` -- Born-Oppenheimer electronic-structure local energy
  (closed-form kinetic + Coulomb potential).

The submodules are bit-for-bit twins of the torch backend; cross-backend
parity is enforced in :mod:`tests.cross_backend`.
"""

from __future__ import annotations

from omnibias.qpinn.jax import (
    cage,
    diagnostics,
    equations,
    molecular,
)

__all__ = ["cage", "diagnostics", "equations", "molecular"]

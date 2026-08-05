# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic schemas for omnibias-pinn.

This module is *pure Python*: no torch, no jax, no numpy. The torch and
jax backends consume the schemas defined here so that cross-backend
parity is structural by construction.

Public surface:

- :class:`CoordinateSpec` -- the input-axis spec.
- :class:`ComponentSpec` -- the output-channel spec (with named groups).
- :class:`FieldState` -- the value object produced by ``field(coords)``.
- :class:`ComponentView`, :class:`VectorView` -- the Option 1 attribute
  DSL views that delegate into the backend ops.
- :class:`SigmaCache` -- the lazy ``sigma^(n)(z)`` cache.
- :class:`EquationSpec`, :class:`ResidualPolicy`,
  :class:`IncompressibilityPolicy` -- PDE configuration types.
- :class:`FieldBase` -- the structural protocol every typed field obeys.
- :mod:`ops_registry` -- third-party op extension point.
- :mod:`registry` -- equation registry (specs + per-backend factories).
"""

from __future__ import annotations

from omnibias.pinn._core import ops_registry, registry
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.field_base import FieldBase
from omnibias.pinn._core.pde import (
    EquationSpec,
    IncompressibilityPolicy,
    ResidualPolicy,
)
from omnibias.pinn._core.sigma_cache import SigmaCache
from omnibias.pinn._core.state import FieldState
from omnibias.pinn._core.view import ComponentView, VectorView, did_you_mean

__all__ = [
    "ComponentSpec",
    "ComponentView",
    "CoordinateSpec",
    "EquationSpec",
    "FieldBase",
    "FieldState",
    "IncompressibilityPolicy",
    "ResidualPolicy",
    "SigmaCache",
    "VectorView",
    "did_you_mean",
    "ops_registry",
    "registry",
]

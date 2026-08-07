# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic field substrate (pure Python: no torch / jax / numpy).

This is the foundational schema layer shared by every omnibias field
package (``omnibias-pinn``, ``omnibias-geometry``, ``omnibias-score``, ...).
The torch and jax backends consume the schemas defined here so that
cross-backend parity is structural by construction.

Public surface:

- :class:`CoordinateSpec` -- the input-axis spec.
- :class:`ComponentSpec` -- the output-channel spec (with named groups).
- :class:`FieldState` -- the value object produced by ``field(coords)``.
- :class:`ComponentView`, :class:`VectorView` -- the attribute-DSL views
  that delegate into the backend ops.
- :class:`SigmaCache` -- the lazy ``sigma^(n)(z)`` cache.
- :class:`FieldBase` -- the structural protocol every typed field obeys.
- :mod:`ops_registry` -- third-party op extension point.
- :data:`DISPATCH_ATTR` -- the field marker attribute name used by the
  backend ops to pick the closed-form reduction path.
- :data:`READOUT_INDEPENDENT_ATTR` -- the field marker declaring that
  per-state caches are independent of the readout parameters.
"""

from __future__ import annotations

from omnibias.fields._core import ops_registry
from omnibias.fields._core.catalog import (
    DOMAINS,
    OperatorInfo,
    get_operator,
    list_operators,
    operator_names,
)
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.field_base import (
    DISPATCH_ATTR,
    READOUT_INDEPENDENT_ATTR,
    FieldBase,
)
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState
from omnibias.fields._core.view import ComponentView, VectorView, did_you_mean

__all__ = [
    "ComponentSpec",
    "ComponentView",
    "CoordinateSpec",
    "DISPATCH_ATTR",
    "DOMAINS",
    "FieldBase",
    "FieldState",
    "OperatorInfo",
    "READOUT_INDEPENDENT_ATTR",
    "SigmaCache",
    "VectorView",
    "did_you_mean",
    "get_operator",
    "list_operators",
    "operator_names",
    "ops_registry",
]

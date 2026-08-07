# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-fields: the backend-agnostic field substrate.

``omnibias-fields`` hosts the :class:`FieldState` value object, the
attribute-DSL views, the lazy ``sigma^(n)(z)`` cache, the op-extension
registry, and the cross-backend (torch + jax) closed-form differential
operator surface (gradient, divergence, curl, laplacian, hessian,
jacobian, ...), plus the quadrature / inner-product / norm / tensor-divergence
and Wirtinger ops.

The pure-Python schemas live in :mod:`omnibias.fields._core`; the backend
kernels live in ``omnibias.fields.torch`` and ``omnibias.fields.jax`` and are
imported lazily (importing :mod:`omnibias.fields` does not pull in torch or
jax).

Downstream packages (``omnibias-pinn``, ``omnibias-geometry``,
``omnibias-score``) build on this substrate; ``omnibias-pinn`` additionally
re-exports the moved symbols through back-compat shims at
``omnibias.pinn._core`` and ``omnibias.pinn.<backend>.ops``.

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

from omnibias.fields._core import (
    DISPATCH_ATTR,
    DOMAINS,
    READOUT_INDEPENDENT_ATTR,
    ComponentSpec,
    ComponentView,
    CoordinateSpec,
    FieldBase,
    FieldState,
    OperatorInfo,
    SigmaCache,
    VectorView,
    did_you_mean,
    get_operator,
    list_operators,
    operator_names,
    ops_registry,
)

try:
    __version__ = _pkg_version("omnibias-fields")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

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
    "__lineage__",
    "__version__",
    "did_you_mean",
    "get_operator",
    "list_operators",
    "operator_names",
    "ops_registry",
]

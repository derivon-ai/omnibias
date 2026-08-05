# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX backend for omnibias-fractional.

``ops`` (grid / spectral + closed-form analytic), ``order``, and ``layers``
(functional / pytree fractional layers) need only ``omnibias-jax``. The ``field``
submodule (closed-form field fractional partial + fractional-diffusion residual)
additionally needs ``omnibias-fields`` and is imported lazily so the rest of the
backend stays importable without it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from omnibias.fractional.jax import layers, ops, order

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fractional.jax import field

__all__ = ["field", "layers", "ops", "order"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "field":
        return importlib.import_module("omnibias.fractional.jax.field")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

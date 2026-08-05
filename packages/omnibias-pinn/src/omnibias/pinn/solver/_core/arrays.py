# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tiny backend-neutral array-namespace resolver.

Canonical source / initial-condition callables need transcendental functions
(``sin``, ``exp``, ...) that differ between torch and jax. Rather than baking a
backend into a :class:`~omnibias.pinn.solver._core.system.System`, a callable receives
a runtime tensor and asks :func:`array_namespace` for the matching module.

The imports are performed *lazily inside the function* so that
``import omnibias.pinn.solver`` never pulls in torch or jax at module load -- the
``_core`` package stays backend-free by construction.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def array_namespace(x: Any) -> ModuleType:
    """Return ``torch`` or ``jax.numpy`` matching the type of ``x``.

    Falls back to :mod:`numpy` for plain arrays / scalars.
    """
    module = type(x).__module__
    if module.startswith("torch"):
        import torch

        return torch
    if module.startswith("jax") or module.startswith("jaxlib"):
        import jax.numpy as jnp

        return jnp
    import numpy as np

    return np


__all__ = ["array_namespace"]

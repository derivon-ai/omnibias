# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Opt-in extensions for the omnibias-pinn field-ops surface.

Backend-neutral helpers that wire model-level operators into the
:mod:`omnibias.fields.ops_registry` extension point. Importing this package does
*not* register anything; call the explicit ``register_*`` helpers to opt in.

Currently provides :func:`register_lim_along`, which exposes the closed-form jet
``lim`` operator as ``state.<component>.lim_along`` (see
:mod:`omnibias.pinn.extensions.lim_along`).
"""

from __future__ import annotations

from omnibias.pinn.extensions.lim_along import (
    LIM_ALONG_KEY,
    register_lim_along,
    unregister_lim_along,
)

__all__ = [
    "LIM_ALONG_KEY",
    "register_lim_along",
    "unregister_lim_along",
]

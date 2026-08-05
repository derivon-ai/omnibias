# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""JAX frontend for omnibias-verify: ingest ``(W, b)`` parameter stacks into a :class:`~omnibias.verify.Network`, plus closed-form-jet warm-start seeds for the certified minimiser."""

from __future__ import annotations

from omnibias.verify.jax.ingest import network_from_params
from omnibias.verify.jax.warm_start import descent_seeds, warm_started_network_minimize

__all__ = [
    "descent_seeds",
    "network_from_params",
    "warm_started_network_minimize",
]

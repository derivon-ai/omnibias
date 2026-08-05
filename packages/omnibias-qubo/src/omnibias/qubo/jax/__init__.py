# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""JAX backend for omnibias-qubo."""

from __future__ import annotations

from omnibias.qubo.jax.relaxation import qubo_relaxation

__all__ = ["qubo_relaxation"]

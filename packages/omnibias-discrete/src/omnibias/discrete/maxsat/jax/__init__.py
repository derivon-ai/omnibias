# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""JAX backend for the MaxSAT front-end: the differentiable relaxation."""

from __future__ import annotations

from omnibias.discrete.maxsat.jax.relaxation import maxsat_relaxation

__all__ = ["maxsat_relaxation"]

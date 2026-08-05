# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""JAX backend for omnibias-discrete: the annealed relaxation core."""

from __future__ import annotations

from omnibias.discrete.jax.relaxation import anneal_descent

__all__ = ["anneal_descent"]

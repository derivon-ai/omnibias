# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""JAX backend for omnibias-logic: the differentiable relaxation twins.

``maxsat_relaxation`` is re-exported unchanged from ``omnibias.discrete.maxsat.jax``;
``sat_relaxation`` is the annealed #SAT model finder.
"""

from __future__ import annotations

from omnibias.discrete.maxsat.jax import maxsat_relaxation
from omnibias.logic.jax.relaxation import sat_relaxation

__all__ = ["maxsat_relaxation", "sat_relaxation"]

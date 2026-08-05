# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic schemas for omnibias-variational."""

from __future__ import annotations

from omnibias.variational._core.constraint import Constraint, ConstraintFn
from omnibias.variational._core.hamiltonian import Hamiltonian, HamiltonianFn
from omnibias.variational._core.lagrangian import (
    Lagrangian,
    LagrangianDensity,
    LagrangianDensityFn,
    LagrangianFn,
)

__all__ = [
    "Constraint",
    "ConstraintFn",
    "Hamiltonian",
    "HamiltonianFn",
    "Lagrangian",
    "LagrangianDensity",
    "LagrangianDensityFn",
    "LagrangianFn",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.pinn.solver: mesh-free solver for coupled systems of PDEs.

The value proposition is the omnibias closed-form derivative tower: a single
forward pass yields *every* mixed partial up to a chosen order exactly, so PDE
operators -- including high-order ones (biharmonic, 4th-order, mixed) -- are
cheap and exact at arbitrary order, mesh-free, with no nested autodiff.

This module re-exports the backend-free schemas (``Domain``, ``System``,
``BoundaryCondition``, ``InitialCondition``, the taxonomy, the sampling
descriptors, the honesty labels, and the canonical problem builders). The
numeric drivers live under ``omnibias.pinn.solver.torch`` and ``omnibias.pinn.solver.jax`` and
are imported lazily so ``import omnibias.pinn.solver`` never pulls in torch or jax. The
optional certified mode lives in ``omnibias.pinn.solver.verify``.

Maturity: this is an **alpha** submodule (folded in from the former standalone
``omnibias-pde`` package) shipped inside the Beta ``omnibias-pinn`` distribution.
The mesh-free coupled-PDE solver API may still change; the rest of
``omnibias-pinn`` is Beta.
"""

from __future__ import annotations

from omnibias.pinn.solver._core import (
    AUTODIFF,
    CLOSED_FORM,
    HIGH_ORDER,
    NUMERICAL,
    SPECTRAL,
    Arity,
    BoundaryCondition,
    Classification,
    Coefficient,
    CollocationSpec,
    DeclinedCondition,
    Domain,
    Field,
    HardConditionPlan,
    InitialCondition,
    Linearity,
    Observations,
    PDEType,
    ProblemKind,
    RefinementSpec,
    Residual,
    System,
    Unknown,
    advection_diffusion,
    array_namespace,
    assert_no_unproven_claim,
    bind_unknowns,
    burgers,
    heat,
    honesty_labels,
    make_system,
    plan_hard_conditions,
    poisson,
    reaction_diffusion,
    sample_observations,
    wave,
)

__all__ = [
    "AUTODIFF",
    "Arity",
    "BoundaryCondition",
    "CLOSED_FORM",
    "Classification",
    "Coefficient",
    "CollocationSpec",
    "DeclinedCondition",
    "Domain",
    "Field",
    "HIGH_ORDER",
    "HardConditionPlan",
    "InitialCondition",
    "Linearity",
    "NUMERICAL",
    "Observations",
    "PDEType",
    "ProblemKind",
    "RefinementSpec",
    "Residual",
    "SPECTRAL",
    "System",
    "Unknown",
    "advection_diffusion",
    "array_namespace",
    "assert_no_unproven_claim",
    "bind_unknowns",
    "burgers",
    "heat",
    "honesty_labels",
    "make_system",
    "plan_hard_conditions",
    "poisson",
    "reaction_diffusion",
    "sample_observations",
    "wave",
]

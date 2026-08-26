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
    PERIODIC_ORDERS,
    READOUT_INDEPENDENT_ATTR,
    SPECTRAL,
    Arity,
    BoundaryCondition,
    Classification,
    Coefficient,
    CollocationSpec,
    DeclinedCondition,
    Domain,
    Field,
    GuidedHuntReport,
    GuidedStep,
    HardConditionPlan,
    InitialCondition,
    Linearity,
    Observations,
    PDEType,
    PicardReport,
    ProblemKind,
    ReadoutDependentError,
    RefinementSpec,
    Residual,
    System,
    Unknown,
    advection_diffusion,
    array_namespace,
    assert_no_unproven_claim,
    bind_unknowns,
    burgers,
    burgers_residual_periodic,
    cinf_fourier_basis,
    cole_hopf_exact_burgers_demo,
    guided_cinf_burgers_hunt,
    guided_cinf_smoke,
    heat,
    honesty_labels,
    honesty_payload,
    make_system,
    picard_frozen_advection,
    plan_hard_conditions,
    poisson,
    reaction_diffusion,
    recursive_toolkit_smoke,
    requires_readout_independent,
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
    "GuidedHuntReport",
    "GuidedStep",
    "HIGH_ORDER",
    "HardConditionPlan",
    "InitialCondition",
    "Linearity",
    "NUMERICAL",
    "Observations",
    "PDEType",
    "PERIODIC_ORDERS",
    "PicardReport",
    "ProblemKind",
    "READOUT_INDEPENDENT_ATTR",
    "ReadoutDependentError",
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
    "burgers_residual_periodic",
    "cinf_fourier_basis",
    "cole_hopf_exact_burgers_demo",
    "guided_cinf_burgers_hunt",
    "guided_cinf_smoke",
    "heat",
    "honesty_labels",
    "honesty_payload",
    "make_system",
    "picard_frozen_advection",
    "plan_hard_conditions",
    "poisson",
    "reaction_diffusion",
    "recursive_toolkit_smoke",
    "requires_readout_independent",
    "sample_observations",
    "wave",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free schemas for omnibias.pinn.solver (no torch / jax imports).

Domain / System / conditions / taxonomy / sampling / honesty and the canonical
problem builders. The torch and jax numeric drivers live in
``omnibias.pinn.solver.torch`` and ``omnibias.pinn.solver.jax`` and are imported lazily.
"""

from __future__ import annotations

from omnibias.pinn.solver._core.arrays import array_namespace
from omnibias.pinn.solver._core.conditions import (
    BoundaryCondition,
    InitialCondition,
    ValueLike,
)
from omnibias.pinn.solver._core.domain import Domain
from omnibias.pinn.solver._core.honesty import (
    AUTODIFF,
    CLOSED_FORM,
    HIGH_ORDER,
    NUMERICAL,
    SPECTRAL,
    assert_no_unproven_claim,
    honesty_labels,
)
from omnibias.pinn.solver._core.observations import (
    Observations,
    check_observations,
    sample_observations,
)
from omnibias.pinn.solver._core.problems import (
    advection_diffusion,
    burgers,
    heat,
    poisson,
    reaction_diffusion,
    wave,
)
from omnibias.pinn.solver._core.sampling import (
    CollocationSpec,
    RefinementSpec,
    boundary_points,
    candidate_points,
    initial_slice_points,
    interior_points,
    select_refinement_points,
    spatial_boundary_points,
)
from omnibias.pinn.solver._core.system import Field, Residual, System, make_system
from omnibias.pinn.solver._core.taxonomy import (
    Arity,
    Classification,
    Linearity,
    PDEType,
    ProblemKind,
)
from omnibias.pinn.solver._core.unknowns import (
    TRANSFORMS,
    Coefficient,
    Unknown,
    bind_unknowns,
    bound_names,
    coefficient,
    collect_unknowns,
)

__all__ = [
    "AUTODIFF",
    "Arity",
    "BoundaryCondition",
    "CLOSED_FORM",
    "Classification",
    "Coefficient",
    "CollocationSpec",
    "Domain",
    "Field",
    "HIGH_ORDER",
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
    "TRANSFORMS",
    "Unknown",
    "ValueLike",
    "advection_diffusion",
    "array_namespace",
    "assert_no_unproven_claim",
    "bind_unknowns",
    "bound_names",
    "boundary_points",
    "burgers",
    "candidate_points",
    "check_observations",
    "coefficient",
    "collect_unknowns",
    "heat",
    "honesty_labels",
    "initial_slice_points",
    "interior_points",
    "make_system",
    "poisson",
    "reaction_diffusion",
    "sample_observations",
    "select_refinement_points",
    "spatial_boundary_points",
    "wave",
]

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Soft-population evolution on the discrete substrate (theory 03-01).

Selection ``w = softmax(-beta E)`` is **temperature collapse**
(``beta -> inf``, feasibility). Pack genes mutate the geometry of the
founding bias collapse (``delta -> 0`` to ``sigma^(K-1)``). Do not conflate
the two. Evolutionary algorithms are not new; the deliverable
is the closed-form selection gap, geometry-aware mutation, exact-curvature
polish with full evaluation accounting, and a certify-loop stop.
Nothing here is a P = NP claim.

Bit-identical torch / jax twins implement :func:`soft_weights` only.
Evolve loops use ``numpy.random.Generator(seed)`` (documented seed policy).
"""

from __future__ import annotations

from omnibias.discrete.evolution._core import (
    CertifiedEvolveResult,
    EvolveResult,
    GenerationRecord,
    GeometryMutation,
    PackGenes,
    PopulationConfig,
    certified_discrete_evolve,
    crossover_packs,
    gd_polish,
    geometry_evolve,
    honesty_payload,
    memetic_evolve,
    mutate_geometry,
    mutate_isotropic,
    named_g2_objective,
    named_pack_target,
    named_quadratic,
    newton_polish,
    pack_fitness,
    selection_gap_bound,
    selection_stats,
    soft_population_evolve,
    soft_weights,
)

__all__ = [
    "CertifiedEvolveResult",
    "EvolveResult",
    "GenerationRecord",
    "GeometryMutation",
    "PackGenes",
    "PopulationConfig",
    "certified_discrete_evolve",
    "crossover_packs",
    "gd_polish",
    "geometry_evolve",
    "honesty_payload",
    "memetic_evolve",
    "mutate_geometry",
    "mutate_isotropic",
    "named_g2_objective",
    "named_pack_target",
    "named_quadratic",
    "newton_polish",
    "pack_fitness",
    "selection_gap_bound",
    "selection_stats",
    "soft_population_evolve",
    "soft_weights",
]

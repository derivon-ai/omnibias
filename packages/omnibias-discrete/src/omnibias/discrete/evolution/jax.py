# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""JAX twin of soft-population selection (theory 03-01).

``soft_weights`` is bit-identical to the numpy / torch twins at float64
(``jax_enable_x64``). Evolve loops stay on ``numpy.random.Generator(seed)``
so a seed is a backend-neutral trajectory. Selection ``beta -> inf`` is
temperature collapse (feasibility), distinct from the founding bias collapse
(``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.discrete.evolution._core import (
    CertifiedEvolveResult,
    EvolveResult,
    GeometryMutation,
    PackGenes,
    PopulationConfig,
    certified_discrete_evolve,
    geometry_evolve,
    memetic_evolve,
    named_g2_objective,
    named_pack_target,
    named_quadratic,
    selection_gap_bound,
    soft_population_evolve,
)


def soft_weights(energies: Array, *, beta: float) -> Array:
    """``w_i = softmax(-beta E_i)``. ``beta == 0`` is uniform."""
    e = jnp.asarray(energies, dtype=jnp.float64).reshape(-1)
    if e.size < 1:
        raise ValueError("energies must be non-empty")
    b = float(beta)
    if b < 0.0:
        raise ValueError(f"beta must be >= 0, got {b}")
    if b == 0.0:
        return jnp.full(e.shape, 1.0 / float(e.size), dtype=jnp.float64)
    z = -b * e
    z = z - jnp.max(z)
    w = jnp.exp(z)
    return w / jnp.sum(w)


__all__ = [
    "CertifiedEvolveResult",
    "EvolveResult",
    "GeometryMutation",
    "PackGenes",
    "PopulationConfig",
    "certified_discrete_evolve",
    "geometry_evolve",
    "memetic_evolve",
    "named_g2_objective",
    "named_pack_target",
    "named_quadratic",
    "selection_gap_bound",
    "soft_population_evolve",
    "soft_weights",
]

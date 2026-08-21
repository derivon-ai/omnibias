# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""PyTorch twin of soft-population selection (theory 03-01).

``soft_weights`` is bit-identical to the numpy / jax twins at float64.
Evolve loops stay on ``numpy.random.Generator(seed)`` so a seed is a
backend-neutral trajectory. Selection ``beta -> inf`` is temperature
collapse (feasibility), distinct from the founding bias collapse
(``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

import torch
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
from torch import Tensor


def soft_weights(energies: Tensor, *, beta: float) -> Tensor:
    """``w_i = softmax(-beta E_i)``. ``beta == 0`` is uniform."""
    e = torch.as_tensor(energies, dtype=torch.float64).reshape(-1)
    if e.numel() < 1:
        raise ValueError("energies must be non-empty")
    b = float(beta)
    if b < 0.0:
        raise ValueError(f"beta must be >= 0, got {b}")
    if b == 0.0:
        return torch.full_like(e, 1.0 / float(e.numel()))
    z = -b * e
    z = z - torch.max(z)
    w = torch.exp(z)
    return w / torch.sum(w)


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

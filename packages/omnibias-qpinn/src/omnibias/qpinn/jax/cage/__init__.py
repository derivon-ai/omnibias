# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard-conservation cage layers for the JAX backend (qpinn)."""

from __future__ import annotations

from omnibias.qpinn.jax.cage.bloch import (
    BlochPeriodicField,
    make_bloch_periodic_field,
)
from omnibias.qpinn.jax.cage.cusp import (
    NuclearCuspField,
    make_nuclear_cusp_field,
    nuclear_cusp_slope,
)
from omnibias.qpinn.jax.cage.hermitian import (
    hermitian_projection,
    hermiticity_loss,
)
from omnibias.qpinn.jax.cage.norm import (
    NormConservationField,
    make_norm_conservation_field,
    norm_loss,
)
from omnibias.qpinn.jax.cage.parity import (
    ParityProjectedField,
    make_parity_projected_field,
)

__all__ = [
    "BlochPeriodicField",
    "NormConservationField",
    "NuclearCuspField",
    "ParityProjectedField",
    "hermitian_projection",
    "hermiticity_loss",
    "make_bloch_periodic_field",
    "make_norm_conservation_field",
    "make_nuclear_cusp_field",
    "make_parity_projected_field",
    "norm_loss",
    "nuclear_cusp_slope",
]

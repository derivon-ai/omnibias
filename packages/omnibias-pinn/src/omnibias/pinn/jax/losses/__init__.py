# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equation-agnostic loss helpers for the jax backend.

Bit-parity twin of :mod:`omnibias.pinn.torch.losses`.
"""

from __future__ import annotations

from omnibias.pinn.jax.losses.asymptotic import (
    asymptotic_bc_loss,
    asymptotic_ratio,
    far_field_decay_loss,
    network_ray_jet,
)
from omnibias.pinn.jax.losses.causal import (
    CausalConfig,
    causal_residual_loss,
    causal_weights_from_per_bin,
)
from omnibias.pinn.jax.losses.entropy import entropy_consistent_residual
from omnibias.pinn.jax.losses.ntk import (
    estimate_ntk_trace,
    ntk_balanced_loss,
)
from omnibias.pinn.jax.losses.sobolev import (
    mse_residual_loss,
    sobolev_residual_loss,
    sobolev_weight,
)

__all__ = [
    "CausalConfig",
    "asymptotic_bc_loss",
    "asymptotic_ratio",
    "causal_residual_loss",
    "causal_weights_from_per_bin",
    "entropy_consistent_residual",
    "estimate_ntk_trace",
    "far_field_decay_loss",
    "mse_residual_loss",
    "network_ray_jet",
    "ntk_balanced_loss",
    "sobolev_residual_loss",
    "sobolev_weight",
]

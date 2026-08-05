# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equation-agnostic loss helpers for the torch backend.

Four families:

1. **Sobolev preconditioning** (:func:`sobolev_residual_loss`,
   :func:`sobolev_weight`) -- downweight high-spatial-frequency
   residual modes so the optimiser sees a better-conditioned problem.
2. **Wang-Perdikaris causal weighting** (:func:`causal_residual_loss`,
   :func:`causal_weights_from_per_bin`) -- respect causality during
   training.
3. **Entropy-consistent residual**
   (:func:`entropy_consistent_residual`) -- a convex-function transform
   applied before MSE, used for hyperbolic conservation laws.
4. **NTK rebalance** (:func:`ntk_balanced_loss`,
   :func:`estimate_ntk_trace`) -- balance multiple loss terms by
   their NTK traces.

These helpers consume *raw tensors* (residual cubes, per-term losses,
etc) so they are equation-agnostic and equally usable with any field
type or PDE.

A fifth family -- **asymptotic / removable boundary conditions**
(:func:`asymptotic_bc_loss`, :func:`asymptotic_ratio`,
:func:`far_field_decay_loss`) -- consumes a *layer stack* rather than a
residual cube: it builds the exact directional Taylor jet of the network
(``mlp_jet``) and imposes a differentiable limit (``lhopital_ratio``) as a
trainable loss (removable regularity at a singular point, far-field decay).
"""

from __future__ import annotations

from omnibias.pinn.torch.losses.asymptotic import (
    asymptotic_bc_loss,
    asymptotic_ratio,
    far_field_decay_loss,
    network_ray_jet,
)
from omnibias.pinn.torch.losses.causal import (
    CausalConfig,
    causal_residual_loss,
    causal_weights_from_per_bin,
)
from omnibias.pinn.torch.losses.entropy import entropy_consistent_residual
from omnibias.pinn.torch.losses.ntk import (
    estimate_ntk_trace,
    ntk_balanced_loss,
)
from omnibias.pinn.torch.losses.sobolev import (
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

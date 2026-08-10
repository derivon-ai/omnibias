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

A sixth -- **adaptive weighting** -- is stateful, which the five above are not.
:class:`LossWeighter` and its subclasses (:class:`GradNormWeighter`,
:class:`NTKWeighter`, :class:`ConstantWeighter`) hold an EMA of the per-term
weights ``lambda_k`` and refresh it on a cadence; they are shared pure Python,
so the torch and jax weights agree by construction, and only the measurement
(:func:`grad_stats`, :func:`ntk_trace_stats`) is written per backend.
:func:`self_adaptive_loss` and :class:`SelfAdaptiveWeights` are the pointwise
counterpart: one *trained* weight per collocation point, ascended rather than
estimated.

Marching those weights across a long time horizon is
:class:`~omnibias.pinn._core.marching.TimeWindowSchedule` and
:class:`~omnibias.pinn._core.marching.TimeMarcher`, re-exported here beside the
causal loss they drive.

A seventh -- **interface residuals** for domain decomposition
(:func:`interface_residual`, :func:`normal_derivative`) -- consumes *two*
:class:`~omnibias.pinn._core.state.FieldState`\\ s, one per subdomain, and
imposes value and normal-flux continuity on the seam between them. Its geometry
(:class:`~omnibias.pinn._core.interface.Interface`,
:func:`~omnibias.pinn._core.interface.interface_points`) is shared pure numpy
and re-exported here too.
"""

from __future__ import annotations

from omnibias.pinn._core.interface import (
    Interface,
    InterfaceSpec,
    interface_points,
    split_by_interface,
)
from omnibias.pinn._core.marching import (
    TimeMarcher,
    TimeWindowSchedule,
    slice_points,
    window_points,
)
from omnibias.pinn._core.weighting import (
    ConstantWeighter,
    GradNormWeighter,
    GradStats,
    LossWeighter,
    NTKWeighter,
)
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
from omnibias.pinn.torch.losses.interface import (
    InterfaceOutput,
    flux_jump,
    interface_loss,
    interface_residual,
    normal_derivative,
    normal_flux,
    value_jump,
)
from omnibias.pinn.torch.losses.ntk import (
    empirical_jacobian,
    estimate_ntk_trace,
    fourier_mode_learning_rates,
    kernel_task_alignment,
    ntk_balanced_loss,
    ntk_eigenspectrum,
    ntk_tail_head_index,
    spectral_bias_index,
)
from omnibias.pinn.torch.losses.sobolev import (
    mse_residual_loss,
    sobolev_residual_loss,
    sobolev_weight,
)
from omnibias.pinn.torch.losses.weighting import (
    SelfAdaptiveWeights,
    grad_stats,
    ntk_trace_stats,
    reverse_gradient,
    self_adaptive_loss,
)

__all__ = [
    "CausalConfig",
    "ConstantWeighter",
    "GradNormWeighter",
    "GradStats",
    "Interface",
    "InterfaceOutput",
    "InterfaceSpec",
    "LossWeighter",
    "NTKWeighter",
    "SelfAdaptiveWeights",
    "TimeMarcher",
    "TimeWindowSchedule",
    "asymptotic_bc_loss",
    "asymptotic_ratio",
    "causal_residual_loss",
    "causal_weights_from_per_bin",
    "empirical_jacobian",
    "entropy_consistent_residual",
    "estimate_ntk_trace",
    "far_field_decay_loss",
    "flux_jump",
    "fourier_mode_learning_rates",
    "grad_stats",
    "interface_loss",
    "interface_points",
    "interface_residual",
    "kernel_task_alignment",
    "mse_residual_loss",
    "network_ray_jet",
    "normal_derivative",
    "normal_flux",
    "ntk_balanced_loss",
    "ntk_eigenspectrum",
    "ntk_tail_head_index",
    "ntk_trace_stats",
    "reverse_gradient",
    "self_adaptive_loss",
    "slice_points",
    "sobolev_residual_loss",
    "sobolev_weight",
    "spectral_bias_index",
    "split_by_interface",
    "value_jump",
    "window_points",
]

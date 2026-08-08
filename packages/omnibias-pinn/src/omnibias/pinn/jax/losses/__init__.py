# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equation-agnostic loss helpers for the jax backend.

Bit-parity twin of :mod:`omnibias.pinn.torch.losses`.
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
from omnibias.pinn.jax.losses.interface import (
    InterfaceOutput,
    flux_jump,
    interface_loss,
    interface_residual,
    normal_derivative,
    normal_flux,
    value_jump,
)
from omnibias.pinn.jax.losses.ntk import (
    empirical_jacobian,
    estimate_ntk_trace,
    fourier_mode_learning_rates,
    kernel_task_alignment,
    ntk_balanced_loss,
    ntk_eigenspectrum,
    ntk_tail_head_index,
    spectral_bias_index,
)
from omnibias.pinn.jax.losses.sobolev import (
    mse_residual_loss,
    sobolev_residual_loss,
    sobolev_weight,
)
from omnibias.pinn.jax.losses.weighting import (
    SelfAdaptiveWeights,
    grad_stats,
    make_self_adaptive_weights,
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
    "entropy_consistent_residual",
    "empirical_jacobian",
    "estimate_ntk_trace",
    "fourier_mode_learning_rates",
    "kernel_task_alignment",
    "far_field_decay_loss",
    "flux_jump",
    "grad_stats",
    "interface_loss",
    "interface_points",
    "interface_residual",
    "make_self_adaptive_weights",
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

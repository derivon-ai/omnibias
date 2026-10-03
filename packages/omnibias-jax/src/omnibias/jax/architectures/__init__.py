# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reusable PINN architectures over the shared activation derivative tower.

Includes multilayer and Fourier-feature fields, multiscale fields, hard boundary
constraints, attention jets, integral kernels, and residual networks.
"""

from omnibias.jax.architectures.attention import (
    AttentionJetMLP,
    make_attention_jet_mlp,
)
from omnibias.jax.architectures.ftc_net import (
    DualFTCConfig,
    FTCNet,
    FTCNetConfig,
    dual_ftc_loss,
    ftc_block,
)
from omnibias.jax.architectures.hardbc import (
    AffineFactor,
    AffineLift,
    BoundaryMask,
    HardConstraintField,
    dirichlet_interval,
    homogeneous_box,
    initial_value,
)
from omnibias.jax.architectures.integral_kernel import (
    IntegralKernelConfig,
    integral_kernel_apply,
)
from omnibias.jax.architectures.jetkan import (
    JetKANConfig,
    JetKANParams,
    init_jet_kan,
    jet_kan_apply,
    jet_kan_from_torch_state,
    jet_kan_jet,
    jet_kan_jet_mv,
    jetkan_from_band_plan,
    refine_pack,
)
from omnibias.jax.architectures.multiscale import (
    AdaptiveActivation,
    AdaptiveJetMLP,
    MscaleMLP,
    make_adaptive_activation,
    make_adaptive_jet_mlp,
    make_mscale_mlp,
)
from omnibias.jax.architectures.pinn import (
    FourierFeatureMLP,
    JetMLP,
    make_fourier_feature_mlp,
    make_jet_mlp,
    make_siren,
)
from omnibias.jax.architectures.piratenet import (
    PirateNetConfig,
    init_pirate_params,
    pirate_apply,
    pirate_features,
)

__all__ = [
    "AdaptiveActivation",
    "AdaptiveJetMLP",
    "AffineFactor",
    "AffineLift",
    "AttentionJetMLP",
    "BoundaryMask",
    "DualFTCConfig",
    "FTCNet",
    "FTCNetConfig",
    "FourierFeatureMLP",
    "HardConstraintField",
    "IntegralKernelConfig",
    "JetKANConfig",
    "JetKANParams",
    "JetMLP",
    "MscaleMLP",
    "PirateNetConfig",
    "dirichlet_interval",
    "dual_ftc_loss",
    "ftc_block",
    "homogeneous_box",
    "init_jet_kan",
    "init_pirate_params",
    "initial_value",
    "integral_kernel_apply",
    "jet_kan_apply",
    "jet_kan_from_torch_state",
    "jet_kan_jet",
    "jet_kan_jet_mv",
    "jetkan_from_band_plan",
    "make_adaptive_activation",
    "make_adaptive_jet_mlp",
    "make_attention_jet_mlp",
    "make_fourier_feature_mlp",
    "make_jet_mlp",
    "make_mscale_mlp",
    "make_siren",
    "pirate_apply",
    "pirate_features",
    "refine_pack",
]

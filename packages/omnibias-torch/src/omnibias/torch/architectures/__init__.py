# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reusable PINN architectures over the shared activation derivative tower.

Includes multilayer and Fourier-feature fields, multiscale fields, hard boundary
constraints, attention jets, integral kernels, and residual networks.
"""

from omnibias.torch.architectures.attention import AttentionJetMLP
from omnibias.torch.architectures.ftc_net import (
    DualFTCConfig,
    FTCNet,
    FTCNetConfig,
    dual_ftc_loss,
    ftc_block,
)
from omnibias.torch.architectures.hardbc import (
    AffineFactor,
    AffineLift,
    BoundaryMask,
    HardConstraintField,
    dirichlet_interval,
    homogeneous_box,
    initial_value,
)
from omnibias.torch.architectures.integral_kernel import (
    IntegralKernelConfig,
    integral_kernel_apply,
)
from omnibias.torch.architectures.jetkan import (
    JetKAN,
    JetKANConfig,
    edge_functions,
    jetkan_from_band_plan,
)
from omnibias.torch.architectures.joint_operator import (
    FittedJointOperatorRegressor,
    JointOperatorRegressor,
    OperatorMetadata,
    fit_joint_operator_regressor,
)
from omnibias.torch.architectures.multiscale import (
    AdaptiveActivation,
    AdaptiveJetMLP,
    MscaleMLP,
)
from omnibias.torch.architectures.pinn import (
    DeepPINNHeat,
    FourierFeatureMLP,
    JetMLP,
    PINNHeat,
    make_siren,
)
from omnibias.torch.architectures.piratenet import (
    PirateNet,
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
    "DeepPINNHeat",
    "DualFTCConfig",
    "FTCNet",
    "FTCNetConfig",
    "FittedJointOperatorRegressor",
    "FourierFeatureMLP",
    "HardConstraintField",
    "IntegralKernelConfig",
    "JetKAN",
    "JetKANConfig",
    "JetMLP",
    "JointOperatorRegressor",
    "MscaleMLP",
    "OperatorMetadata",
    "PINNHeat",
    "PirateNet",
    "PirateNetConfig",
    "dirichlet_interval",
    "dual_ftc_loss",
    "edge_functions",
    "fit_joint_operator_regressor",
    "ftc_block",
    "homogeneous_box",
    "init_pirate_params",
    "initial_value",
    "integral_kernel_apply",
    "jetkan_from_band_plan",
    "make_siren",
    "pirate_apply",
    "pirate_features",
]

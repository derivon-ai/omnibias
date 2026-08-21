# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reference architectures built from OMBU + OperatorBlock primitives.

Three families:

- :mod:`pinn`: physics-informed networks where each spatial / temporal
  derivative is one OperatorBlock with the appropriate K (gradient = K=2,
  Laplacian = K=3, arbitrary nth derivative = K=n+1).
- :mod:`multiscale`: the frequency-aware PINN constructions -- a trainable
  activation slope ``sigma(n a z)`` (a real ``ActivationSpec`` from the
  ``tempered`` combinator, so the tower stays exact) and the MscaleDNN band
  mixture ``u(x) = sum_j f_j(alpha_j x)``.
- :mod:`attention`: the first *non-local* block on the substrate -- a softmax
  mixture over a trainable memory whose *coordinate* derivatives stay closed
  form through ``jet_attention``.
- :mod:`cmbnet`: operator-typed CNN where each convolution layer carries
  an explicit operator role (gradient / Laplacian / band / integral).
- :mod:`scannet`: gated grid-free stacked bias-scan banks (theory 02-01).
  Equivariance is per-layer, on-lattice, not the translation group of
  ``R^D``. Templates reuse the six ``OperatorBlock`` roles.
- :mod:`jetkan`: gated univariate multi-pack edges (theory 02-03). Exactness
  is of the model jet; the Kolmogorov-Arnold theorem does not justify the
  architecture.
- :mod:`piratenet`: jaxpi α-skip (``α=0`` is identity). Not ImageNet / ViT
  and not CCF stretch.
- :mod:`cvxlayer`: differentiable embedded convex solvers (LASSO, logistic)
  unrolled as depth-T multi-bias networks where each layer is one solver
  iteration realised by a K=2 collapse.
"""

from omnibias.torch.architectures.attention import AttentionJetMLP
from omnibias.torch.architectures.cmbnet import CmbNet
from omnibias.torch.architectures.cvxlayer import CvxLasso, CvxLogistic
from omnibias.torch.architectures.frame_unet import FrameUNetConfig, frame_unet_forward
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
from omnibias.torch.architectures.jet_token import (
    JetTokenConfig,
    jet_token_forward,
    worked_compose_jet,
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
from omnibias.torch.architectures.ladder import HermiteBasis, LadderNet
from omnibias.torch.architectures.multiscale import (
    AdaptiveActivation,
    AdaptiveJetMLP,
    MscaleMLP,
)
from omnibias.torch.architectures.pack_moe import PackMoEConfig, pack_moe_forward
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
from omnibias.torch.architectures.riccati_flow import RiccatiFlowConfig, riccati_flow
from omnibias.torch.architectures.scannet import ScanNet, ScanNetConfig, scannet_from_band_plan

__all__ = [
    "AdaptiveActivation",
    "AdaptiveJetMLP",
    "AffineFactor",
    "AffineLift",
    "AttentionJetMLP",
    "BoundaryMask",
    "CmbNet",
    "CvxLasso",
    "CvxLogistic",
    "DeepPINNHeat",
    "DualFTCConfig",
    "FTCNet",
    "FTCNetConfig",
    "FittedJointOperatorRegressor",
    "FourierFeatureMLP",
    "FrameUNetConfig",
    "HardConstraintField",
    "HermiteBasis",
    "JetKAN",
    "JetKANConfig",
    "JetMLP",
    "JetTokenConfig",
    "JointOperatorRegressor",
    "LadderNet",
    "MscaleMLP",
    "OperatorMetadata",
    "PINNHeat",
    "PackMoEConfig",
    "PirateNet",
    "PirateNetConfig",
    "RiccatiFlowConfig",
    "ScanNet",
    "ScanNetConfig",
    "dirichlet_interval",
    "dual_ftc_loss",
    "edge_functions",
    "fit_joint_operator_regressor",
    "frame_unet_forward",
    "ftc_block",
    "homogeneous_box",
    "init_pirate_params",
    "initial_value",
    "jet_token_forward",
    "jetkan_from_band_plan",
    "make_siren",
    "pack_moe_forward",
    "pirate_apply",
    "pirate_features",
    "riccati_flow",
    "scannet_from_band_plan",
    "worked_compose_jet",
]

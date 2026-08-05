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
- :mod:`cvxlayer`: differentiable embedded convex solvers (LASSO, logistic)
  unrolled as depth-T multi-bias networks where each layer is one solver
  iteration realised by a K=2 collapse.
"""

from omnibias.torch.architectures.attention import AttentionJetMLP
from omnibias.torch.architectures.cmbnet import CmbNet
from omnibias.torch.architectures.cvxlayer import CvxLasso, CvxLogistic
from omnibias.torch.architectures.hardbc import (
    AffineFactor,
    AffineLift,
    BoundaryMask,
    HardConstraintField,
    dirichlet_interval,
    homogeneous_box,
    initial_value,
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
    "FittedJointOperatorRegressor",
    "FourierFeatureMLP",
    "HardConstraintField",
    "JetMLP",
    "JointOperatorRegressor",
    "MscaleMLP",
    "OperatorMetadata",
    "PINNHeat",
    "dirichlet_interval",
    "fit_joint_operator_regressor",
    "homogeneous_box",
    "initial_value",
    "make_siren",
]

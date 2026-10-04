# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch activation derivatives and Taylor jets for physics-informed models.

Shared polynomial coefficients define the derivative tower. Operator blocks,
mixed jets, and PINN architectures reuse it without nested coordinate autodiff.
Architecture and optimization helpers are available in their own modules.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-torch")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

from omnibias.torch.activations import (
    ActivationSpec,
    get_activation,
    is_registered,
    list_activations,
    register_activation,
)
from omnibias.torch.blocks import (
    AnalyticGaussianConv1d,
    AnalyticGaussianConv2d,
    OperatorBlock,
    analytic_gaussian_taps,
    cmbConv1d,
    cmbConv2d,
    cmbLinear,
)
from omnibias.torch.growable import GrowableOperatorMultiBiasUnit, GrowStrategy
from omnibias.torch.implicit import (
    DEQConfig,
    DEQNotContractive,
    DEQResult,
    DEQSolverUnknown,
    deq_du_dW,
    deq_solve,
    deq_vjp,
)
from omnibias.torch.jet import (
    affine_jet,
    antiderivative_jet,
    compose_jet,
    compose_jet_riccati,
    derivative_jet,
    jet_to_tower,
    layer_jet,
    lhopital_ratio,
    limit_of_ratio,
    mlp_jet,
    removable_value,
    tower_to_jet,
)
from omnibias.torch.jet_mv import (
    affine_jet_mv,
    compose_jet_mv,
    identity_jet,
    jet_attention,
    jet_exp,
    jet_gradient,
    jet_hessian,
    jet_multiply,
    jet_partials,
    jet_reciprocal,
    jet_softmax,
    layer_jet_mv,
    mlp_jet_mv,
)
from omnibias.torch.line_search import (
    GradientSecant,
    JetLineSearchConfig,
    LineSearchResult,
    jet_line_search,
    jet_line_search_on_ray,
)
from omnibias.torch.multipack import BirkhoffOMBU, MultiPackUnit, multipack_response
from omnibias.torch.optim_block_search import (
    BlockSpec,
    arrangement_w_block,
    block_direction,
    block_exact_search,
    block_exact_sweep,
    last_linear_block,
    ombu_bias_block,
)
from omnibias.torch.refine import AdaptivePackBank, grow_ombu, refine
from omnibias.torch.scan import BankSpec, BiasScan, scan_response, soft_argmax_offset
from omnibias.torch.scan_equivariant import EquivariantScan, steerable_basis
from omnibias.torch.tempered_blocks import LearnablePReLU, TemperedActivation
from omnibias.torch.unit import OperatorMultiBiasUnit
from omnibias.torch.weight_loss_jet import (
    WeightLossJetSpec,
    one_layer_loss,
    one_layer_loss_grad,
    one_layer_loss_hessian,
    one_layer_loss_jet,
    one_layer_newton_direction,
)

OMBU = OperatorMultiBiasUnit  # short alias
GrowableOMBU = GrowableOperatorMultiBiasUnit  # short alias

# Limit family exposed as package metadata.
__lineage__ = "bias collapse"

__all__ = [
    "ActivationSpec",
    "AdaptivePackBank",
    "AnalyticGaussianConv1d",
    "AnalyticGaussianConv2d",
    "BankSpec",
    "BiasScan",
    "BirkhoffOMBU",
    "BlockSpec",
    "DEQConfig",
    "DEQNotContractive",
    "DEQResult",
    "DEQSolverUnknown",
    "EquivariantScan",
    "GradientSecant",
    "GrowStrategy",
    "GrowableOMBU",
    "GrowableOperatorMultiBiasUnit",
    "JetLineSearchConfig",
    "LearnablePReLU",
    "LineSearchResult",
    "MultiPackUnit",
    "OMBU",
    "OperatorBlock",
    "OperatorMultiBiasUnit",
    "TemperedActivation",
    "WeightLossJetSpec",
    "__lineage__",
    "__version__",
    "affine_jet",
    "affine_jet_mv",
    "analytic_gaussian_taps",
    "antiderivative_jet",
    "arrangement_w_block",
    "block_direction",
    "block_exact_search",
    "block_exact_sweep",
    "cmbConv1d",
    "cmbConv2d",
    "cmbLinear",
    "compose_jet",
    "compose_jet_mv",
    "compose_jet_riccati",
    "deq_du_dW",
    "deq_solve",
    "deq_vjp",
    "derivative_jet",
    "get_activation",
    "grow_ombu",
    "identity_jet",
    "is_registered",
    "jet_attention",
    "jet_exp",
    "jet_gradient",
    "jet_hessian",
    "jet_line_search",
    "jet_line_search_on_ray",
    "jet_multiply",
    "jet_partials",
    "jet_reciprocal",
    "jet_softmax",
    "jet_to_tower",
    "last_linear_block",
    "layer_jet",
    "layer_jet_mv",
    "lhopital_ratio",
    "limit_of_ratio",
    "list_activations",
    "mlp_jet",
    "mlp_jet_mv",
    "multipack_response",
    "ombu_bias_block",
    "one_layer_loss",
    "one_layer_loss_grad",
    "one_layer_loss_hessian",
    "one_layer_loss_jet",
    "one_layer_newton_direction",
    "refine",
    "register_activation",
    "removable_value",
    "scan_response",
    "soft_argmax_offset",
    "steerable_basis",
    "tower_to_jet",
]

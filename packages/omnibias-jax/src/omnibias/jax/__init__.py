# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX activation derivatives and Taylor jets for physics-informed models.

Shared polynomial coefficients define the derivative tower. Operator blocks,
mixed jets, and PINN architectures reuse it without nested coordinate autodiff.
Architecture and optimization helpers are available in their own modules.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-jax")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

from omnibias.jax.activations import (
    JaxActivationSpec,
    get_activation,
    is_registered,
    list_activations,
    register_activation,
)
from omnibias.jax.bo_derivatives import (
    coulomb_potential,
    make_bo_force,
    make_bo_hessian,
    make_local_energy,
    vibrational_frequencies,
)
from omnibias.jax.implicit import (
    DEQConfig,
    DEQNotContractive,
    DEQResult,
    DEQSolverUnknown,
    deq_du_dW,
    deq_solve,
    deq_vjp,
)
from omnibias.jax.jet import (
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
from omnibias.jax.jet_mv import (
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
from omnibias.jax.laplacian import (
    neural_field_hessian,
    neural_field_laplacian,
    neural_field_value,
    neural_field_value_and_laplacian,
    neural_field_value_grad_hessian,
    neural_field_value_grad_laplacian,
)
from omnibias.jax.line_search import (
    GradientSecant,
    JetLineSearchConfig,
    LineSearchResult,
    jet_line_search,
    jet_line_search_on_ray,
)
from omnibias.jax.multipack import (
    BirkhoffOMBU,
    init_multipack,
    multipack_apply,
    multipack_response,
)
from omnibias.jax.optim_block_search import (
    BlockSpec,
    arrangement_w_block,
    block_direction,
    block_exact_search,
    block_exact_sweep,
    last_linear_block,
    ombu_bias_block,
)
from omnibias.jax.precision import X64_HINT, require_x64, x64_enabled
from omnibias.jax.refine import (
    AdaptivePackBank,
    bank_forward,
    init_pack_bank,
    refine,
)
from omnibias.jax.scan import (
    BankSpec,
    bias_scan,
    init_bias_scan,
    scan_response,
    soft_argmax_offset,
)
from omnibias.jax.scan_equivariant import equivariant_scan_apply, steerable_basis
from omnibias.jax.weight_loss_jet import (
    WeightLossJetSpec,
    one_layer_loss,
    one_layer_loss_grad,
    one_layer_loss_hessian,
    one_layer_loss_jet,
    one_layer_newton_direction,
)

# Limit family exposed as package metadata.
__lineage__ = "bias collapse"

__all__ = [
    "AdaptivePackBank",
    "BankSpec",
    "BirkhoffOMBU",
    "BlockSpec",
    "DEQConfig",
    "DEQNotContractive",
    "DEQResult",
    "DEQSolverUnknown",
    "GradientSecant",
    "JaxActivationSpec",
    "JetLineSearchConfig",
    "LineSearchResult",
    "WeightLossJetSpec",
    "X64_HINT",
    "__lineage__",
    "__version__",
    "affine_jet",
    "affine_jet_mv",
    "antiderivative_jet",
    "arrangement_w_block",
    "bank_forward",
    "bias_scan",
    "block_direction",
    "block_exact_search",
    "block_exact_sweep",
    "compose_jet",
    "compose_jet_mv",
    "compose_jet_riccati",
    "coulomb_potential",
    "deq_du_dW",
    "deq_solve",
    "deq_vjp",
    "derivative_jet",
    "equivariant_scan_apply",
    "get_activation",
    "identity_jet",
    "init_bias_scan",
    "init_multipack",
    "init_pack_bank",
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
    "make_bo_force",
    "make_bo_hessian",
    "make_local_energy",
    "mlp_jet",
    "mlp_jet_mv",
    "multipack_apply",
    "multipack_response",
    "neural_field_hessian",
    "neural_field_laplacian",
    "neural_field_value",
    "neural_field_value_and_laplacian",
    "neural_field_value_grad_hessian",
    "neural_field_value_grad_laplacian",
    "ombu_bias_block",
    "one_layer_loss",
    "one_layer_loss_grad",
    "one_layer_loss_hessian",
    "one_layer_loss_jet",
    "one_layer_newton_direction",
    "refine",
    "register_activation",
    "removable_value",
    "require_x64",
    "scan_response",
    "soft_argmax_offset",
    "steerable_basis",
    "tower_to_jet",
    "vibrational_frequencies",
    "x64_enabled",
]

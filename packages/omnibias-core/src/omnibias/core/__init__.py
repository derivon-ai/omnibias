# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.core: backend-agnostic mathematical core.

This subpackage exposes the polynomial coefficient generators that power
omnibias's closed-form n-th derivative kernels, plus the generic
:class:`ActivationSpec` protocol that backends specialise.

Public API:

* :func:`sigmoid_polynomial_coeffs` -- Eulerian polynomial recurrence,
  ``sigma^(n)(z) = P_n(sigmoid(z))``.
* :func:`tanh_polynomial_coeffs` -- Legendre-style recurrence,
  ``tanh^(n)(z) = T_n(tanh(z))``.
* :func:`hermite_coeffs` -- probabilist's Hermite polynomial coefficients,
  ``g^(n)(z) = (-1)^n He_n(z) g(z)`` for ``g(z) = exp(-z^2 / 2)``.
* :func:`bell_partial`, :func:`bell_complete`, :func:`bell_number`,
  :func:`faa_di_bruno_terms` -- Bell polynomials and the Faà di Bruno
  decomposition powering exact multi-layer (directional) jet composition.
* :func:`multi_indices`, :func:`multiply_table`, :func:`multi_index_factorial`,
  :func:`index_position`, :func:`num_multi_indices` -- multi-index
  combinatorics for the *multivariate* (multi-index) jet primitive.
* :class:`ActivationSpec` -- generic activation descriptor.
* :class:`BankSpec` -- offset / scale bank for the bias scan (theory 01-02).
* :class:`MollifierSpec` -- pack-as-mollifier algebra and certified tails (theory 01-05).
* :class:`BandPlan` -- order-as-frequency spectral design (theory 01-07).
* :class:`EqualitySystem` -- equality-locus residual / Jacobian (theory 01-09).
* :class:`HardyDictionary` -- conjugate Hilbert dictionary (theory 01-12).
* :class:`JetLineSearchConfig`, :func:`run_model_line_search` -- exact
  jet line-search algebra (theory 03-12).
* :class:`RefinePolicy`, :func:`refine_bank` -- adaptive pack
  refinement algebra (theory 03-13).
* :class:`ComposedCurvatureConfig`, :func:`select_composed_step` --
  composed-curvature joint Newton algebra (theory 08-02).
* :class:`SharpnessSchedule`, :func:`scheduled_value` -- map a Ritz
  ``lambda_max`` to cubic ``sigma`` or a learning rate (theory 08-06).
* :class:`BlockSpec`, :func:`block_exact_search` algebra -- structured
  03-12 search on one coordinate block (theory 08-07).
* :class:`LocalJetConfig`, :class:`LocalJetForbidden` -- depth-causal
  local jet flood forbid and invert-and-match (theory 08-03).
* :class:`DEQConfig`, :class:`DEQNotContractive` -- implicit DEQ
  contraction raise and Newton / Banach budget (theory 08-08).
* :class:`MomentSystem`, :func:`solve_rule` -- neural quadrature
  from pack moments with a Peano enclosure (theory 03-06).
* :class:`ScaleBand`, :class:`ScaledPack`, :func:`rescale_pack` --
  exact scale-flow algebra and linear coarse-graining (theory 03-07).

There are no framework dependencies in this package.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.core.bell import (
    bell_complete,
    bell_number,
    bell_partial,
    faa_di_bruno_terms,
)
from omnibias.core.block_search import (
    BlockKind,
    BlockSpec,
    apply_block_step,
    arrangement_w_block,
    default_block_config,
    last_linear_block,
    ombu_bias_block,
    resolve_block_mask,
    unit_direction_from_mask,
)
from omnibias.core.composed_curvature import (
    ComposedCurvatureConfig,
    ComposedCurvatureReport,
    chain_rule_mse_blocks,
    eigh_symmetric,
    eval_tanh_derivative,
    reject_full_parameter_jacobian,
    scalar_nest_hessian,
    select_composed_step,
    solve_dense,
    symmetrize,
)
from omnibias.core.conjugate import (
    HardyAtom,
    HardyDictionary,
    hardy_conjugate_dictionary,
)
from omnibias.core.conjugate import (
    evaluate as evaluate_hardy_dictionary,
)
from omnibias.core.conjugate import (
    hilbert as hilbert_hardy_dictionary,
)
from omnibias.core.cubature import (
    MomentSystem,
    QuadratureRule,
    apply_rule,
    certified_error,
    design_rule,
    pack_moment,
    peano_kernel,
    solve_rule,
    target_moments,
)
from omnibias.core.frames import (
    FrameSpec,
    admissibility_constant,
    compile_bank,
    dilated_sigma_n,
    vanishing_moments,
)
from omnibias.core.hierarchy import Cluster, build_pack_tree, hierarchical_value, truncation_bound
from omnibias.core.implicit import (
    DEQConfig,
    DEQNotContractive,
    DEQSolverUnknown,
    honesty_payload,
    reject_anderson,
    reject_deq_contraction,
    spectral_radius_inf_bound,
)
from omnibias.core.information import (
    binary_entropy,
    has_cumulant_tower,
    is_log_partition_activation,
)
from omnibias.core.jets import contact_residual, is_holonomic
from omnibias.core.ladder import Normalization, hermite_function, tower_lower, tower_raise
from omnibias.core.line_search import (
    JetLineSearchConfig,
    LineSearchResult,
    certified_truncation_radius,
    polynomial_wolfe,
    run_model_line_search,
    select_model_step,
    taylor_coeffs_from_derivatives,
)
from omnibias.core.local_jet import (
    LocalJetConfig,
    LocalJetForbidden,
    LocalJetReport,
    invert_sigma,
    mlp_param_count,
    reject_local_jet_flood,
    require_invertible_sigma,
)
from omnibias.core.locus import (
    AffineSet,
    EqualitySystem,
    NewtonResult,
    UnitTerm,
    affine_locus,
    certify_locus_point,
)
from omnibias.core.mollifier import (
    MollifierSpec,
    design_order,
    is_admissible,
    moments,
    tail_bound,
)
from omnibias.core.moments import (
    central_moments_from_cumulants,
    central_to_raw_moments,
    cumulants_from_raw_moments,
    delta_method_central_moments,
    delta_method_from_cumulants,
    gaussian_central_moments,
    raw_moments_from_cumulants,
    raw_to_central_moments,
    second_order_delta,
)
from omnibias.core.multi_index import (
    index_position,
    multi_index_factorial,
    multi_indices,
    multiply_table,
    num_multi_indices,
)
from omnibias.core.multipack import (
    MultiPackSpec,
    PackSpec,
    central_stencil_weights,
    incidence_matrix,
    is_poised,
    polya_condition,
)
from omnibias.core.polynomials import (
    hermite_coeffs,
    mish_inner_coeffs,
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
)
from omnibias.core.probability import (
    cdf_normalization,
    dkw_epsilon,
    is_cdf_activation,
)
from omnibias.core.refine import (
    Indicator,
    RefinedPack,
    RefinePolicy,
    RefineReport,
    assert_zero_perturbation,
    hp_decision,
    local_scale_from_derivatives,
    refine_bank,
)
from omnibias.core.scale import (
    EffectiveOperator,
    FlowSystem,
    ScaleBand,
    ScaledPack,
    coarse_grain_linear,
    eval_gaussian_derivative,
    flow_coefficients,
    gram_matrix,
    overlap,
    report_exponents,
    rescale_pack,
    stiffness_matrix,
)
from omnibias.core.sharpness import (
    SharpnessReport,
    SharpnessSchedule,
    make_report,
    scheduled_value,
)
from omnibias.core.spec import (
    ActivationSpec,
    NthDerivativeFn,
    TensorFn,
    TensorT,
    TransformKernels,
    make_tempered_fastpath,
    make_tempered_transforms,
    tempered,
)
from omnibias.core.spectral_design import (
    BandPlan,
    alpha_for_peak,
    design_band_plan,
    peak_frequency,
    relative_bandwidth,
    response_profile,
)
from omnibias.core.tanh_method import TravellingWaveAnsatz, verify_exact
from omnibias.core.transfer import Layer, certified_band_gap
from omnibias.core.transforms import (
    TransformIdentity,
    TransformName,
    registered_activations,
)
from omnibias.core.transforms_pde import (
    LinearizingTransform,
    cole_hopf_jet,
    cole_hopf_u,
    factorial_jet_multiply,
    factorial_jet_reciprocal,
    verify_cole_hopf_burgers_jet,
    verify_transform,
)
from omnibias.core.weight_loss_jet import (
    WeightLossJetSpec,
    eval_sigma_derivative,
    one_layer_loss,
    one_layer_loss_grad,
    one_layer_loss_hessian,
    one_layer_loss_jet,
    one_layer_newton_direction,
    one_layer_output_jet,
    one_layer_param_count,
    pack_one_layer_params,
    unpack_one_layer_params,
)

try:
    __version__ = _pkg_version("omnibias-core")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "ActivationSpec",
    "AffineSet",
    "BandPlan",
    "BankSpec",
    "BlockKind",
    "BlockSpec",
    "Cluster",
    "ComposedCurvatureConfig",
    "ComposedCurvatureReport",
    "DEQConfig",
    "DEQNotContractive",
    "DEQSolverUnknown",
    "EffectiveOperator",
    "EqualitySystem",
    "FlowSystem",
    "FrameSpec",
    "HardyAtom",
    "HardyDictionary",
    "Indicator",
    "JetLineSearchConfig",
    "Layer",
    "LineSearchResult",
    "LinearizingTransform",
    "LocalJetConfig",
    "LocalJetForbidden",
    "LocalJetReport",
    "MollifierSpec",
    "MomentSystem",
    "MultiPackSpec",
    "NewtonResult",
    "Normalization",
    "NthDerivativeFn",
    "PackSpec",
    "QuadratureRule",
    "RefinePolicy",
    "RefineReport",
    "RefinedPack",
    "ScaleBand",
    "ScaledPack",
    "SharpnessReport",
    "SharpnessSchedule",
    "TensorFn",
    "TensorT",
    "TransformIdentity",
    "TransformKernels",
    "TransformName",
    "TravellingWaveAnsatz",
    "UnitTerm",
    "WeightLossJetSpec",
    "__lineage__",
    "__version__",
    "admissibility_constant",
    "affine_locus",
    "alpha_for_peak",
    "apply_block_step",
    "apply_rule",
    "arrangement_w_block",
    "assert_zero_perturbation",
    "bell_complete",
    "bell_number",
    "bell_partial",
    "binary_entropy",
    "build_pack_tree",
    "cdf_normalization",
    "central_moments_from_cumulants",
    "central_stencil_weights",
    "central_to_raw_moments",
    "certified_band_gap",
    "certified_error",
    "certified_truncation_radius",
    "certify_locus_point",
    "chain_rule_mse_blocks",
    "coarse_grain_linear",
    "cole_hopf_jet",
    "cole_hopf_u",
    "compile_bank",
    "contact_residual",
    "cumulants_from_raw_moments",
    "default_block_config",
    "delta_method_central_moments",
    "delta_method_from_cumulants",
    "design_band_plan",
    "design_order",
    "design_rule",
    "dilated_sigma_n",
    "dkw_epsilon",
    "eigh_symmetric",
    "eval_gaussian_derivative",
    "eval_sigma_derivative",
    "eval_tanh_derivative",
    "evaluate_hardy_dictionary",
    "factorial_jet_multiply",
    "factorial_jet_reciprocal",
    "flow_coefficients",
    "gaussian_central_moments",
    "gram_matrix",
    "hardy_conjugate_dictionary",
    "has_cumulant_tower",
    "hermite_coeffs",
    "hermite_function",
    "hierarchical_value",
    "hilbert_hardy_dictionary",
    "honesty_payload",
    "hp_decision",
    "incidence_matrix",
    "index_position",
    "invert_sigma",
    "is_admissible",
    "is_cdf_activation",
    "is_holonomic",
    "is_log_partition_activation",
    "is_poised",
    "last_linear_block",
    "local_scale_from_derivatives",
    "make_report",
    "make_tempered_fastpath",
    "make_tempered_transforms",
    "mish_inner_coeffs",
    "mlp_param_count",
    "moments",
    "multi_index_factorial",
    "multi_indices",
    "multiply_table",
    "num_multi_indices",
    "ombu_bias_block",
    "one_layer_loss",
    "one_layer_loss_grad",
    "one_layer_loss_hessian",
    "one_layer_loss_jet",
    "one_layer_newton_direction",
    "one_layer_output_jet",
    "one_layer_param_count",
    "overlap",
    "pack_moment",
    "pack_one_layer_params",
    "peak_frequency",
    "peano_kernel",
    "polya_condition",
    "polynomial_wolfe",
    "raw_moments_from_cumulants",
    "raw_to_central_moments",
    "refine_bank",
    "registered_activations",
    "reject_anderson",
    "reject_deq_contraction",
    "reject_full_parameter_jacobian",
    "reject_local_jet_flood",
    "relative_bandwidth",
    "report_exponents",
    "require_invertible_sigma",
    "rescale_pack",
    "resolve_block_mask",
    "response_profile",
    "run_model_line_search",
    "scalar_nest_hessian",
    "scheduled_value",
    "second_order_delta",
    "select_composed_step",
    "select_model_step",
    "sigmoid_polynomial_coeffs",
    "solve_dense",
    "solve_rule",
    "spectral_radius_inf_bound",
    "stiffness_matrix",
    "symmetrize",
    "tail_bound",
    "tanh_polynomial_coeffs",
    "target_moments",
    "taylor_coeffs_from_derivatives",
    "tempered",
    "tower_lower",
    "tower_raise",
    "truncation_bound",
    "unit_direction_from_mask",
    "unpack_one_layer_params",
    "vanishing_moments",
    "verify_cole_hopf_burgers_jet",
    "verify_exact",
    "verify_transform",
]

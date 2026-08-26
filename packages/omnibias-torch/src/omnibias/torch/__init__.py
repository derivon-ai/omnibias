# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.torch: PyTorch backend for omnibias.

Public API:

- :class:`OperatorMultiBiasUnit` (alias :class:`OMBU`): the trainable
  scalar-operator primitive.
- :class:`GrowableOperatorMultiBiasUnit` (alias :class:`GrowableOMBU`):
  the OMBU primitive with a learnable K (curriculum-style annealing).
- :class:`OperatorBlock`: typed wrapper that selects K and stencil based on
  ``op="grad"|"laplacian"|"derivative"|"band"|"integral"|"identity"``.
- :class:`cmbLinear`, :class:`cmbConv1d`, :class:`cmbConv2d`: drop-in
  replacements for :class:`torch.nn.Linear` / :class:`Conv1d` / :class:`Conv2d`
  with an inline :class:`OperatorBlock`.
- :func:`get_activation`, :func:`list_activations`,
  :func:`register_activation`, :func:`is_registered`: registry accessors for
  the activation dictionary.
- :mod:`omnibias.torch.architectures`: the three reference architectures
  (:class:`PINNHeat`, :class:`CmbNet`, :class:`CvxLasso`,
  :class:`CvxLogistic`).

Theory references:

- ``Lemma identity``: tied biases plus signs summing to one give bit-identical
  recovery of the base activation. See ``docs/theory.md`` and the multi-bias
  paper's Lemma 1.
- ``Lemma collapse``: the K-bias unit at the rescaled forward-difference
  stencil converges to the (K-1)-th derivative of the base activation as
  ``delta -> 0``. See ``docs/theory.md`` and the multi-bias paper's Lemma 2.
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
from omnibias.torch.conjugate import hardy_atoms, hilbert_coeffs
from omnibias.torch.growable import GrowableOperatorMultiBiasUnit, GrowStrategy
from omnibias.torch.hierarchy import hierarchical_scan
from omnibias.torch.implicit import (
    DEQConfig,
    DEQNotContractive,
    DEQResult,
    DEQSolverUnknown,
    deq_du_dW,
    deq_solve,
    deq_vjp,
)
from omnibias.torch.information import (
    chi_squared_divergence,
    cross_entropy,
    entropy,
    exponential_family_cumulants,
    f_divergence,
    fisher_information,
    fit_natural_parameter,
    glm_mean,
    glm_variance,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    moment_match,
    mutual_information,
    renyi_divergence,
    renyi_entropy,
    sinkhorn_distance,
    sliced_wasserstein,
    total_variation_distance,
    tsallis_entropy,
    wasserstein1,
    wasserstein1_cdf,
    wasserstein2_gaussian,
    wassersteinp,
)
from omnibias.torch.line_search import (
    JetLineSearchConfig,
    LineSearchResult,
    jet_line_search,
    jet_line_search_on_ray,
)
from omnibias.torch.jet import (
    affine_jet,
    antiderivative_jet,
    compose_jet,
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
from omnibias.torch.moments import (
    delta_method_gaussian,
    delta_method_moments,
    gaussian_moment_propagation,
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
from omnibias.torch.optim_composed import (
    ComposedCurvatureConfig,
    ComposedCurvatureReport,
    composed_block_hessian,
    composed_curvature_step,
)
from omnibias.torch.optim_kantorovich import (
    CONTINUUM_PDE_CLAIM_KEY,
    FINITE_RESIDUAL_CLAIM,
    KantorovichAccept,
    approximate_inverse_jacobian,
    kantorovich_accept_step,
    kantorovich_gated_gauss_newton_step,
    polynomial_sqrt2_maps,
    select_accepted_params,
)
from omnibias.torch.optim_sharpness import (
    SharpnessReport,
    SharpnessSchedule,
    scheduled_value,
    sharpness_lambda_max,
    sharpness_scheduled_minimize,
    sharpness_scheduled_step,
)
from omnibias.torch.probability import (
    binned_calibration_error,
    cdf,
    empirical_band_mass,
    ks_statistic,
    model_band_mass,
    soft_histogram,
)
from omnibias.torch.refine import AdaptivePackBank, grow_ombu, refine
from omnibias.torch.scan import BankSpec, BiasScan, scan_response, soft_argmax_offset
from omnibias.torch.scan_equivariant import EquivariantScan, steerable_basis
from omnibias.torch.tempered_blocks import LearnablePReLU, TemperedActivation
from omnibias.torch.train_local import (
    LocalJetConfig,
    LocalJetForbidden,
    LocalJetReport,
    LocalLayerState,
    invert_sigma,
    local_jet_step,
    make_input_jet,
)
from omnibias.torch.transforms import (
    FourierTransform,
    LaplaceTransform,
    MellinTransform,
    TransformBlock,
    fermi_dirac_mellin,
    fourier_transform,
    has_transform,
    laplace_transform,
    mellin_transform,
    region_of_convergence,
)
from omnibias.torch.train_stack import (
    TrainStackConfig,
    TrainStackReport,
    recommended_stack_step,
    stack_minimize,
)
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

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
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
    "CONTINUUM_PDE_CLAIM_KEY",
    "ComposedCurvatureConfig",
    "ComposedCurvatureReport",
    "DEQConfig",
    "DEQNotContractive",
    "DEQResult",
    "DEQSolverUnknown",
    "EquivariantScan",
    "FINITE_RESIDUAL_CLAIM",
    "FourierTransform",
    "GrowStrategy",
    "GrowableOMBU",
    "GrowableOperatorMultiBiasUnit",
    "JetLineSearchConfig",
    "KantorovichAccept",
    "LaplaceTransform",
    "LearnablePReLU",
    "LineSearchResult",
    "LocalJetConfig",
    "LocalJetForbidden",
    "LocalJetReport",
    "LocalLayerState",
    "MellinTransform",
    "MultiPackUnit",
    "OMBU",
    "OperatorBlock",
    "OperatorMultiBiasUnit",
    "SharpnessReport",
    "SharpnessSchedule",
    "TemperedActivation",
    "TrainStackConfig",
    "TrainStackReport",
    "TransformBlock",
    "WeightLossJetSpec",
    "__lineage__",
    "__version__",
    "affine_jet",
    "affine_jet_mv",
    "analytic_gaussian_taps",
    "antiderivative_jet",
    "approximate_inverse_jacobian",
    "arrangement_w_block",
    "binned_calibration_error",
    "block_direction",
    "block_exact_search",
    "block_exact_sweep",
    "cdf",
    "chi_squared_divergence",
    "cmbConv1d",
    "cmbConv2d",
    "cmbLinear",
    "compose_jet",
    "compose_jet_mv",
    "composed_block_hessian",
    "composed_curvature_step",
    "cross_entropy",
    "delta_method_gaussian",
    "delta_method_moments",
    "deq_du_dW",
    "deq_solve",
    "deq_vjp",
    "derivative_jet",
    "empirical_band_mass",
    "entropy",
    "exponential_family_cumulants",
    "f_divergence",
    "fermi_dirac_mellin",
    "fisher_information",
    "fit_natural_parameter",
    "fourier_transform",
    "gaussian_moment_propagation",
    "get_activation",
    "glm_mean",
    "glm_variance",
    "grow_ombu",
    "hardy_atoms",
    "has_transform",
    "hellinger_distance",
    "hierarchical_scan",
    "hilbert_coeffs",
    "identity_jet",
    "invert_sigma",
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
    "js_divergence",
    "kantorovich_accept_step",
    "kantorovich_gated_gauss_newton_step",
    "kl_divergence",
    "ks_statistic",
    "laplace_transform",
    "last_linear_block",
    "layer_jet",
    "layer_jet_mv",
    "lhopital_ratio",
    "limit_of_ratio",
    "list_activations",
    "local_jet_step",
    "make_input_jet",
    "mellin_transform",
    "mlp_jet",
    "mlp_jet_mv",
    "model_band_mass",
    "moment_match",
    "multipack_response",
    "mutual_information",
    "ombu_bias_block",
    "one_layer_loss",
    "one_layer_loss_grad",
    "one_layer_loss_hessian",
    "one_layer_loss_jet",
    "one_layer_newton_direction",
    "polynomial_sqrt2_maps",
    "recommended_stack_step",
    "refine",
    "region_of_convergence",
    "register_activation",
    "removable_value",
    "renyi_divergence",
    "renyi_entropy",
    "scan_response",
    "scheduled_value",
    "select_accepted_params",
    "sharpness_lambda_max",
    "sharpness_scheduled_minimize",
    "sharpness_scheduled_step",
    "sinkhorn_distance",
    "sliced_wasserstein",
    "soft_argmax_offset",
    "soft_histogram",
    "stack_minimize",
    "steerable_basis",
    "total_variation_distance",
    "tower_to_jet",
    "tsallis_entropy",
    "wasserstein1",
    "wasserstein1_cdf",
    "wasserstein2_gaussian",
    "wassersteinp",
]

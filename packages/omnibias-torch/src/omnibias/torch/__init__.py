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
from omnibias.torch.growable import GrowableOperatorMultiBiasUnit, GrowStrategy
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
    jet_gradient,
    jet_hessian,
    jet_multiply,
    jet_partials,
    layer_jet_mv,
    mlp_jet_mv,
)
from omnibias.torch.moments import (
    delta_method_gaussian,
    delta_method_moments,
    gaussian_moment_propagation,
)
from omnibias.torch.probability import (
    binned_calibration_error,
    cdf,
    empirical_band_mass,
    ks_statistic,
    model_band_mass,
    soft_histogram,
)
from omnibias.torch.tempered_blocks import LearnablePReLU, TemperedActivation
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
from omnibias.torch.unit import OperatorMultiBiasUnit

OMBU = OperatorMultiBiasUnit  # short alias
GrowableOMBU = GrowableOperatorMultiBiasUnit  # short alias

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "ActivationSpec",
    "AnalyticGaussianConv1d",
    "AnalyticGaussianConv2d",
    "FourierTransform",
    "GrowStrategy",
    "GrowableOMBU",
    "GrowableOperatorMultiBiasUnit",
    "LaplaceTransform",
    "LearnablePReLU",
    "MellinTransform",
    "OMBU",
    "OperatorBlock",
    "OperatorMultiBiasUnit",
    "TemperedActivation",
    "TransformBlock",
    "__lineage__",
    "__version__",
    "affine_jet",
    "affine_jet_mv",
    "analytic_gaussian_taps",
    "antiderivative_jet",
    "binned_calibration_error",
    "cdf",
    "chi_squared_divergence",
    "cmbConv1d",
    "cmbConv2d",
    "cmbLinear",
    "compose_jet",
    "compose_jet_mv",
    "cross_entropy",
    "delta_method_gaussian",
    "delta_method_moments",
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
    "has_transform",
    "hellinger_distance",
    "identity_jet",
    "is_registered",
    "jet_gradient",
    "jet_hessian",
    "jet_multiply",
    "jet_partials",
    "jet_to_tower",
    "js_divergence",
    "kl_divergence",
    "ks_statistic",
    "laplace_transform",
    "layer_jet",
    "layer_jet_mv",
    "lhopital_ratio",
    "limit_of_ratio",
    "list_activations",
    "mellin_transform",
    "mlp_jet",
    "mlp_jet_mv",
    "model_band_mass",
    "moment_match",
    "mutual_information",
    "region_of_convergence",
    "register_activation",
    "removable_value",
    "renyi_divergence",
    "renyi_entropy",
    "sinkhorn_distance",
    "sliced_wasserstein",
    "soft_histogram",
    "total_variation_distance",
    "tower_to_jet",
    "tsallis_entropy",
    "wasserstein1",
    "wasserstein1_cdf",
    "wasserstein2_gaussian",
    "wassersteinp",
]

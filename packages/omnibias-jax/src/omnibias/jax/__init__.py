# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.jax: JAX backend for omnibias.

Mirrors :mod:`omnibias.torch` for JAX users. Same closed-form derivative
kernels (sigmoid via Eulerian polynomials, tanh via Legendre-style
recursion, Gaussian via probabilist's Hermite polynomials) and same
activation dictionary, with bit-identical polynomial coefficients
because both backends import :mod:`omnibias.core.polynomials`.

Public API:

* :class:`JaxActivationSpec` -- alias for
  :class:`omnibias.core.spec.ActivationSpec` pinned to :class:`jax.Array`.
* :func:`get_activation`, :func:`list_activations`,
  :func:`register_activation`, :func:`is_registered` -- registry accessors.
* :func:`neural_field_laplacian`, :func:`neural_field_hessian`,
  :func:`neural_field_value`, :func:`neural_field_value_and_laplacian`,
  :func:`neural_field_value_grad_laplacian`,
  :func:`neural_field_value_grad_hessian` -- closed-form Laplacian /
  Hessian primitives for the one-layer neural field
  ``f(x) = b + sum_h c_h sigma(W_h . x + b_h)`` on ``R^D``.
* :func:`coulomb_potential`, :func:`make_local_energy`,
  :func:`make_bo_force`, :func:`make_bo_hessian`,
  :func:`vibrational_frequencies` -- Born-Oppenheimer derivative kernels
  used by the nuclear-Hessian work.

* :func:`affine_jet`, :func:`compose_jet`, :func:`layer_jet`, :func:`mlp_jet`,
  :func:`tower_to_jet`, :func:`jet_to_tower` -- exact multi-layer directional
  Taylor jets via Faà di Bruno composition of the closed-form activation
  derivative towers.
* :func:`mlp_jet_mv`, :func:`layer_jet_mv`, :func:`compose_jet_mv`,
  :func:`jet_multiply`, :func:`affine_jet_mv`, :func:`identity_jet`,
  :func:`jet_partials`, :func:`jet_gradient`, :func:`jet_hessian` -- exact
  *multivariate* (multi-index) Taylor jets: every mixed partial up to total
  order ``N`` from a single forward pass.
* :func:`jet_reciprocal`, :func:`jet_exp`, :func:`jet_softmax`,
  :func:`jet_attention` -- jet algebra beyond the elementwise layer: rational
  maps, and the non-local attention block whose *coordinate* derivatives
  (not merely its score derivatives) stay closed form.

The complex-valued activation dictionary lives in
:mod:`omnibias.jax.activations_complex` and is imported on demand.

Importing :mod:`omnibias.jax` does **not** import the FermiNet bridge;
that lives in the separate :mod:`omnibias.ferminet` package.
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
from omnibias.jax.information import (
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
from omnibias.jax.jet import (
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
from omnibias.jax.moments import (
    delta_method_gaussian,
    delta_method_moments,
    gaussian_moment_propagation,
)
from omnibias.jax.multipack import (
    BirkhoffOMBU,
    init_multipack,
    multipack_apply,
    multipack_response,
)
from omnibias.jax.precision import X64_HINT, require_x64, x64_enabled
from omnibias.jax.probability import (
    binned_calibration_error,
    cdf,
    empirical_band_mass,
    ks_statistic,
    model_band_mass,
    soft_histogram,
)
from omnibias.jax.scan import (
    BankSpec,
    bias_scan,
    init_bias_scan,
    scan_response,
    soft_argmax_offset,
)
from omnibias.jax.transforms import (
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

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "BankSpec",
    "BirkhoffOMBU",
    "FourierTransform",
    "JaxActivationSpec",
    "LaplaceTransform",
    "MellinTransform",
    "TransformBlock",
    "X64_HINT",
    "__lineage__",
    "__version__",
    "affine_jet",
    "affine_jet_mv",
    "antiderivative_jet",
    "bias_scan",
    "binned_calibration_error",
    "cdf",
    "chi_squared_divergence",
    "compose_jet",
    "compose_jet_mv",
    "coulomb_potential",
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
    "init_bias_scan",
    "init_multipack",
    "is_registered",
    "jet_attention",
    "jet_exp",
    "jet_gradient",
    "jet_hessian",
    "jet_multiply",
    "jet_partials",
    "jet_reciprocal",
    "jet_softmax",
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
    "make_bo_force",
    "make_bo_hessian",
    "make_local_energy",
    "mellin_transform",
    "mlp_jet",
    "mlp_jet_mv",
    "model_band_mass",
    "moment_match",
    "multipack_apply",
    "multipack_response",
    "mutual_information",
    "neural_field_hessian",
    "neural_field_laplacian",
    "neural_field_value",
    "neural_field_value_and_laplacian",
    "neural_field_value_grad_hessian",
    "neural_field_value_grad_laplacian",
    "region_of_convergence",
    "register_activation",
    "removable_value",
    "renyi_divergence",
    "renyi_entropy",
    "require_x64",
    "scan_response",
    "sinkhorn_distance",
    "sliced_wasserstein",
    "soft_argmax_offset",
    "soft_histogram",
    "total_variation_distance",
    "tower_to_jet",
    "tsallis_entropy",
    "vibrational_frequencies",
    "wasserstein1",
    "wasserstein1_cdf",
    "wasserstein2_gaussian",
    "wassersteinp",
    "x64_enabled",
]

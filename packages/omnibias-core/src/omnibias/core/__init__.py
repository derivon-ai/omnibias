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
from omnibias.core.information import (
    binary_entropy,
    has_cumulant_tower,
    is_log_partition_activation,
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
from omnibias.core.scan import BankSpec
from omnibias.core.spectral_design import (
    BandPlan,
    alpha_for_peak,
    design_band_plan,
    peak_frequency,
    relative_bandwidth,
    response_profile,
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
from omnibias.core.transforms import (
    TransformIdentity,
    TransformName,
    registered_activations,
)

try:
    __version__ = _pkg_version("omnibias-core")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "ActivationSpec",
    "BandPlan",
    "BankSpec",
    "MollifierSpec",
    "MultiPackSpec",
    "NthDerivativeFn",
    "PackSpec",
    "TensorFn",
    "TensorT",
    "TransformIdentity",
    "TransformKernels",
    "TransformName",
    "__lineage__",
    "__version__",
    "alpha_for_peak",
    "bell_complete",
    "bell_number",
    "bell_partial",
    "binary_entropy",
    "cdf_normalization",
    "central_moments_from_cumulants",
    "central_stencil_weights",
    "central_to_raw_moments",
    "cumulants_from_raw_moments",
    "delta_method_central_moments",
    "delta_method_from_cumulants",
    "design_band_plan",
    "design_order",
    "dkw_epsilon",
    "faa_di_bruno_terms",
    "gaussian_central_moments",
    "has_cumulant_tower",
    "hermite_coeffs",
    "incidence_matrix",
    "index_position",
    "is_admissible",
    "is_cdf_activation",
    "is_log_partition_activation",
    "is_poised",
    "make_tempered_fastpath",
    "make_tempered_transforms",
    "mish_inner_coeffs",
    "moments",
    "multi_index_factorial",
    "multi_indices",
    "multiply_table",
    "num_multi_indices",
    "peak_frequency",
    "polya_condition",
    "raw_moments_from_cumulants",
    "raw_to_central_moments",
    "registered_activations",
    "relative_bandwidth",
    "response_profile",
    "second_order_delta",
    "sigmoid_polynomial_coeffs",
    "tail_bound",
    "tanh_polynomial_coeffs",
    "tempered",
]

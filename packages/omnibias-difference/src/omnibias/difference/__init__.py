# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias-difference -- the founding ``delta -> 0`` register for discrete calculus.

This package is the **founding bias collapse** the library is named for: ``K``
biases on a difference stencil (spread ``delta``), with signs
``s_k = (-1)^(K-k) C(K-1, k-1) / delta^(K-1)``, so

.. math::

    f_K(z) = \sum_k s_k\,\sigma(z + b_k) \;\longrightarrow\; \sigma^{(K-1)}(z + b_{\text{mean}})
        \qquad \text{as } \delta \to 0.

The many biases coalesce and the finite difference *becomes* the derivative
``sigma^(K-1)``. The closed-form tower evaluates that limit **exactly**, with no
``1/delta^(K-1)`` catastrophic cancellation.

This is a ``delta -> 0`` limit yielding a smooth **derivative** -- it is **not**
**temperature collapse** -- the ``beta -> inf`` *feasibility penalty* of
``omnibias-convex`` / ``-control`` / ``-routing`` (a 0/1 step). Same word, opposite limit: **do not
conflate** the two (see the ``omnibias-dev-core-concepts`` skill and
``docs/theory.md``).

Honesty labels used throughout: **closed-form** (the sigma / sech / tanh towers
and the exact integer/rational special-number coefficients) and **numerical**
(the finite-difference estimate and any floating-point asymptotic). There is no
``autodiff-exact`` path in this package.

The pure-Python core (:mod:`omnibias.difference._core`) imports only
``omnibias.core.*``. The optional :mod:`omnibias.difference.torch` /
:mod:`omnibias.difference.jax` twins add a bit-identical finite-difference
stencil operator and require the respective backend.

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.difference._core import (
    DerivativeEnclosure,
    DerivativeProofVerdict,
    DerivBound,
    DifferenceEstimate,
    FiniteDifferenceCertificate,
    IrregularStencil,
    RationalIdentityVerdict,
    ShefferClass,
    StencilRequest,
    TransferEstimate,
    accuracy_order,
    appell_sequence,
    apply_irregular_stencil,
    associated_sequence,
    bell_asymptotic_relative_error,
    bell_dobinski_enclosure,
    bell_number,
    bell_number_asymptotic,
    bell_number_asymptotic_refined,
    bernoulli_asymptotic,
    bernoulli_enclosure,
    bernoulli_number,
    bernoulli_polynomial,
    bernoulli_recurrence_identity,
    bernoulli_sign_certificate,
    binomial_coefficient,
    binomial_transform,
    catalan_asymptotic,
    cauchy_product,
    certified_derivative_enclosure,
    certified_fd_error,
    certified_fd_error_general,
    certified_irregular_error,
    check_derivative_certificate,
    check_identity_certificate,
    compose_series,
    compositional_inverse,
    connection_constants,
    delta_operator_apply,
    derivative_sign_certificate,
    dirichlet_beta_odd_enclosure,
    dominant_pole_coefficient_asymptotic,
    euler_asymptotic,
    euler_enclosure,
    euler_number,
    euler_polynomial,
    euler_recurrence_identity,
    eulerian_number,
    exponential_generating_coeffs,
    falling_factorial_coeffs,
    falling_to_monomial,
    finite_difference_estimate,
    forward_difference,
    inverse_binomial_transform,
    is_poised_exact,
    log_bell_number_asymptotic,
    log_bell_number_asymptotic_refined,
    monomial_to_falling,
    newton_forward_coeffs,
    newton_forward_value,
    offsets_exact,
    ordinary_from_exponential,
    pade_approximant,
    pade_certified_remainder,
    pade_evaluate,
    pade_evaluate_interval,
    physical_weights,
    pincherle_derivative,
    polya_screen,
    power_sum_coeffs,
    rational_ogf_coefficients,
    rational_ogf_growth_base,
    rational_series,
    rational_value_identity,
    recommended_bell_fallback_n,
    riordan_array,
    riordan_inverse,
    riordan_product,
    rising_factorial_coeffs,
    series_reciprocal,
    sheffer_classify,
    sheffer_sequence,
    shift_polynomial,
    sigma_deriv_bound,
    signs_exact,
    singular_template_coefficient,
    solve_irregular_stencil,
    stencil_offsets,
    stencil_signs,
    stirling_first_signed,
    stirling_first_signed_row,
    stirling_first_unsigned,
    stirling_second,
    stirling_second_asymptotic,
    stirling_second_row,
    thiele_evaluate,
    thiele_interpolation,
    transfer_theorem,
    umbral_composition,
    umbral_functional,
    zeta_int_enclosure,
    zeta_negative_odd_identity,
)

try:
    __version__ = _pkg_version("omnibias-difference")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "DerivBound",
    "DerivativeEnclosure",
    "DerivativeProofVerdict",
    "DifferenceEstimate",
    "FiniteDifferenceCertificate",
    "IrregularStencil",
    "RationalIdentityVerdict",
    "ShefferClass",
    "StencilRequest",
    "TransferEstimate",
    "__lineage__",
    "__version__",
    "accuracy_order",
    "appell_sequence",
    "apply_irregular_stencil",
    "associated_sequence",
    "bell_asymptotic_relative_error",
    "bell_dobinski_enclosure",
    "bell_number",
    "bell_number_asymptotic",
    "bell_number_asymptotic_refined",
    "bernoulli_asymptotic",
    "bernoulli_enclosure",
    "bernoulli_number",
    "bernoulli_polynomial",
    "bernoulli_recurrence_identity",
    "bernoulli_sign_certificate",
    "binomial_coefficient",
    "binomial_transform",
    "catalan_asymptotic",
    "cauchy_product",
    "certified_derivative_enclosure",
    "certified_fd_error",
    "certified_fd_error_general",
    "certified_irregular_error",
    "check_derivative_certificate",
    "check_identity_certificate",
    "compose_series",
    "compositional_inverse",
    "connection_constants",
    "delta_operator_apply",
    "derivative_sign_certificate",
    "dirichlet_beta_odd_enclosure",
    "dominant_pole_coefficient_asymptotic",
    "euler_asymptotic",
    "euler_enclosure",
    "euler_number",
    "euler_polynomial",
    "euler_recurrence_identity",
    "eulerian_number",
    "exponential_generating_coeffs",
    "falling_factorial_coeffs",
    "falling_to_monomial",
    "finite_difference_estimate",
    "forward_difference",
    "inverse_binomial_transform",
    "is_poised_exact",
    "log_bell_number_asymptotic",
    "log_bell_number_asymptotic_refined",
    "monomial_to_falling",
    "newton_forward_coeffs",
    "newton_forward_value",
    "offsets_exact",
    "ordinary_from_exponential",
    "pade_approximant",
    "pade_certified_remainder",
    "pade_evaluate",
    "pade_evaluate_interval",
    "physical_weights",
    "pincherle_derivative",
    "polya_screen",
    "power_sum_coeffs",
    "rational_ogf_coefficients",
    "rational_ogf_growth_base",
    "rational_series",
    "rational_value_identity",
    "recommended_bell_fallback_n",
    "riordan_array",
    "riordan_inverse",
    "riordan_product",
    "rising_factorial_coeffs",
    "series_reciprocal",
    "sheffer_classify",
    "sheffer_sequence",
    "shift_polynomial",
    "sigma_deriv_bound",
    "signs_exact",
    "singular_template_coefficient",
    "solve_irregular_stencil",
    "stencil_offsets",
    "stencil_signs",
    "stirling_first_signed",
    "stirling_first_signed_row",
    "stirling_first_unsigned",
    "stirling_second",
    "stirling_second_asymptotic",
    "stirling_second_row",
    "thiele_evaluate",
    "thiele_interpolation",
    "transfer_theorem",
    "umbral_composition",
    "umbral_functional",
    "zeta_int_enclosure",
    "zeta_negative_odd_identity",
]

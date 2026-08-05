# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias.difference.umbral -- the umbral / Sheffer calculus surface (public namespace).

The discrete counterpart of the derivative tower, gathered into one public submodule. It
pairs the umbral *transforms* (from :mod:`omnibias.difference._core.umbral`) with the
*generative* operator layer (from :mod:`omnibias.difference._core.sheffer`):

* **Finite differences / interpolation** -- :func:`forward_difference`,
  :func:`newton_forward_coeffs`, :func:`newton_forward_value`, :func:`binomial_coefficient`.
* **Transforms** -- :func:`binomial_transform` / :func:`inverse_binomial_transform`,
  :func:`monomial_to_falling` / :func:`falling_to_monomial` (the Stirling transforms),
  :func:`connection_constants`.
* **Formal power series** -- :func:`compose_series`, :func:`compositional_inverse`,
  :func:`series_reciprocal`.
* **Sheffer sequences** -- :class:`ShefferClass` / :func:`sheffer_classify`,
  :func:`sheffer_sequence`, :func:`associated_sequence`, :func:`appell_sequence`,
  :func:`umbral_composition`.
* **Umbral operators** -- :func:`shift_polynomial` (``E^a``),
  :func:`delta_operator_apply` (``Q = f(D)``), :func:`pincherle_derivative`
  (``Q' = f'(D)``), :func:`umbral_functional` (the linear functional ``<L | x^k> = mu_k``).
* **Riordan group** -- :func:`riordan_array`, :func:`riordan_product`,
  :func:`riordan_inverse`.

Everything is exact :class:`~fractions.Fraction` arithmetic (**closed-form**). This is the
founding ``delta -> 0`` derivative register -- the biases coalesce and the finite difference
*becomes* the derivative. It is **not** the ``beta -> inf`` feasibility penalty of the
optimization packages, nor the ``q -> 1`` deformation of
:mod:`omnibias.qcalculus.umbral` (its quantum twin); same "collapse" word, different limits,
never conflated. Every symbol here is also re-exported flat from :mod:`omnibias.difference`
for backward compatibility.
"""

from __future__ import annotations

from omnibias.difference._core.sheffer import (
    associated_sequence,
    delta_operator_apply,
    pincherle_derivative,
    sheffer_sequence,
    shift_polynomial,
    umbral_composition,
    umbral_functional,
)
from omnibias.difference._core.umbral import (
    ShefferClass,
    appell_sequence,
    binomial_coefficient,
    binomial_transform,
    compose_series,
    compositional_inverse,
    connection_constants,
    falling_to_monomial,
    forward_difference,
    inverse_binomial_transform,
    monomial_to_falling,
    newton_forward_coeffs,
    newton_forward_value,
    riordan_array,
    riordan_inverse,
    riordan_product,
    series_reciprocal,
    sheffer_classify,
)

__all__ = [
    "ShefferClass",
    "appell_sequence",
    "associated_sequence",
    "binomial_coefficient",
    "binomial_transform",
    "compose_series",
    "compositional_inverse",
    "connection_constants",
    "delta_operator_apply",
    "falling_to_monomial",
    "forward_difference",
    "inverse_binomial_transform",
    "monomial_to_falling",
    "newton_forward_coeffs",
    "newton_forward_value",
    "pincherle_derivative",
    "riordan_array",
    "riordan_inverse",
    "riordan_product",
    "series_reciprocal",
    "sheffer_classify",
    "sheffer_sequence",
    "shift_polynomial",
    "umbral_composition",
    "umbral_functional",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias.qcalculus.umbral -- the q-umbral / q-Sheffer calculus surface (public namespace).

The quantum twin of :mod:`omnibias.difference.umbral`, gathered into one public submodule:

* **q-Stirling numbers & factorial bases** -- :func:`q_stirling_second`,
  :func:`q_stirling_first_signed`, their rows, and :func:`q_falling_factorial_coeffs` /
  :func:`q_rising_factorial_coeffs`.
* **q-transforms** -- :func:`q_binomial_transform` / :func:`q_inverse_binomial_transform`,
  :func:`q_monomial_to_falling` / :func:`q_falling_to_monomial`,
  :func:`q_newton_forward_coeffs` / :func:`q_newton_forward_value`.
* **q-Sheffer sequences** -- :class:`QShefferClass` / :func:`q_sheffer_classify`,
  :func:`q_sheffer_sequence`, :func:`q_associated_sequence`, :func:`q_appell_sequence`,
  :func:`q_umbral_composition`.
* **q-umbral operators** -- :func:`q_delta_operator_apply` (``Q = f(D_q)``),
  :func:`q_pincherle_derivative`.

Everything is exact :class:`~fractions.Fraction` arithmetic (**closed-form**), and carries
the exact q-Sheffer recurrence ``Q s_n = [n]_q s_{n-1}``. Every object reduces to its
classical :mod:`omnibias.difference.umbral` counterpart as **``q -> 1``**. That is a
**distinct** limit from the ``delta -> 0`` founding bias-collapse of
:mod:`omnibias.difference` and the ``beta -> inf`` **temperature collapse** feasibility penalty of the optimization
packages -- same "collapse" word, different limits, never conflated. Every symbol here is
also re-exported flat from :mod:`omnibias.qcalculus`.
"""

from __future__ import annotations

from omnibias.qcalculus._core.qstirling import (
    q_falling_factorial_coeffs,
    q_rising_factorial_coeffs,
    q_stirling_first_signed,
    q_stirling_first_signed_row,
    q_stirling_first_unsigned,
    q_stirling_second,
    q_stirling_second_row,
)
from omnibias.qcalculus._core.qumbral import (
    QShefferClass,
    q_appell_sequence,
    q_associated_sequence,
    q_binomial_transform,
    q_delta_operator_apply,
    q_falling_to_monomial,
    q_inverse_binomial_transform,
    q_monomial_to_falling,
    q_newton_forward_coeffs,
    q_newton_forward_value,
    q_pincherle_derivative,
    q_sheffer_classify,
    q_sheffer_sequence,
    q_umbral_composition,
)

__all__ = [
    "QShefferClass",
    "q_appell_sequence",
    "q_associated_sequence",
    "q_binomial_transform",
    "q_delta_operator_apply",
    "q_falling_factorial_coeffs",
    "q_falling_to_monomial",
    "q_inverse_binomial_transform",
    "q_monomial_to_falling",
    "q_newton_forward_coeffs",
    "q_newton_forward_value",
    "q_pincherle_derivative",
    "q_rising_factorial_coeffs",
    "q_sheffer_classify",
    "q_sheffer_sequence",
    "q_stirling_first_signed",
    "q_stirling_first_signed_row",
    "q_stirling_first_unsigned",
    "q_stirling_second",
    "q_stirling_second_row",
    "q_umbral_composition",
]

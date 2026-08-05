# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-Python q-calculus core (no torch / jax imports)."""

from __future__ import annotations

from omnibias.qcalculus._core.qderiv import (
    q_antiderivative_poly,
    q_derivative,
    q_derivative_poly,
    q_integral,
)
from omnibias.qcalculus._core.qhyper import (
    basic_hypergeometric,
    basic_hypergeometric_enclosure,
    q_exp_enclosure,
)
from omnibias.qcalculus._core.qnumbers import (
    q_binomial,
    q_binomial_poly,
    q_bracket,
    q_bracket_poly,
    q_factorial,
    q_pochhammer,
)
from omnibias.qcalculus._core.qspecial import (
    q_bernoulli,
    q_euler,
    q_exp,
    q_exp_big,
)
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
    "basic_hypergeometric",
    "basic_hypergeometric_enclosure",
    "q_antiderivative_poly",
    "q_appell_sequence",
    "q_associated_sequence",
    "q_bernoulli",
    "q_binomial",
    "q_binomial_poly",
    "q_binomial_transform",
    "q_bracket",
    "q_bracket_poly",
    "q_delta_operator_apply",
    "q_derivative",
    "q_derivative_poly",
    "q_euler",
    "q_exp",
    "q_exp_big",
    "q_exp_enclosure",
    "q_factorial",
    "q_falling_factorial_coeffs",
    "q_falling_to_monomial",
    "q_integral",
    "q_inverse_binomial_transform",
    "q_monomial_to_falling",
    "q_newton_forward_coeffs",
    "q_newton_forward_value",
    "q_pincherle_derivative",
    "q_pochhammer",
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

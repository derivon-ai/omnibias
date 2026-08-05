# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias-qcalculus: quantum / q-calculus on the omnibias tower.

Exact q-combinatorics (``[n]_q``, ``[n]_q!``, Gaussian binomial, ``(a;q)_n``), the Jackson
q-derivative / q-integral, the two q-exponentials, q-deformed Bernoulli / Euler numbers,
and basic hypergeometric series ``_r phi_s`` with certified geometric tails.

The **``q -> 1`` limit** recovers ordinary calculus (``[n]_q -> n``, ``D_q f -> f'``). This
is a **distinct** limit from the ``delta -> 0`` founding bias-collapse of
:mod:`omnibias.difference` and the ``beta -> inf`` **temperature collapse** feasibility penalty elsewhere -- same
spirit, different parameter, never conflated.

Backend twins (:mod:`omnibias.qcalculus.torch`, :mod:`omnibias.qcalculus.jax`) provide the
bit-identical Jackson q-derivative of the activation dictionary; import them explicitly.

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

from omnibias.qcalculus._core import (
    QShefferClass,
    basic_hypergeometric,
    basic_hypergeometric_enclosure,
    q_antiderivative_poly,
    q_appell_sequence,
    q_associated_sequence,
    q_bernoulli,
    q_binomial,
    q_binomial_poly,
    q_binomial_transform,
    q_bracket,
    q_bracket_poly,
    q_delta_operator_apply,
    q_derivative,
    q_derivative_poly,
    q_euler,
    q_exp,
    q_exp_big,
    q_exp_enclosure,
    q_factorial,
    q_falling_factorial_coeffs,
    q_falling_to_monomial,
    q_integral,
    q_inverse_binomial_transform,
    q_monomial_to_falling,
    q_newton_forward_coeffs,
    q_newton_forward_value,
    q_pincherle_derivative,
    q_pochhammer,
    q_rising_factorial_coeffs,
    q_sheffer_classify,
    q_sheffer_sequence,
    q_stirling_first_signed,
    q_stirling_first_signed_row,
    q_stirling_first_unsigned,
    q_stirling_second,
    q_stirling_second_row,
    q_umbral_composition,
)

try:
    __version__ = _pkg_version("omnibias-qcalculus")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "exempt: q->1 third limit"

__all__ = [
    "QShefferClass",
    "__lineage__",
    "__version__",
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

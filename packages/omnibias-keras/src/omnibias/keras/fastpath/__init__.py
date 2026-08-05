# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form derivative / integral fast paths for the Keras backend.

Every kernel is written against ``keras.ops`` and shares its polynomial
coefficients with the torch and JAX backends via
:mod:`omnibias.core.polynomials`.
"""

from omnibias.keras.fastpath.dispatch import (
    multibias_collapsed_forward,
    multibias_integral_forward,
    multibias_integral_window_forward,
    multibias_literal_forward,
)
from omnibias.keras.fastpath.eulerian import (
    sigmoid_nth_derivative,
    sigmoid_polynomial_coeffs,
    softplus_nth_derivative,
)
from omnibias.keras.fastpath.hermite import (
    gaussian_forward,
    gaussian_nth_derivative,
    hermite_coeffs,
)
from omnibias.keras.fastpath.legendre import (
    tanh_nth_derivative,
    tanh_polynomial_coeffs,
)

__all__ = [
    "gaussian_forward",
    "gaussian_nth_derivative",
    "hermite_coeffs",
    "multibias_collapsed_forward",
    "multibias_integral_forward",
    "multibias_integral_window_forward",
    "multibias_literal_forward",
    "sigmoid_nth_derivative",
    "sigmoid_polynomial_coeffs",
    "softplus_nth_derivative",
    "tanh_nth_derivative",
    "tanh_polynomial_coeffs",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form fast paths vs finite differences and shared coefficients."""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops
from omnibias.core.polynomials import (
    hermite_coeffs,
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
)
from omnibias.keras.fastpath import (
    gaussian_nth_derivative,
    sigmoid_nth_derivative,
    tanh_nth_derivative,
)


def _t(x: np.ndarray):
    return ops.convert_to_tensor(x)


def _np(x):
    return ops.convert_to_numpy(x)


def test_shared_polynomial_coeffs() -> None:
    assert sigmoid_polynomial_coeffs(1) == (0.0, 1.0, -1.0)
    assert tanh_polynomial_coeffs(1) == (1.0, 0.0, -1.0)
    assert hermite_coeffs(2) == (-1.0, 0.0, 1.0)


@pytest.mark.parametrize(
    "fn", [sigmoid_nth_derivative, tanh_nth_derivative, gaussian_nth_derivative]
)
def test_fastpath_order_zero_is_forward(fn) -> None:
    z = np.linspace(-2.0, 2.0, 11)
    out0 = _np(fn(_t(z), 0))
    assert out0.shape == z.shape
    assert np.all(np.isfinite(out0))


@pytest.mark.parametrize(
    "fn", [sigmoid_nth_derivative, tanh_nth_derivative, gaussian_nth_derivative]
)
def test_fastpath_finite_difference(fn) -> None:
    z_np = np.linspace(-2.0, 2.0, 21)
    # h=1e-3 keeps the check valid on both float32 and float64 backends.
    h = 1e-3
    for n in range(1, 5):
        analytic = _np(fn(_t(z_np), n))
        fd = (_np(fn(_t(z_np + h), n - 1)) - _np(fn(_t(z_np - h), n - 1))) / (2 * h)
        np.testing.assert_allclose(analytic, fd, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize(
    "fn", [sigmoid_nth_derivative, tanh_nth_derivative, gaussian_nth_derivative]
)
def test_negative_order_raises(fn) -> None:
    with pytest.raises(ValueError):
        fn(_t(np.zeros(3)), -1)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Piecewise (almost-everywhere) tower checks (Keras backend).

Bit-identical agreement with torch / jax is covered by
``tests/test_keras_parity.py``; here we pin the Keras-side a.e. contract:
registration, integer-order reduction, and the linear-piece ``n >= 2 -> 0``
convention via ``keras.ops``.
"""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops
from omnibias.keras.activations import get_activation, list_activations


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))


_PIECEWISE_NAMES = [
    "leaky_relu", "prelu", "relu6", "hardtanh", "hardsigmoid", "hardswish",
    "elu", "selu", "celu", "softshrink", "hardshrink", "threshold",
    "abs", "sign", "step", "softsign",
]
_LINEAR_PIECE_NAMES = [
    "leaky_relu", "prelu", "relu6", "hardtanh", "hardsigmoid",
    "softshrink", "hardshrink", "threshold", "abs",
]


def test_all_piecewise_registered() -> None:
    registered = set(list_activations())
    for name in _PIECEWISE_NAMES:
        assert name in registered, f"{name!r} not registered"


@pytest.mark.parametrize("name", _PIECEWISE_NAMES)
def test_order_reduction(name: str) -> None:
    spec = get_activation(name)
    z = ops.convert_to_tensor(np.linspace(-4.0, 4.0, 41))
    np.testing.assert_allclose(_np(spec.fastpath(z, 0)), _np(spec.forward(z)), atol=1e-9)
    assert spec.derivative is not None
    np.testing.assert_allclose(_np(spec.fastpath(z, 1)), _np(spec.derivative(z)), atol=1e-9)


@pytest.mark.parametrize("name", _LINEAR_PIECE_NAMES)
def test_linear_pieces_zero_from_order_two(name: str) -> None:
    spec = get_activation(name)
    z = ops.convert_to_tensor(np.linspace(-4.0, 4.0, 41))
    for n in (2, 3, 4):
        assert np.all(_np(spec.fastpath(z, n)) == 0.0), f"{name!r} order {n} not zero"


def test_relu_huber_ae_orders() -> None:
    z = ops.convert_to_tensor(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]))
    relu = get_activation("relu")
    assert np.all(_np(relu.fastpath(z, 2)) == 0.0)
    huber = get_activation("huber")
    np.testing.assert_allclose(
        _np(huber.fastpath(z, 2)), (np.abs(np.array([-2.0, -0.5, 0.0, 0.5, 2.0])) <= 1.0).astype(float)
    )
    assert np.all(_np(huber.fastpath(z, 3)) == 0.0)

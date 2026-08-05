# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Activation registry + closed-form fast-path correctness (Keras backend)."""

from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import ops
from omnibias.keras.activations import get_activation, list_activations


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))

_NAMES = [
    "sigmoid",
    "tanh",
    "softplus",
    "gaussian",
    "exp",
    "relu",
    "silu",
    "gelu",
    "huber",
    "arctan",
    "log1pu2",
    "sin",
    "cos",
    "sinh",
    "cosh",
    "tan",
    "cot",
    "coth",
    "sech",
    "log_cosh",
    "softabs",
    "smooth_sign",
    "mish",
]


def test_all_expected_activations_registered() -> None:
    registered = set(list_activations())
    for name in _NAMES:
        assert name in registered, f"{name!r} not registered"


def test_list_matches_torch_when_available() -> None:
    torch_reg = pytest.importorskip("omnibias.torch.activations.registry")
    assert sorted(torch_reg.list_activations()) == sorted(list_activations())


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "gaussian", "exp"])
def test_forward_matches_numpy_reference(name: str) -> None:
    z = np.linspace(-3.0, 3.0, 41)
    out = _np(get_activation(name).forward(ops.convert_to_tensor(z)))
    if name == "sigmoid":
        ref = 1.0 / (1.0 + np.exp(-z))
    elif name == "tanh":
        ref = np.tanh(z)
    elif name == "gaussian":
        ref = np.exp(-0.5 * z * z)
    else:
        ref = np.exp(z)
    # Tolerance is backend-dependent: the unified Keras backend may run in
    # float32 (TF/JAX defaults) or float64 (torch here). Use a tolerance
    # that holds on float32; bit-exact cross-backend agreement is covered
    # by the shared-coefficients test and the three-way parity test.
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_fastpath_negative_order_raises() -> None:
    z = ops.convert_to_tensor(np.zeros(3))
    with pytest.raises(ValueError):
        get_activation("tanh").fastpath(z, -1)


def test_fastpath_unsupported_order_raises_not_implemented() -> None:
    z = ops.convert_to_tensor(np.zeros(3))
    # relu now carries an all-orders a.e. tower; arctan genuinely caps at n = 2.
    with pytest.raises(NotImplementedError):
        get_activation("arctan").fastpath(z, 3)


@pytest.mark.parametrize(
    ("name", "max_order"),
    [("sigmoid", 4), ("tanh", 4), ("gaussian", 4), ("exp", 4)],
)
def test_fastpath_matches_finite_difference(name: str, max_order: int) -> None:
    spec = get_activation(name)
    z_np = np.linspace(-2.0, 2.0, 21)
    z = ops.convert_to_tensor(z_np)
    # h=1e-3 balances central-difference truncation (~h^2) against float32
    # cancellation (~eps/h), so the test holds on both float32 and float64.
    h = 1e-3
    for n in range(1, max_order + 1):
        analytic = _np(spec.fastpath(z, n))
        f_plus = _np(spec.fastpath(ops.convert_to_tensor(z_np + h), n - 1))
        f_minus = _np(spec.fastpath(ops.convert_to_tensor(z_np - h), n - 1))
        fd = (f_plus - f_minus) / (2.0 * h)
        np.testing.assert_allclose(analytic, fd, rtol=1e-2, atol=2e-3)


def test_floatx_is_float64_under_test() -> None:
    assert keras.config.floatx() == "float64"

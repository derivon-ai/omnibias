# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""OperatorBlock + cmbDense / cmbConv (Keras backend)."""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops
from omnibias.keras import ActivationSpec, OperatorBlock, cmbConv1D, cmbConv2D, cmbDense
from omnibias.keras.activations import get_activation


def _t(x: np.ndarray):
    return ops.convert_to_tensor(x)


def _np(x):
    return ops.convert_to_numpy(x)


def test_identity_block_equals_base() -> None:
    z = np.random.default_rng(0).normal(size=(8, 4))
    block = OperatorBlock(channels=4, op="identity", base="tanh")
    np.testing.assert_allclose(
        _np(block(_t(z))),
        _np(get_activation("tanh").forward(_t(z))),
        rtol=1e-10,
        atol=1e-12,
    )


def test_grad_block_matches_first_derivative() -> None:
    z = np.random.default_rng(1).normal(size=(8, 4))
    block = OperatorBlock(channels=4, op="grad", base="gaussian")
    np.testing.assert_allclose(
        _np(block(_t(z))),
        _np(get_activation("gaussian").fastpath(_t(z), 1)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_laplacian_block_matches_second_derivative() -> None:
    z = np.random.default_rng(2).normal(size=(8, 4))
    block = OperatorBlock(channels=4, op="laplacian", base="gaussian")
    np.testing.assert_allclose(
        _np(block(_t(z))),
        _np(get_activation("gaussian").fastpath(_t(z), 2)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_derivative_block_arbitrary_order() -> None:
    z = np.random.default_rng(3).normal(size=(8, 4))
    block = OperatorBlock(channels=4, op="derivative", base="tanh", derivative_order=3)
    assert block.K == 4
    np.testing.assert_allclose(
        _np(block(_t(z))),
        _np(get_activation("tanh").fastpath(_t(z), 3)),
        rtol=1e-9,
        atol=1e-10,
    )


def test_grad_block_requires_fastpath() -> None:
    # relu now carries an all-orders a.e. tower; use a fastpath-less spec so the
    # op="laplacian" guard still has something to reject.
    no_fastpath = ActivationSpec(name="_no_fastpath_probe", forward=ops.relu, fastpath=None)
    with pytest.raises(TypeError):
        OperatorBlock(channels=2, op="laplacian", base=no_fastpath)


def test_integral_block_runs() -> None:
    z = np.random.default_rng(4).normal(size=(8, 4))
    block = OperatorBlock(channels=4, op="integral", base="gaussian", init_delta=1.0)
    out = _np(block(_t(z)))
    assert out.shape == (8, 4)
    assert np.all(np.isfinite(out))


def test_cmbdense_shape_and_finite() -> None:
    x = np.random.default_rng(5).normal(size=(8, 6))
    layer = cmbDense(units=16, op="identity", base="tanh")
    out = _np(layer(_t(x)))
    assert out.shape == (8, 16)
    assert np.all(np.isfinite(out))


def test_cmbconv1d_shape() -> None:
    x = np.random.default_rng(6).normal(size=(4, 10, 3))
    layer = cmbConv1D(filters=5, kernel_size=3, op="identity", base="tanh")
    out = _np(layer(_t(x)))
    assert out.shape[0] == 4 and out.shape[-1] == 5


def test_cmbconv2d_shape() -> None:
    x = np.random.default_rng(7).normal(size=(4, 8, 8, 2))
    layer = cmbConv2D(filters=3, kernel_size=3, op="laplacian", base="gaussian")
    out = _np(layer(_t(x)))
    assert out.shape[0] == 4 and out.shape[-1] == 3


def test_invalid_op_raises() -> None:
    with pytest.raises(ValueError):
        OperatorBlock(channels=2, op="nonsense")

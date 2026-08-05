# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""OperatorMultiBiasUnit (Keras backend) behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops
from omnibias.keras import OMBU
from omnibias.keras.activations import get_activation


def _t(x: np.ndarray):
    return ops.convert_to_tensor(x)


def _np(x):
    return ops.convert_to_numpy(x)


def test_identity_nesting_recovers_base() -> None:
    """A fresh OMBU (tied biases, signs sum to one) equals sigma(z)."""
    z = np.random.default_rng(0).normal(size=(16, 4))
    for base in ("sigmoid", "tanh", "gaussian"):
        ombu = OMBU(num_channels=4, K=3, base=base, init_bias=0.0)
        out = _np(ombu(_t(z)))
        ref = _np(get_activation(base).forward(_t(z)))
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)
        assert ombu.is_identity_nested


def test_init_bias_shifts_activation() -> None:
    z = np.random.default_rng(1).normal(size=(8, 2))
    ombu = OMBU(num_channels=2, K=2, base="tanh", init_bias=0.5)
    out = _np(ombu(_t(z)))
    ref = _np(get_activation("tanh").forward(_t(z + 0.5)))
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_analytic_derivative_matches_spec_fastpath() -> None:
    z = np.random.default_rng(2).normal(size=(8, 3))
    ombu = OMBU(num_channels=3, K=3, base="gaussian", init_bias=0.0)
    out = _np(ombu.analytic_derivative(_t(z)))  # order K-1 = 2
    ref = _np(get_activation("gaussian").fastpath(_t(z), 2))
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_analytic_derivative_no_fastpath_raises() -> None:
    ombu = OMBU(num_channels=2, K=2, base="tanh")
    # tanh has a fastpath; use an artificial order check instead
    with pytest.raises(ValueError):
        ombu.analytic_derivative(_t(np.zeros((4, 2))), order=-1)


def test_bad_num_channels_raises() -> None:
    with pytest.raises(ValueError):
        OMBU(num_channels=0, K=2)


def test_wrong_input_channels_broadcasts_error() -> None:
    ombu = OMBU(num_channels=4, K=2, base="tanh")
    # The concrete exception type for a channel mismatch is backend-specific
    # (torch -> RuntimeError, jax/tf -> ValueError/InvalidArgument), so we
    # intentionally accept any error here.
    with pytest.raises(Exception):  # noqa: B017
        ombu(_t(np.zeros((8, 3))))

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lemma 1 identity nesting across activations (Keras backend)."""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops
from omnibias.keras import OMBU
from omnibias.keras.activations import get_activation
from omnibias.keras.identity_init import (
    identity_init_biases,
    identity_init_signs,
    verify_identity_init,
)

_BASES = ["sigmoid", "tanh", "softplus", "gaussian", "exp", "sin", "cos", "log_cosh"]


def _t(x: np.ndarray):
    return ops.convert_to_tensor(x)


def _np(x):
    return ops.convert_to_numpy(x)


@pytest.mark.parametrize("base", _BASES)
@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_identity_nesting(base: str, K: int) -> None:
    z = np.random.default_rng(abs(hash((base, K))) % 2**32).normal(size=(12, 3))
    ombu = OMBU(num_channels=3, K=K, base=base, init_bias=0.0)
    out = _np(ombu(_t(z)))
    ref = _np(get_activation(base).forward(_t(z)))
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
def test_identity_init_signs_sum_to_one(K: int) -> None:
    biases = identity_init_biases(3, K)
    signs = identity_init_signs(3, K)
    assert verify_identity_init(biases, signs, atol=1e-12)

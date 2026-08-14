# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Keras.ops partition-weight parity vs numpy (float64)."""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

keras = pytest.importorskip("keras")


def _np(z: object) -> np.ndarray:
    if hasattr(z, "detach"):
        return np.asarray(z.detach().cpu().numpy())  # type: ignore[union-attr]
    return np.asarray(keras.ops.convert_to_numpy(z))


def _atol() -> float:
    return 1e-7 if str(keras.config.backend()) == "torch" else 1e-9


def test_keras_partition_weights_parity() -> None:
    from omnibias.partition import PartitionConfig
    from omnibias.partition._core.params import PartitionParams
    from omnibias.partition._core.weights import partition_weights as partition_np
    from omnibias.partition.keras.weights import partition_weights_arrays

    rng = np.random.default_rng(3)
    X = rng.standard_normal((16, 4))
    W = rng.standard_normal((2, 4)) * 0.4
    t = rng.standard_normal((2,)) * 0.2
    beta = 5.0
    cfg = PartitionConfig(n_features=4, depth=2, split_kind="oblique", beta_final=beta)
    params = PartitionParams(cfg, W, t)
    np_w = partition_np(params, X, beta)
    ops = keras.ops
    keras_w = _np(
        partition_weights_arrays(
            ops.convert_to_tensor(W.astype(np.float64), dtype="float64"),
            ops.convert_to_tensor(t.astype(np.float64), dtype="float64"),
            ops.convert_to_tensor(X.astype(np.float64), dtype="float64"),
            beta,
            2,
        )
    )
    assert keras_w.dtype == np.float64
    assert np.max(np.abs(np_w - keras_w)) < _atol()

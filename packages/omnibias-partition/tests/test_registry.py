# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The RegionModels registry + the shared combine engine."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.partition import PartitionConfig, init_params, partition_weights
from omnibias.partition.registry import RegionModels, combine_outputs


def test_combine_outputs_matches_manual_blend() -> None:
    rng = np.random.default_rng(0)
    w = rng.uniform(size=(10, 4))
    w = w / w.sum(axis=1, keepdims=True)  # a partition of unity
    outs = rng.standard_normal((10, 4, 3))
    got = combine_outputs(w, outs)
    ref = np.einsum("nl,nlk->nk", w, outs)
    assert np.allclose(got, ref)


def test_region_models_combine_is_convex_blend() -> None:
    cfg = PartitionConfig(n_features=2, depth=1, seed=1)
    params = init_params(cfg, rng=1)
    consts = [0.0, 10.0]  # two constant region models
    models = [(lambda X, c=c: np.full((X.shape[0], 1), c)) for c in consts]
    reg = RegionModels(params, models)
    X = np.random.default_rng(2).standard_normal((25, 2))
    out = reg.combine(X, beta=3.0)
    w = partition_weights(params, X, beta=3.0)
    expected = w[:, 0] * consts[0] + w[:, 1] * consts[1]
    assert np.allclose(out, expected)
    assert np.all((out >= -1e-9) & (out <= 10.0 + 1e-9))  # bounded by the region values


def test_wrong_number_of_models_raises() -> None:
    cfg = PartitionConfig(n_features=2, depth=2, seed=0)  # 4 regions
    params = init_params(cfg)
    with pytest.raises(ValueError, match="region models"):
        RegionModels(params, [lambda X: X[:, :1]])

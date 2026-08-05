# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""GrowableOperatorMultiBiasUnit + KGrowthScheduler (Keras backend)."""

from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import ops
from omnibias.keras import GrowableOMBU, KGrowthScheduler


def _t(x: np.ndarray):
    return ops.convert_to_tensor(x)


def _np(x):
    return ops.convert_to_numpy(x)


def test_pair_growth_preserves_output() -> None:
    z = np.random.default_rng(0).normal(size=(8, 4))
    unit = GrowableOMBU(num_channels=4, init_K=1, K_max=5, base="tanh")
    before = _np(unit(_t(z)))
    added = unit.grow("pair")
    assert added == 2
    assert unit.active_K == 3
    after = _np(unit(_t(z)))
    np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-12)


def test_saturate_growth_friendly_only() -> None:
    unit = GrowableOMBU(num_channels=2, init_K=1, K_max=4, base="sigmoid")
    unit.grow("saturate")
    assert unit.active_K == 2

    bad = GrowableOMBU(num_channels=2, init_K=1, K_max=4, base="tanh")
    with pytest.raises(ValueError):
        bad.grow("saturate")


def test_grow_beyond_kmax_raises() -> None:
    unit = GrowableOMBU(num_channels=2, init_K=1, K_max=2, base="sigmoid")
    with pytest.raises(RuntimeError):
        unit.grow("pair")


def test_scheduler_grows_on_plateau() -> None:
    model = keras.Sequential(
        [keras.layers.Input(shape=(4,)), GrowableOMBU(num_channels=4, init_K=1, K_max=5)]
    )
    sched = KGrowthScheduler(model, patience=2, min_delta=1e-3, max_K=5, strategy="pair")
    # Feed a flat metric to trigger a plateau.
    for epoch in range(6):
        sched.step(1.0, epoch=epoch)
    assert sched.total_growth_events() >= 1


def test_scheduler_requires_growable_unit() -> None:
    model = keras.Sequential(
        [keras.layers.Input(shape=(4,)), keras.layers.Dense(4)]
    )
    with pytest.raises(ValueError):
        KGrowthScheduler(model)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soundness of the partition soft->hard gap certificate + the interval weight enclosure."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.partition import (
    PartitionConfig,
    certify_partition_gap,
    init_params,
    partition_weights,
)
from omnibias.partition._core.verified import (
    interval_weight_bounds,
    weight_rounding_gap,
)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("beta", [1.0, 5.0, 30.0])
def test_rounding_gap_bound_dominates_measured(depth: int, beta: float) -> None:
    cfg = PartitionConfig(n_features=4, depth=depth, seed=2)
    params = init_params(cfg, rng=1)
    # dense grid + random samples (the enclosure must contain both)
    grid = np.stack(np.meshgrid(*[np.linspace(-2, 2, 6)] * 2, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 2)
    grid = np.concatenate([grid, np.zeros((grid.shape[0], 2))], axis=1)  # pad to 4 features
    rand = np.random.default_rng(3).standard_normal((200, 4))
    X = np.concatenate([grid, rand], axis=0)
    bound, measured = weight_rounding_gap(params, X, beta)
    assert np.all(bound >= measured - 1e-9)  # sound
    assert np.all(bound <= 2.0 + 1e-12)  # L1 of two probability vectors is <= 2


def test_certificate_is_sound_and_hardens() -> None:
    cfg = PartitionConfig(n_features=3, depth=2, seed=4)
    params = init_params(cfg, rng=2)
    X = np.random.default_rng(5).standard_normal((150, 3))
    lo = certify_partition_gap(params, X, beta=2.0)
    hi = certify_partition_gap(params, X, beta=64.0)
    assert lo.is_sound and hi.is_sound
    assert hi.max_gap <= lo.max_gap + 1e-9  # sharper beta -> smaller certified gap
    assert hi.gibbs_scale < lo.gibbs_scale  # log(n_regions)/beta shrinks with beta


def test_interval_weight_bounds_enclose_samples_in_box() -> None:
    cfg = PartitionConfig(n_features=2, depth=2, seed=6)
    params = init_params(cfg, rng=3)
    box = np.array([[-1.0, -1.0], [1.0, 1.0]])  # (2, d) lo/hi
    beta = 6.0
    bounds = interval_weight_bounds(params, box, beta)
    # sample densely inside the box; every region weight must lie in its interval
    xs = np.random.default_rng(7).uniform(-1.0, 1.0, size=(500, 2))
    P = partition_weights(params, xs, beta)
    for region, iv in enumerate(bounds):
        assert iv.lo - 1e-9 <= P[:, region].min()
        assert P[:, region].max() <= iv.hi + 1e-9


def test_interval_weight_bounds_prove_containment_far_from_boundary() -> None:
    # a box entirely on one side of a single axis split -> that region's weight ~ 1
    cfg = PartitionConfig(n_features=1, depth=1, split_kind="axis", seed=0)
    params = init_params(cfg)
    params.W[:] = 1.0
    params.t[:] = 0.0
    box = np.array([[5.0], [10.0]])  # x in [5, 10], well past the x > 0 split
    bounds = interval_weight_bounds(params, box, beta=50.0)
    # region 1 ("gate fired") is certified to carry essentially all the mass
    assert bounds[1].lo > 0.99

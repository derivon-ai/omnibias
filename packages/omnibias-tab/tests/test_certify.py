# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certificate soundness: every enclosure must contain a dense grid AND a random sample."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.tab import (
    SoftTreeConfig,
    TabParams,
    certify_tab,
    certify_tab_gap,
    forward_np,
    init_params,
)
from omnibias.tab._core.verified import interval_output_bounds


def _box_and_samples(d: int, seed: int, per_axis: int = 4, n_rand: int = 200):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-2.0, -0.5, size=d)
    hi = rng.uniform(0.5, 2.0, size=d)
    box = np.stack([lo, hi])
    axes = [np.linspace(lo[f], hi[f], per_axis) for f in range(d)]
    grid = np.array(list(itertools.product(*axes))) if d <= 5 else np.empty((0, d))
    rand = rng.uniform(lo, hi, size=(n_rand, d))
    samples = np.vstack([grid, rand]) if grid.size else rand
    return box, samples


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("beta", [1.0, 5.0])
def test_output_bounds_enclose_samples(depth: int, beta: float) -> None:
    d = 4
    cfg = SoftTreeConfig(n_features=d, n_trees=5, depth=depth, task="regression", n_outputs=2, seed=depth)
    p = init_params(cfg, depth, leaf_scale=1.0)
    box, samples = _box_and_samples(d, seed=depth)
    bounds = interval_output_bounds(p, box, beta)
    F = forward_np(p, samples, beta)  # (n, 2)
    for c, iv in enumerate(bounds):
        assert iv.lo - 1e-9 <= F[:, c].min()
        assert F[:, c].max() <= iv.hi + 1e-9


@pytest.mark.parametrize("depth", [1, 2])
def test_lipschitz_bounds_true_sensitivity(depth: int) -> None:
    d = 4
    cfg = SoftTreeConfig(n_features=d, n_trees=6, depth=depth, task="binary", beta_final=3.0, seed=1)
    p = init_params(cfg, 1, leaf_scale=1.0)
    box, samples = _box_and_samples(d, seed=7, per_axis=3, n_rand=150)
    cert = certify_tab(p, box, beta=3.0, use_verify=False)
    F = forward_np(p, samples, 3.0)[:, 0]
    rng = np.random.default_rng(2)
    for _ in range(400):
        i, j = rng.integers(0, samples.shape[0], size=2)
        dist = float(np.linalg.norm(samples[i] - samples[j]))
        assert abs(F[i] - F[j]) <= cert.lipschitz * dist + 1e-9


def test_monotone_model_is_certified_increasing() -> None:
    d = 3
    cfg = SoftTreeConfig(n_features=d, n_trees=5, depth=1, task="binary", beta_final=3.0, seed=0)
    rng = np.random.default_rng(0)
    W = np.abs(rng.standard_normal((5, 1, d))) + 0.2       # all positive directions
    t = rng.standard_normal((5, 1)) * 0.5
    leaf0 = rng.standard_normal((5, 1, 1))
    leaf1 = leaf0 + (np.abs(rng.standard_normal((5, 1, 1))) + 0.2)  # u = leaf1 - leaf0 > 0
    leaves = np.concatenate([leaf0, leaf1], axis=1)
    p = TabParams(cfg, W, t, leaves, np.zeros(1))

    box, samples = _box_and_samples(d, seed=3, per_axis=4, n_rand=100)
    cert = certify_tab(
        p, box, monotone_features={f: +1 for f in range(d)}, beta=3.0, use_verify=False
    )
    assert cert.monotone_ok is True
    assert all(v == "increasing" for v in cert.monotonicity.values())

    # empirical confirmation: increasing any single feature never decreases the score
    F0 = forward_np(p, samples, 3.0)[:, 0]
    for f in range(d):
        bumped = samples.copy()
        bumped[:, f] += 0.3
        F1 = forward_np(p, bumped, 3.0)[:, 0]
        assert np.all(F1 >= F0 - 1e-9)


def test_nonmonotone_model_fails_the_constraint() -> None:
    cfg = SoftTreeConfig(n_features=4, n_trees=8, depth=2, task="binary", beta_final=4.0, seed=4)
    p = init_params(cfg, 4, leaf_scale=1.0)
    box, _ = _box_and_samples(4, seed=5)
    cert = certify_tab(p, box, monotone_features={0: +1, 1: +1}, beta=4.0, use_verify=False)
    # a generic random tree is not monotone in every feature -> the constraint must not pass
    assert cert.monotone_ok is False


def test_rounding_gap_is_sound_and_shrinks_with_beta() -> None:
    cfg = SoftTreeConfig(n_features=5, n_trees=6, depth=2, task="binary", seed=2)
    p = init_params(cfg, 2, leaf_scale=1.0)
    rng = np.random.default_rng(6)
    X = rng.standard_normal((128, 5))
    small = certify_tab_gap(p, X, beta=40.0)
    large = certify_tab_gap(p, X, beta=4.0)
    assert small.is_sound and large.is_sound
    assert small.max_gap >= small.measured_max - 1e-9  # certified bound dominates measured
    assert small.max_gap < large.max_gap               # hardening gap shrinks as beta -> inf


def test_verify_path_is_sound_when_available() -> None:
    # The additive-tier verify engine (branch-and-bound) is the tighter/sealed path; keep the
    # instance minimal because it is far heavier than the always-available interval engine.
    pytest.importorskip("torch")
    pytest.importorskip("omnibias.verify")
    d = 2
    cfg = SoftTreeConfig(n_features=d, n_trees=2, depth=1, task="binary", beta_final=3.0, seed=1)
    p = init_params(cfg, 1, leaf_scale=1.0)
    box, samples = _box_and_samples(d, seed=8, per_axis=6, n_rand=200)
    cert_v = certify_tab(p, box, monotone_features={}, beta=3.0, use_verify=True)
    cert_i = certify_tab(p, box, monotone_features={}, beta=3.0, use_verify=False)
    assert cert_v.method == "verify"
    assert cert_i.method == "interval"
    F = forward_np(p, samples, 3.0)[:, 0]
    # both engines are sound: their output enclosures contain the sampled outputs ...
    for cert in (cert_v, cert_i):
        lo, hi = cert.output_bounds[0]
        assert lo - 1e-9 <= F.min() and F.max() <= hi + 1e-9
    # ... and the sealed verify enclosure is no looser than the interval one.
    assert cert_v.output_bounds[0][0] >= cert_i.output_bounds[0][0] - 1e-9
    assert cert_v.output_bounds[0][1] <= cert_i.output_bounds[0][1] + 1e-9

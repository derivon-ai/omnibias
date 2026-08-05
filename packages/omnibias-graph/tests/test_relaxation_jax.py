# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable combinatorial relaxations (jax backend) -- key oracles."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

import omnibias.graph.jax.ops as G


def test_sinkhorn_doubly_stochastic() -> None:
    rng = np.random.default_rng(0)
    p = np.asarray(G.sinkhorn_normalize(jnp.asarray(rng.normal(size=(6, 6))), n_iters=300))
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-9)
    assert np.allclose(p.sum(axis=-2), 1.0, atol=1e-9)


def test_gumbel_sinkhorn_recovers_permutation() -> None:
    perm = np.array([4, 3, 2, 1, 0])
    log_alpha = np.full((5, 5), -5.0)
    for i, j in enumerate(perm):
        log_alpha[i, j] = 5.0
    p = np.asarray(G.gumbel_sinkhorn(jnp.asarray(log_alpha), temperature=0.05, n_iters=200))
    assert np.array_equal(p.argmax(axis=-1), perm)


@pytest.mark.parametrize("descending", [True, False])
def test_soft_sort_hard_limit(descending: bool) -> None:
    s = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.0])
    out = np.asarray(G.soft_sort(jnp.asarray(s), temperature=1e-4, descending=descending))
    ref = np.sort(s)
    if descending:
        ref = ref[::-1]
    assert np.allclose(out, ref, atol=1e-6)


def test_soft_top_k_sums_to_k_and_hard_limit() -> None:
    s = jnp.asarray([3.0, 1.0, 4.0, 1.5, 9.0, 2.0])
    for tau in (0.01, 0.5, 3.0):
        m = np.asarray(G.soft_top_k(s, 3, temperature=tau))
        assert abs(m.sum() - 3.0) < 1e-9
    m = np.asarray(G.soft_top_k(s, 3, temperature=1e-4))
    assert set(np.nonzero(m > 0.5)[0].tolist()) == {4, 2, 0}


def test_temperature_validation() -> None:
    with pytest.raises(ValueError, match="temperature"):
        G.soft_sort(jnp.zeros(3), temperature=0.0)

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified leaf-routing decision of a soft tree (omnibias.tab.decision).

The per-tree routing distribution P sums to one, so logits log(P)/beta reproduce P as a Gibbs
law and omnibias-struct's certify_argmax gives a faithful SelectionCertificate of tab's own
beta -> inf routing collapse: which leaf dominates, value gap <= log(2**depth)/beta, mode-mass
concentration, and the L^inf argmax-stability radius. No change to tab's trained forward.
"""

from __future__ import annotations

import math

import numpy as np
from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab._core.forward import leaf_memberships
from omnibias.tab._core.params import init_params
from omnibias.tab.decision import (
    certified_leaf_decision,
    certified_leaf_decisions,
    leaf_logits,
)


def _params(depth: int = 2, seed: int = 0):  # type: ignore[no-untyped-def]
    cfg = SoftTreeConfig(
        n_features=3, n_trees=2, depth=depth, n_outputs=1, task="regression", beta_final=16.0
    )
    return init_params(cfg, seed)


def test_routing_matches_memberships_and_is_sound() -> None:
    params = _params(depth=2)
    x = np.array([0.4, -0.7, 1.1])
    routing, cert = certified_leaf_decision(params, x, beta=16.0, tree=0)

    ref = leaf_memberships(params, x.reshape(1, -1), 16.0)[0, 0]
    assert np.allclose(routing, ref)
    assert routing.shape == (4,)  # 2**depth leaves
    assert math.isclose(float(routing.sum()), 1.0, rel_tol=1e-12)
    assert cert.is_sound
    # the certified mode is the leaf tab actually routes most mass to
    assert cert.argmax == int(np.argmax(ref))
    # softmax(beta * logits) reproduces the routing distribution exactly
    assert math.isclose(cert.p_max, float(np.max(ref)), rel_tol=1e-9)
    # value gap is bounded by log(n_leaves)/beta
    assert cert.gap_bound <= math.log(4) / 16.0 + 1e-12


def test_logits_reproduce_routing_distribution() -> None:
    params = _params(depth=2)
    X = np.array([[0.4, -0.7, 1.1], [-1.0, 0.5, 0.2]])
    beta = 16.0
    s = leaf_logits(params, X, beta)  # (n, T, L)
    # softmax(beta * s) == leaf_memberships
    z = beta * s
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    recovered = ez / ez.sum(axis=-1, keepdims=True)
    assert np.allclose(recovered, leaf_memberships(params, X, beta), atol=1e-9)


def test_decision_hardens_with_beta() -> None:
    params = _params(depth=2)
    x = np.array([0.9, -0.2, 0.6])
    _, c_soft = certified_leaf_decision(params, x, beta=4.0)
    _, c_sharp = certified_leaf_decision(params, x, beta=64.0)
    # both sound; the gap shrinks and the mode mass concentrates as beta grows
    assert c_soft.is_sound and c_sharp.is_sound
    assert c_sharp.gap_bound < c_soft.gap_bound
    assert c_sharp.p_max >= c_soft.p_max


def test_batched_and_stability_radius() -> None:
    params = _params(depth=2)
    X = np.array([[0.4, -0.7, 1.1], [-1.0, 0.5, 0.2], [0.0, 0.0, 0.0]])
    out = certified_leaf_decisions(params, X, beta=16.0, tree=1, eps=0.0)
    assert len(out) == 3
    for routing, cert in out:
        assert cert.is_sound
        assert math.isclose(float(routing.sum()), 1.0, rel_tol=1e-12)
        # robust radius is the top-two logit half-margin
        assert cert.robust_radius >= 0.0


def test_bad_tree_index_raises() -> None:
    params = _params(depth=1)
    try:
        certified_leaf_decision(params, np.zeros(3), tree=5)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for an out-of-range tree index")

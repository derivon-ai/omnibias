# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cardinality surrogates and gate-lifecycle helpers (torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.shape.torch import ops as shape  # noqa: E402


def test_l0_surrogate_sum_and_concave():
    gates = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert abs(float(shape.l0_surrogate(gates, kind="sum")) - 2.0) < 1e-6
    concave = float(shape.l0_surrogate(gates, kind="concave", eps=1e-3))
    assert abs(concave - 2.0) < 1e-2
    with pytest.raises(ValueError):
        shape.l0_surrogate(gates, kind="bogus")


def test_anneal_lambda_endpoints_and_monotone():
    kw = dict(lam_start=0.01, lam_end=1.0, num_steps=100)
    assert abs(shape.anneal_lambda(0, **kw) - 0.01) < 1e-9
    assert abs(shape.anneal_lambda(100, **kw) - 1.0) < 1e-9
    seq = [shape.anneal_lambda(s, **kw) for s in range(0, 101, 10)]
    assert all(b >= a for a, b in zip(seq, seq[1:], strict=False))
    for sched in ("linear", "exp", "cosine"):
        assert abs(shape.anneal_lambda(100, schedule=sched, **kw) - 1.0) < 1e-6
    with pytest.raises(ValueError):
        shape.anneal_lambda(0, lam_start=1.0, lam_end=2.0, num_steps=0)


def test_prune_inactive_keeps_active_and_never_empties():
    centers = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    logits = torch.tensor([3.0, -3.0, 3.0])
    c2, l2 = shape.prune_inactive(centers, logits, threshold=0.5)
    assert c2.shape[0] == 2 and l2.shape[0] == 2
    dead = torch.tensor([-5.0, -6.0, -7.0])
    c3, _ = shape.prune_inactive(centers, dead, threshold=0.5)
    assert c3.shape[0] == 1  # keeps the least-dead shape

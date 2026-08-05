# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Differentiable relaxation layers: feasibility, torch<->jax parity, gradients."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.routing import RelaxationSchedule

KINDS = ("assignment", "flow", "held_karp")


def _cost(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    d = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def _jax_layers() -> dict:
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    from omnibias.routing.jax import (
        assignment_relaxation,
        flow_relaxation,
        held_karp_layer,
    )

    return {
        "assignment": assignment_relaxation,
        "flow": flow_relaxation,
        "held_karp": held_karp_layer,
    }


def _torch_layers() -> dict:
    pytest.importorskip("torch")
    from omnibias.routing.torch import (
        assignment_relaxation,
        flow_relaxation,
        held_karp_layer,
    )

    return {
        "assignment": assignment_relaxation,
        "flow": flow_relaxation,
        "held_karp": held_karp_layer,
    }


@pytest.mark.parametrize("kind", KINDS)
def test_relaxation_is_near_feasible(kind: str) -> None:
    """Arc-use degree (row / column) sums are ~1: a valid fractional assignment."""
    layers = _jax_layers()
    x = np.asarray(layers[kind](_cost(7, 0)))
    assert x.shape == (7, 7)
    # soft-hinge exterior penalty -> a near-feasible heatmap (small box slack is expected)
    assert np.all(x >= -5e-2) and np.all(x <= 1.0 + 5e-2)
    assert np.allclose(x.sum(axis=1), 1.0, atol=5e-2)  # out-degree 1
    assert np.allclose(x.sum(axis=0), 1.0, atol=5e-2)  # in-degree 1


@pytest.mark.parametrize("kind", KINDS)
def test_torch_jax_parity(kind: str) -> None:
    """The two backends are bit-identical (float64) by construction."""
    jl, tl = _jax_layers(), _torch_layers()
    cost = _cost(7, 2)
    xj = np.asarray(jl[kind](cost))
    xt = tl[kind](cost).detach().numpy()
    assert np.max(np.abs(xj - xt)) < 1e-10


def test_batched_matches_single() -> None:
    """A batched solve equals stacking single-instance solves (vmap consistency)."""
    jl = _jax_layers()
    costs = np.stack([_cost(6, s) for s in range(4)])
    batched = np.asarray(jl["flow"](costs))
    singles = np.stack([np.asarray(jl["flow"](costs[b])) for b in range(4)])
    assert batched.shape == (4, 6, 6)
    assert np.max(np.abs(batched - singles)) < 1e-10


def test_relaxation_is_differentiable_jax() -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    layers = _jax_layers()
    cost = jnp.asarray(_cost(6, 1))

    def scalar(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(layers["assignment"](c) * c)

    g = jax.grad(scalar)(cost)
    assert g.shape == (6, 6)
    assert bool(np.all(np.isfinite(np.asarray(g))))
    assert bool(np.any(np.asarray(g) != 0.0))


def test_relaxation_is_differentiable_torch() -> None:
    import torch

    tl = _torch_layers()
    cost = torch.tensor(_cost(6, 1), dtype=torch.float64, requires_grad=True)
    loss = torch.sum(tl["assignment"](cost) * cost)
    loss.backward()
    assert cost.grad is not None
    assert bool(torch.all(torch.isfinite(cost.grad)))
    assert bool(torch.any(cost.grad != 0.0))


def test_heavier_schedule_tightens_feasibility() -> None:
    """More homotopy stages drive the Held-Karp heatmap closer to degree-feasible."""
    jl = _jax_layers()
    cost = _cost(7, 4)
    default = np.asarray(jl["held_karp"](cost))
    heavy = np.asarray(
        jl["held_karp"](cost, RelaxationSchedule(mu_growth=1.6, stages=16, steps=250))
    )
    err_default = np.max(np.abs(default.sum(axis=1) - 1.0))
    err_heavy = np.max(np.abs(heavy.sum(axis=1) - 1.0))
    assert err_heavy <= err_default + 1e-9

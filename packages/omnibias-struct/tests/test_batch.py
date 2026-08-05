# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Batching, transition-pair marginals, and the large-beta numerical envelope.

The ``*_batched`` layers must be bit-identical to looping the per-example layer (and to
each other across backends); the closed-form transition marginals must equal autodiff of
:func:`soft_viterbi` w.r.t. ``transitions``; and in float64 the soft value must stay finite
far past the annealing range the applications use.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _struct_helpers import dag_weight_matrix, random_chain, random_dag, sample_ctc

ATOL = 1e-12


# --- batched twins: vmap == loop, torch == jax ---------------------------


def test_soft_viterbi_batched_matches_loop_and_shapes_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_viterbi, soft_viterbi_batched

    trellises = [random_chain(s) for s in range(4)]
    emissions = torch.stack([torch.tensor(t.emissions) for t in trellises])
    transitions = torch.tensor(trellises[0].transitions)  # shared
    start = torch.tensor(trellises[0].start)
    batched = soft_viterbi_batched(emissions, transitions, 5.0, start=start)
    loop = torch.stack([soft_viterbi(emissions[i], transitions, 5.0, start=start) for i in range(4)])
    assert batched.shape == (4,)
    assert torch.max(torch.abs(batched - loop)).item() < ATOL


def test_soft_viterbi_batched_per_example_transitions_and_start_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_viterbi, soft_viterbi_batched

    trellises = [random_chain(s) for s in range(3)]
    emissions = torch.stack([torch.tensor(t.emissions) for t in trellises])
    transitions = torch.stack([torch.tensor(t.transitions) for t in trellises])
    start = torch.stack([torch.tensor(t.start) for t in trellises])
    batched = soft_viterbi_batched(emissions, transitions, 3.0, start=start)
    loop = torch.stack(
        [soft_viterbi(emissions[i], transitions[i], 3.0, start=start[i]) for i in range(3)]
    )
    assert torch.max(torch.abs(batched - loop)).item() < ATOL


def test_soft_shortest_path_and_ctc_batched_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import (
        soft_ctc,
        soft_ctc_batched,
        soft_shortest_path,
        soft_shortest_path_batched,
    )

    dag = random_dag(0)
    weights = torch.stack([torch.tensor(dag_weight_matrix(dag)) * (1.0 + 0.1 * i) for i in range(4)])
    b = soft_shortest_path_batched(weights, dag, 6.0)
    loop = torch.stack([soft_shortest_path(weights[i], dag, 6.0) for i in range(4)])
    assert torch.max(torch.abs(b - loop)).item() < ATOL

    lattice, lp = sample_ctc(0)
    log_probs = torch.stack([torch.tensor(lp) + 0.05 * i for i in range(4)])
    bc = soft_ctc_batched(log_probs, lattice, 4.0)
    loopc = torch.stack([soft_ctc(log_probs[i], lattice, 4.0) for i in range(4)])
    assert torch.max(torch.abs(bc - loopc)).item() < ATOL


def test_batched_torch_jax_parity() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from omnibias.struct.jax import soft_viterbi_batched as jax_b
    from omnibias.struct.torch import soft_viterbi_batched as torch_b

    trellises = [random_chain(s) for s in range(4)]
    em = np.stack([t.emissions for t in trellises])
    trn = trellises[0].transitions
    st = trellises[0].start
    bt = torch_b(torch.tensor(em), torch.tensor(trn), 5.0, start=torch.tensor(st)).numpy()
    bj = np.asarray(jax_b(jnp.asarray(em), jnp.asarray(trn), 5.0, start=jnp.asarray(st)))
    assert np.max(np.abs(bt - bj)) < ATOL


# --- transition-pair marginals ------------------------------------------


def test_transition_marginals_equal_autograd_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_viterbi, soft_viterbi_transition_marginals

    trellis = random_chain(1)
    emissions = torch.tensor(trellis.emissions)
    transitions = torch.tensor(trellis.transitions, requires_grad=True)
    start = torch.tensor(trellis.start)
    soft_viterbi(emissions, transitions, 6.0, start=start).backward()
    xi = soft_viterbi_transition_marginals(emissions, transitions.detach(), 6.0, start=start)
    assert np.allclose(xi.sum(dim=0).numpy(), transitions.grad.numpy(), atol=1e-9)
    assert np.allclose(xi.numpy().sum(axis=(1, 2)), 1.0, atol=1e-9)  # each (S,S) slab sums to 1


def test_transition_marginals_equal_grad_jax() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import soft_viterbi, soft_viterbi_transition_marginals

    trellis = random_chain(1)
    emissions = jnp.asarray(trellis.emissions)
    transitions = jnp.asarray(trellis.transitions)
    start = jnp.asarray(trellis.start)
    grad = jax.grad(lambda a: soft_viterbi(emissions, a, 6.0, start=start))(transitions)
    xi = soft_viterbi_transition_marginals(emissions, transitions, 6.0, start=start)
    assert np.allclose(np.asarray(xi).sum(axis=0), np.asarray(grad), atol=1e-9)


def test_transition_marginals_torch_jax_parity() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from omnibias.struct.jax import soft_viterbi_transition_marginals as jax_xi
    from omnibias.struct.torch import soft_viterbi_transition_marginals as torch_xi

    trellis = random_chain(2)
    xt = torch_xi(
        torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), 5.0,
        start=torch.tensor(trellis.start),
    ).numpy()
    xj = np.asarray(
        jax_xi(
            jnp.asarray(trellis.emissions), jnp.asarray(trellis.transitions), 5.0,
            start=jnp.asarray(trellis.start),
        )
    )
    assert np.max(np.abs(xt - xj)) < ATOL


# --- large-beta numerical envelope (float64) -----------------------------


@pytest.mark.parametrize("beta", [1.0e2, 1.0e6, 1.0e10, 1.0e12])
def test_soft_viterbi_value_finite_and_stable_float64(beta: float) -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct import viterbi
    from omnibias.struct.torch import soft_viterbi

    trellis = random_chain(0)
    hard, _ = viterbi(trellis)
    value = float(
        soft_viterbi(
            torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), beta,
            start=torch.tensor(trellis.start),
        )
    )
    assert math.isfinite(value)
    assert value >= hard - 1e-9  # lse_beta >= max (temperature axis) always


def test_viterbi_marginals_accurate_through_beta_1e6() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_viterbi_marginals

    trellis = random_chain(0)
    for beta in (1.0e2, 1.0e4, 1.0e6):
        gamma = soft_viterbi_marginals(
            torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), beta,
            start=torch.tensor(trellis.start),
        ).numpy()
        assert np.max(np.abs(gamma.sum(axis=1) - 1.0)) < 1e-8

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The exact gradient from the tower: closed-form marginals equal autodiff, sum to 1.

The soft-DP gradient is the forward-backward path marginal assembled from the tower's
softmax; here it is pinned equal to backend autodiff and shown to concentrate on the
hard optimum as ``beta -> inf``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _struct_helpers import dag_weight_matrix, random_chain, random_dag
from omnibias.struct import shortest_path, viterbi

ATOL = 1e-9


def test_viterbi_marginals_equal_autograd_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_viterbi, soft_viterbi_marginals

    trellis = random_chain(0)
    emissions = torch.tensor(trellis.emissions, requires_grad=True)
    transitions = torch.tensor(trellis.transitions)
    start = torch.tensor(trellis.start)
    soft_viterbi(emissions, transitions, 6.0, start=start).backward()
    gamma = soft_viterbi_marginals(emissions.detach(), transitions, 6.0, start=start)
    assert np.allclose(emissions.grad.numpy(), gamma.numpy(), atol=ATOL)
    assert np.allclose(gamma.numpy().sum(axis=1), 1.0, atol=ATOL)


def test_viterbi_marginals_equal_grad_jax() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import soft_viterbi, soft_viterbi_marginals

    trellis = random_chain(0)
    emissions = jnp.asarray(trellis.emissions)
    transitions = jnp.asarray(trellis.transitions)
    start = jnp.asarray(trellis.start)
    grad = jax.grad(lambda e: soft_viterbi(e, transitions, 6.0, start=start))(emissions)
    gamma = soft_viterbi_marginals(emissions, transitions, 6.0, start=start)
    assert np.allclose(np.asarray(grad), np.asarray(gamma), atol=ATOL)
    assert np.allclose(np.asarray(gamma).sum(axis=1), 1.0, atol=ATOL)


def test_viterbi_marginals_concentrate_on_the_hard_path() -> None:
    pytest.importorskip("torch")
    import torch
    from omnibias.struct.torch import soft_viterbi_marginals

    trellis = random_chain(5)
    _, path = viterbi(trellis)
    gamma = soft_viterbi_marginals(
        torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), 128.0,
        start=torch.tensor(trellis.start),
    ).numpy()
    assert tuple(int(i) for i in gamma.argmax(axis=1)) == path
    assert np.all(gamma.max(axis=1) > 1.0 - 1e-3)


def test_shortest_path_marginals_equal_autograd_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_shortest_path, soft_shortest_path_marginals

    dag = random_dag(1)
    weights = torch.tensor(dag_weight_matrix(dag), requires_grad=True)
    soft_shortest_path(weights, dag, 8.0).backward()
    xi = soft_shortest_path_marginals(weights.detach(), dag, 8.0).numpy()
    grad = np.nan_to_num(weights.grad.numpy())
    for u, v in dag.edges:
        assert abs(grad[u, v] - xi[u, v]) < ATOL


def test_shortest_path_marginals_equal_grad_jax() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import soft_shortest_path, soft_shortest_path_marginals

    dag = random_dag(1)
    weights = jnp.asarray(dag_weight_matrix(dag))
    grad = np.asarray(jax.grad(lambda w: soft_shortest_path(w, dag, 8.0))(weights))
    xi = np.asarray(soft_shortest_path_marginals(weights, dag, 8.0))
    for u, v in dag.edges:
        assert abs(grad[u, v] - xi[u, v]) < ATOL


def test_shortest_path_marginals_concentrate() -> None:
    pytest.importorskip("torch")
    import torch
    from omnibias.struct.torch import soft_shortest_path_marginals

    dag = random_dag(2)
    _, path = shortest_path(dag)
    xi = soft_shortest_path_marginals(torch.tensor(dag_weight_matrix(dag)), dag, 256.0).numpy()
    hard_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
    for u, v in dag.edges:
        expected = 1.0 if (u, v) in hard_edges else 0.0
        assert abs(xi[u, v] - expected) < 1e-2


def test_ctc_gradient_is_finite() -> None:
    torch = pytest.importorskip("torch")
    from _struct_helpers import sample_ctc
    from omnibias.struct.torch import soft_ctc

    lattice, log_probs = sample_ctc(0)
    lp = torch.tensor(log_probs, requires_grad=True)
    soft_ctc(lp, lattice, 4.0).backward()
    assert lp.grad is not None and torch.all(torch.isfinite(lp.grad))


def test_ctc_marginals_equal_autograd_torch() -> None:
    torch = pytest.importorskip("torch")
    from _struct_helpers import sample_ctc
    from omnibias.struct.torch import soft_ctc, soft_ctc_marginals

    lattice, log_probs = sample_ctc(0)
    lp = torch.tensor(log_probs, requires_grad=True)
    soft_ctc(lp, lattice, 3.0).backward()
    marg = soft_ctc_marginals(lp.detach(), lattice, 3.0)
    assert np.allclose(lp.grad.numpy(), marg.numpy(), atol=ATOL)
    assert np.allclose(marg.numpy().sum(axis=1), 1.0, atol=ATOL)


def test_ctc_marginals_equal_grad_jax() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from _struct_helpers import sample_ctc
    from omnibias.struct.jax import soft_ctc, soft_ctc_marginals

    lattice, log_probs = sample_ctc(0)
    lp = jnp.asarray(log_probs)
    grad = jax.grad(lambda x: soft_ctc(x, lattice, 3.0))(lp)
    marg = soft_ctc_marginals(lp, lattice, 3.0)
    assert np.allclose(np.asarray(grad), np.asarray(marg), atol=ATOL)
    assert np.allclose(np.asarray(marg).sum(axis=1), 1.0, atol=ATOL)


def test_ctc_marginals_concentrate_on_best_alignment() -> None:
    pytest.importorskip("torch")
    import torch
    from _struct_helpers import sample_ctc
    from omnibias.struct import ctc_best_alignment
    from omnibias.struct.torch import soft_ctc_marginals

    lattice, log_probs = sample_ctc(3)
    _, alignment = ctc_best_alignment(lattice, log_probs)
    marg = soft_ctc_marginals(torch.tensor(log_probs), lattice, 256.0).numpy()
    assert tuple(int(c) for c in marg.argmax(axis=1)) == alignment


def test_ctc_marginals_torch_jax_parity() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from _struct_helpers import sample_ctc
    from omnibias.struct.jax import soft_ctc_marginals as jax_marg
    from omnibias.struct.torch import soft_ctc_marginals as torch_marg

    lattice, log_probs = sample_ctc(2)
    mt = torch_marg(torch.tensor(log_probs), lattice, 5.0).numpy()
    mj = np.asarray(jax_marg(jnp.asarray(log_probs), lattice, 5.0))
    assert np.max(np.abs(mt - mj)) < 1e-12

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact higher-order DP-value jets: closed-form == autodiff Hessian == finite differences.

The ``delta -> 0`` tower differentiates the ``beta -> inf`` soft-DP value to all orders. We
pin ``chain_lse_jet`` / ``dag_lse_jet`` against the value, the directional gradient, and the
directional curvature ``d^T H d`` from backend autodiff, cross-check with finite
differences of the value, and require torch <-> jax parity to ``1e-9``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _struct_helpers import dag_weight_matrix, random_chain, random_dag


def _dag_direction(dag: object, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = np.zeros_like(dag_weight_matrix(dag))  # type: ignore[arg-type]
    for u, v in dag.edges:  # type: ignore[attr-defined]
        d[u, v] = rng.standard_normal()
    return d


# --- chain (Viterbi) jet -------------------------------------------------


def test_chain_jet_matches_autodiff_value_grad_hessian_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_viterbi
    from omnibias.struct.torch.jets import chain_lse_jet

    trellis = random_chain(0)
    e = torch.tensor(trellis.emissions)
    trn = torch.tensor(trellis.transitions)
    st = torch.tensor(trellis.start)
    d = torch.tensor(np.random.default_rng(0).standard_normal(e.shape))
    beta = 3.0
    jet = chain_lse_jet(e, trn, d, beta, order=2, start=st)

    e2 = e.clone().requires_grad_(True)
    value = soft_viterbi(e2, trn, beta, start=st)
    (grad,) = torch.autograd.grad(value, e2, create_graph=True)
    grad_d = (grad * d).sum()
    (hd,) = torch.autograd.grad(grad_d, e2, retain_graph=True)
    d_h_d = (hd * d).sum()

    assert abs(float(jet[0]) - float(value.detach())) < 1e-9
    assert abs(float(jet[1]) - float(grad_d.detach())) < 1e-9
    assert abs(2.0 * float(jet[2]) - float(d_h_d.detach())) < 1e-9


def test_chain_jet_matches_finite_differences_order3_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_viterbi
    from omnibias.struct.torch.jets import chain_lse_jet

    trellis = random_chain(2)
    e = torch.tensor(trellis.emissions)
    trn = torch.tensor(trellis.transitions)
    st = torch.tensor(trellis.start)
    d = torch.tensor(np.random.default_rng(2).standard_normal(e.shape))
    beta = 2.0
    jet = chain_lse_jet(e, trn, d, beta, order=3, start=st)

    def value_at(t: float) -> float:
        return float(soft_viterbi(e + t * d, trn, beta, start=st))

    h = 1e-3
    fd1 = (value_at(h) - value_at(-h)) / (2 * h)
    fd2 = (value_at(h) - 2 * value_at(0.0) + value_at(-h)) / h**2
    assert abs(float(jet[1]) - fd1) < 1e-6
    assert abs(2.0 * float(jet[2]) - fd2) < 1e-4


# --- DAG shortest-path jet ----------------------------------------------


def test_dag_jet_matches_autodiff_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import soft_shortest_path
    from omnibias.struct.torch.jets import dag_lse_jet

    dag = random_dag(1)
    w = torch.tensor(dag_weight_matrix(dag))
    dw = torch.tensor(_dag_direction(dag, 1))
    beta = 3.0
    jet = dag_lse_jet(w, dw, dag, beta, order=2)

    w2 = w.clone().requires_grad_(True)
    cost = soft_shortest_path(w2, dag, beta)
    (grad,) = torch.autograd.grad(cost, w2, create_graph=True)
    grad_d = (grad * dw).sum()
    (hd,) = torch.autograd.grad(grad_d, w2, retain_graph=True)
    d_h_d = (hd * dw).sum()

    assert abs(float(jet[0]) - float(cost.detach())) < 1e-9
    assert abs(float(jet[1]) - float(grad_d.detach())) < 1e-9
    assert abs(2.0 * float(jet[2]) - float(d_h_d.detach())) < 1e-9


# --- torch <-> jax parity ------------------------------------------------


def test_chain_and_dag_jets_torch_jax_parity() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from omnibias.struct.jax.jets import chain_lse_jet as jax_chain
    from omnibias.struct.jax.jets import dag_lse_jet as jax_dag
    from omnibias.struct.torch.jets import chain_lse_jet as torch_chain
    from omnibias.struct.torch.jets import dag_lse_jet as torch_dag

    trellis = random_chain(3)
    d = np.random.default_rng(3).standard_normal(trellis.emissions.shape)
    jt = torch_chain(
        torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), torch.tensor(d),
        4.0, order=3, start=torch.tensor(trellis.start),
    ).numpy()
    jj = np.asarray(
        jax_chain(
            jnp.asarray(trellis.emissions), jnp.asarray(trellis.transitions), jnp.asarray(d),
            4.0, order=3, start=jnp.asarray(trellis.start),
        )
    )
    assert np.max(np.abs(jt - jj)) < 1e-9

    dag = random_dag(2)
    w = dag_weight_matrix(dag)
    dw = _dag_direction(dag, 2)
    dt = torch_dag(torch.tensor(w), torch.tensor(dw), dag, 3.0, order=2).numpy()
    dj = np.asarray(jax_dag(jnp.asarray(w), jnp.asarray(dw), dag, 3.0, order=2))
    assert np.max(np.abs(dt - dj)) < 1e-9


def test_lse2_jet_generalizes_pairwise_lse_jet_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import pairwise_lse_jet
    from omnibias.struct.torch.jets import lse2_jet

    a0 = torch.tensor(0.3)
    b0 = torch.tensor(-0.4)
    db = torch.tensor(1.1)
    beta = 2.0
    order = 3
    # pairwise_lse_jet fixes a constant and moves b linearly: build the same two jets.
    a_jet = torch.stack([a0] + [torch.zeros(()) for _ in range(order)])
    b_jet = torch.stack([b0, db] + [torch.zeros(()) for _ in range(order - 1)])
    got = lse2_jet(a_jet, b_jet, beta, order)
    ref = pairwise_lse_jet(a0, b0, db, beta, order)
    assert np.max(np.abs(got.numpy() - ref.numpy())) < 1e-12

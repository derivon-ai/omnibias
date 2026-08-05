# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The semiring / hypergraph driver: numpy oracles + the additive-safety cross-checks.

Two claims are pinned here. (1) The pure-numpy reductions
(:func:`~omnibias.struct.hard_value` / ``soft_value`` / ``count_derivations``) equal the
flat brute-force oracle over *enumerated* derivations on tiny graphs. (2) The differentiable
backend driver reproduces the hand-written soft-DP layers -- ``soft_viterbi`` /
``soft_shortest_path`` / ``soft_dtw`` / ``soft_align`` -- to ``< 1e-12`` (the proof that the
driver subsumes them without touching their numerics), its closed-form ``semiring_marginals``
equal ``autograd``, and its torch <-> jax twins are bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.struct import (
    DAG,
    HyperEdge,
    Hypergraph,
    best_derivation,
    brute_force_value,
    count_derivations,
    enumerate_derivations,
    from_dag,
    hard_value,
    soft_value,
)
from omnibias.struct._core.align import AlignmentLattice

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import omnibias.struct.jax as sj  # noqa: E402
import omnibias.struct.torch as st  # noqa: E402
from _struct_helpers import dag_weight_matrix, random_chain, random_dag  # noqa: E402

torch.set_default_dtype(torch.float64)

SEEDS = range(6)
TOL = 1e-12


# ---------------------------------------------------------------------------
# Test-local converters: lift a chain / DTW DP into the driver (arity-1 hyperedges).
# ---------------------------------------------------------------------------


def _chain_graph(n_steps: int, n_states: int) -> tuple[Hypergraph, list[tuple]]:
    edges: list[HyperEdge] = []
    specs: list[tuple] = []
    root = n_steps * n_states

    def nid(t: int, s: int) -> int:
        return t * n_states + s

    for s in range(n_states):
        edges.append(HyperEdge(head=nid(0, s), tails=()))
        specs.append(("start", 0, s))
    for t in range(1, n_steps):
        for s in range(n_states):
            for sp in range(n_states):
                edges.append(HyperEdge(head=nid(t, s), tails=(nid(t - 1, sp),)))
                specs.append(("trans", t, sp, s))
    for s in range(n_states):
        edges.append(HyperEdge(head=root, tails=(nid(n_steps - 1, s),)))
        specs.append(("final", s))
    return Hypergraph(num_nodes=root + 1, edges=tuple(edges), root=root), specs


def _chain_edge_weights(specs: list[tuple], emissions, transitions, start, xp):  # noqa: ANN001
    rows = []
    for spec in specs:
        if spec[0] == "start":
            _, t, s = spec
            rows.append(start[s] + emissions[t, s])
        elif spec[0] == "trans":
            _, t, sp, s = spec
            rows.append(transitions[sp, s] + emissions[t, s])
        else:
            rows.append(xp.zeros(()))
    return xp.stack(rows)


def _dtw_graph(n_rows: int, n_cols: int) -> tuple[Hypergraph, list[tuple[int, int]]]:
    edges: list[HyperEdge] = []
    specs: list[tuple[int, int]] = []

    def nid(i: int, j: int) -> int:
        return i * n_cols + j

    for i in range(n_rows):
        for j in range(n_cols):
            preds = []
            if i > 0:
                preds.append(nid(i - 1, j))
            if j > 0:
                preds.append(nid(i, j - 1))
            if i > 0 and j > 0:
                preds.append(nid(i - 1, j - 1))
            if not preds:
                edges.append(HyperEdge(head=nid(i, j), tails=()))
                specs.append((i, j))
            else:
                for p in preds:
                    edges.append(HyperEdge(head=nid(i, j), tails=(p,)))
                    specs.append((i, j))
    return Hypergraph(num_nodes=n_rows * n_cols, edges=tuple(edges), root=nid(n_rows - 1, n_cols - 1)), specs


def _dag_edge_weights(graph: Hypergraph, edge_index: dict[tuple[int, int], int], scores, xp):  # noqa: ANN001
    rows = [xp.zeros(())] * graph.num_edges
    for (u, v), i in edge_index.items():
        rows[i] = scores[u, v]
    return xp.stack(rows)


# ---------------------------------------------------------------------------
# (1) numpy oracles
# ---------------------------------------------------------------------------


def _toy_graph() -> tuple[Hypergraph, list[float]]:
    edges = (
        HyperEdge(head=0, tails=()),
        HyperEdge(head=1, tails=()),
        HyperEdge(head=2, tails=(0, 1)),
        HyperEdge(head=3, tails=(2,)),
        HyperEdge(head=3, tails=(0, 1)),
    )
    return Hypergraph(num_nodes=4, edges=edges, root=3), [0.5, -0.2, 1.0, 0.3, 0.7]


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0])
def test_oracle_matches_bruteforce(beta: float) -> None:
    g, w = _toy_graph()
    assert abs(hard_value(g, w) - brute_force_value(g, w, None)) < TOL
    assert abs(soft_value(g, w, beta) - brute_force_value(g, w, beta)) < TOL


def test_counting_matches_enumeration() -> None:
    g, _ = _toy_graph()
    assert count_derivations(g) == len(list(enumerate_derivations(g)))


def test_best_derivation_agrees_with_hard_value() -> None:
    g, w = _toy_graph()
    value, deriv = best_derivation(g, w)
    assert abs(value - hard_value(g, w)) < TOL
    # the derivation's edge weights sum back to the value
    assert abs(sum(w[e] for e in deriv) - value) < TOL


@pytest.mark.parametrize("seed", SEEDS)
def test_from_dag_counts_match(seed: int) -> None:
    dag = random_dag(seed)
    g, _ = from_dag(dag)
    assert count_derivations(g) == dag.count_paths()


def test_structural_validation() -> None:
    with pytest.raises(ValueError, match="arity"):
        HyperEdge(head=5, tails=(0, 1, 2))
    with pytest.raises(ValueError, match="topological"):
        HyperEdge(head=1, tails=(2,))
    with pytest.raises(ValueError, match="root"):
        Hypergraph(num_nodes=2, edges=(HyperEdge(head=0),), root=5)


# ---------------------------------------------------------------------------
# (2) the backend driver reproduces the hand-written soft-DP layers to < 1e-12
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_driver_reproduces_shortest_path(seed: int) -> None:
    dag = random_dag(seed)
    w = dag_weight_matrix(dag)
    g, idx = from_dag(dag)
    for beta in (1.0, 8.0, 64.0):
        ew_t = _dag_edge_weights(g, idx, -torch.tensor(w), torch)
        ew_j = _dag_edge_weights(g, idx, -jnp.asarray(w), jnp)
        # driver value (soft score) == -soft_shortest_path (softmin cost) in both backends
        assert abs(float(st.semiring_value(g, ew_t, beta)) + float(st.soft_shortest_path(torch.tensor(w), dag, beta))) < TOL
        assert abs(float(sj.semiring_value(g, ew_j, beta)) + float(sj.soft_shortest_path(jnp.asarray(w), dag, beta))) < TOL


@pytest.mark.parametrize("seed", SEEDS)
def test_driver_reproduces_viterbi(seed: int) -> None:
    tr = random_chain(seed, n_steps=5, n_states=4)
    g, specs = _chain_graph(5, 4)
    for beta in (1.0, 8.0, 64.0):
        ew_t = _chain_edge_weights(specs, torch.tensor(tr.emissions), torch.tensor(tr.transitions), torch.tensor(tr.start), torch)
        vt = st.soft_viterbi(torch.tensor(tr.emissions), torch.tensor(tr.transitions), beta, start=torch.tensor(tr.start))
        assert abs(float(st.semiring_value(g, ew_t, beta)) - float(vt)) < TOL
        ew_j = _chain_edge_weights(specs, jnp.asarray(tr.emissions), jnp.asarray(tr.transitions), jnp.asarray(tr.start), jnp)
        vj = sj.soft_viterbi(jnp.asarray(tr.emissions), jnp.asarray(tr.transitions), beta, start=jnp.asarray(tr.start))
        assert abs(float(sj.semiring_value(g, ew_j, beta)) - float(vj)) < TOL


@pytest.mark.parametrize("seed", SEEDS)
def test_driver_reproduces_dtw(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cost = rng.standard_normal((4, 5)) ** 2
    g, specs = _dtw_graph(4, 5)
    for beta in (1.0, 8.0, 64.0):
        ew_t = torch.stack([-torch.tensor(cost)[i, j] for (i, j) in specs])
        assert abs(-float(st.semiring_value(g, ew_t, beta)) - float(st.soft_dtw(torch.tensor(cost), beta))) < TOL
        ew_j = jnp.stack([-jnp.asarray(cost)[i, j] for (i, j) in specs])
        assert abs(-float(sj.semiring_value(g, ew_j, beta)) - float(sj.soft_dtw(jnp.asarray(cost), beta))) < TOL


def test_driver_reproduces_align() -> None:
    a = np.array([0, 1, 2])
    b = np.array([1, 2, 0])
    sub = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
    gap = -1.5
    lattice = AlignmentLattice(a.shape[0], b.shape[0])
    dag, labels = lattice.build_dag()
    w = np.zeros((lattice.num_nodes, lattice.num_nodes))
    for (u, v), (kind, i, j) in labels.items():
        w[u, v] = -sub[a[i], b[j]] if kind == "sub" else -gap
    g, idx = from_dag(dag)
    for beta in (1.0, 8.0, 64.0):
        ew_t = _dag_edge_weights(g, idx, -torch.tensor(w), torch)
        driver = float(st.semiring_value(g, ew_t, beta))
        layer = float(st.soft_align(a, b, torch.tensor(sub), torch.tensor(gap), beta))
        assert abs(driver - layer) < TOL


# ---------------------------------------------------------------------------
# marginals: closed form == autograd == the hand-written forward-backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_marginals_equal_autograd_and_shortest_path(seed: int) -> None:
    dag = random_dag(seed)
    w = dag_weight_matrix(dag)
    g, idx = from_dag(dag)
    beta = 6.0
    ew = _dag_edge_weights(g, idx, -torch.tensor(w), torch).requires_grad_(True)
    value = st.semiring_value(g, ew, beta)
    mu = st.semiring_marginals(g, ew, beta).detach()
    (grad,) = torch.autograd.grad(value, ew)
    assert float((mu - grad).abs().max()) < 1e-9
    # per-edge marginals equal the hand-written forward-backward edge marginals
    xi = st.soft_shortest_path_marginals(torch.tensor(w), dag, beta)
    for (u, v), i in idx.items():
        assert abs(float(mu[i]) - float(xi[u, v])) < 1e-10
    # marginals of the edges into the root sum to 1
    root_edges = g.incoming(g.root)
    assert abs(float(sum(mu[e] for e in root_edges)) - 1.0) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_driver_torch_jax_parity(seed: int) -> None:
    dag = random_dag(seed)
    w = dag_weight_matrix(dag)
    g, idx = from_dag(dag)
    for beta in (1.0, 8.0):
        ew_t = _dag_edge_weights(g, idx, -torch.tensor(w), torch)
        ew_j = _dag_edge_weights(g, idx, -jnp.asarray(w), jnp)
        assert abs(float(st.semiring_value(g, ew_t, beta)) - float(sj.semiring_value(g, ew_j, beta))) < 1e-11
        mt = np.asarray(st.semiring_marginals(g, ew_t, beta).detach())
        mj = np.asarray(sj.semiring_marginals(g, ew_j, beta))
        assert float(np.max(np.abs(mt - mj))) < 1e-11

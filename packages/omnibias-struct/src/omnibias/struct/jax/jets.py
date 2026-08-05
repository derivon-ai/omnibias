# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact higher-order jets of a soft-DP value through the unrolled recursion (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.jets` (float64 -- enable
``jax_enable_x64``). This is the ``delta -> 0`` founding bias collapse tower differentiating
the ``beta -> inf`` relaxation to **all** orders: a univariate Taylor jet is propagated
along a perturbation direction through every ``lse_beta`` reduction of the DP, each
pairwise combine ``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`` carrying its whole
jet from the beta-tempered ``softplus`` tower (``softplus^(n) = sigma^(n-1)``) via
:func:`omnibias.jax.jet.compose_jet`, and multi-way reductions folding the exact
associative pairwise jet.

The returned jet is ``(order + 1,)`` Taylor coefficients ``c_k = f^(k)(0) / k!`` of
``t -> V(params + t * direction)``: ``c_0`` value, ``c_1`` directional gradient,
``2 c_2 = direction^T H direction`` directional curvature.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from omnibias.jax.activations import get_activation
from omnibias.jax.jet import compose_jet
from omnibias.struct._core.eisner import EisnerSpec, eisner_hypergraph
from omnibias.struct._core.parse import BinaryGrammar, build_chart
from omnibias.struct._core.semiring import Hypergraph
from omnibias.struct._core.trellis import DAG, ChainTrellis


def _softplus_tower(u0: Array, order: int) -> Array:
    r"""Derivative tower ``[softplus(u0), softplus'(u0), ..., softplus^(order)(u0)]``."""
    spec = get_activation("softplus")
    fastpath = spec.fastpath
    if fastpath is None:  # defensive: softplus always ships its closed-form tower
        raise NotImplementedError("softplus is missing its closed-form fastpath tower")
    rows = [spec.forward(u0)]
    for k in range(1, order + 1):
        rows.append(fastpath(u0, k))
    return jnp.stack(rows, axis=0)


def lse2_jet(a: Array, b: Array, beta: float = 1.0, order: int = 2) -> Array:
    r"""Jet of ``t -> lse_beta(a(t), b(t))`` from two input jets ``a``, ``b``.

    ``a`` and ``b`` are ``(order + 1, ...)`` Taylor-coefficient jets sharing a trailing
    shape. Uses ``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`` and composes the
    ``softplus`` tower through :func:`omnibias.jax.jet.compose_jet`. Generalises
    :func:`omnibias.struct.jax.pairwise_lse_jet` (which fixes ``a`` constant).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    u = beta * (b - a)
    tower = _softplus_tower(u[0], order)
    sp = compose_jet(u, tower) / beta
    result: Array = a + sp
    return result


def _lse_reduce_jet(jets: list[Array], beta: float, order: int) -> Array:
    r"""Fold the exact pairwise :func:`lse2_jet` over a list (``lse`` is associative)."""
    acc = jets[0]
    for nxt in jets[1:]:
        acc = lse2_jet(acc, nxt, beta, order)
    return acc


def _const_jet(value: Array, order: int) -> Array:
    rows = [value] + [jnp.zeros_like(value) for _ in range(order)]
    return jnp.stack(rows, axis=0)


def _linear_jet(value: Array, slope: Array, order: int) -> Array:
    rows = [value]
    if order >= 1:
        rows.append(slope)
    rows.extend(jnp.zeros_like(value) for _ in range(order - 1))
    return jnp.stack(rows[: order + 1], axis=0)


def chain_lse_jet(
    emissions: Array,
    transitions: Array,
    direction: Array,
    beta: float = 1.0,
    *,
    order: int = 2,
    start: Array | None = None,
) -> Array:
    r"""Exact jet of ``t -> soft_viterbi(emissions + t * direction, transitions, beta)``.

    ``emissions`` and ``direction`` are ``(T, S)``; ``transitions`` is ``(S, S)`` (held
    constant); ``start`` is ``(S,)`` or ``None``. Returns the ``(order + 1,)`` jet of the
    soft Viterbi value along ``direction`` in emission space.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    n_steps, n_states = int(emissions.shape[0]), int(emissions.shape[1])
    start_t = jnp.zeros(n_states, dtype=emissions.dtype) if start is None else start
    alpha_cols = [
        _linear_jet(start_t[s] + emissions[0, s], direction[0, s], order) for s in range(n_states)
    ]
    alpha = jnp.stack(alpha_cols, axis=1)  # (order + 1, S)
    for t in range(1, n_steps):
        new_cols = []
        for s in range(n_states):
            reduced = _lse_reduce_jet(
                [alpha[:, sp] + _const_jet(transitions[sp, s], order) for sp in range(n_states)],
                beta,
                order,
            )
            new_cols.append(reduced + _linear_jet(emissions[t, s], direction[t, s], order))
        alpha = jnp.stack(new_cols, axis=1)
    return _lse_reduce_jet([alpha[:, s] for s in range(n_states)], beta, order)


def dag_lse_jet(
    weights: Array,
    direction: Array,
    dag: DAG,
    beta: float = 1.0,
    *,
    order: int = 2,
) -> Array:
    r"""Exact jet of ``t -> soft_shortest_path(weights + t * direction, dag, beta)``.

    ``weights`` and ``direction`` are ``(n, n)`` edge tensors (only the ``dag`` edge
    entries matter). Returns the ``(order + 1,)`` jet of the softmin cost along
    ``direction`` (``value = -alpha[sink]`` in the max convention).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    alpha: list[Array | None] = [None] * dag.num_nodes
    alpha[dag.source] = _const_jet(jnp.zeros((), dtype=weights.dtype), order)
    for v in range(dag.num_nodes):
        if v == dag.source:
            continue
        preds = [u for u in dag.incoming(v) if alpha[u] is not None]
        if not preds:
            continue
        contribs = [
            alpha[u] - _linear_jet(weights[u, v], direction[u, v], order)  # type: ignore[operator]
            for u in preds
        ]
        alpha[v] = _lse_reduce_jet(contribs, beta, order)
    sink = alpha[dag.sink]
    if sink is None:
        raise ValueError("sink is not reachable from source in this DAG")
    return -sink


def _sum_jets(jets: Sequence[Array]) -> Array:
    r"""Add a list of ``(order + 1, ...)`` jets coefficient-wise (the semiring edge product)."""
    acc = jets[0]
    for nxt in jets[1:]:
        acc = acc + nxt
    return acc


def hypergraph_lse_jet(
    graph: Hypergraph,
    weight_jets: Sequence[Array],
    beta: float = 1.0,
    *,
    order: int = 2,
) -> Array:
    r"""Exact jet of the ``lse_beta`` inside value of a :class:`Hypergraph` at its ``root``.

    ``weight_jets`` is a length-``graph.num_edges`` sequence of ``(order + 1,)`` scalar jets,
    one per hyperedge (typically a :func:`_linear_jet` of a perturbed edge weight). The inside
    recursion ``inside[v] = lse_beta_e (weight[e] + sum_tails inside)`` is a
    ``+``/``lse_beta`` fold, so every step is either a jet sum (:func:`_sum_jets`) or the exact
    associative pairwise :func:`lse2_jet` fold (:func:`_lse_reduce_jet`) -- the whole jet is
    closed form, no autodiff. This is the driver the CKY / Eisner jets lift to (mirroring
    :func:`omnibias.struct.jax.semiring_value`).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if len(weight_jets) != graph.num_edges:
        raise ValueError(
            f"weight_jets must have length num_edges={graph.num_edges}, got {len(weight_jets)}"
        )
    node_jets: list[Array | None] = [None] * graph.num_nodes
    for v in range(graph.num_nodes):
        edge_jets: list[Array] = []
        for ei in graph.incoming(v):
            tail_jets = [node_jets[t] for t in graph.edges[ei].tails]
            if any(tj is None for tj in tail_jets):
                continue  # a derivation through an unreachable tail is the -inf semiring zero
            edge_jets.append(_sum_jets([weight_jets[ei], *tail_jets]))  # type: ignore[list-item]
        if edge_jets:
            node_jets[v] = _lse_reduce_jet(edge_jets, beta, order)
    root = node_jets[graph.root]
    if root is None:
        raise ValueError("root is unreachable from the axioms")
    return root


def cky_lse_jet(
    grammar: BinaryGrammar,
    emit: Array,
    rule: Array,
    emit_dir: Array,
    rule_dir: Array,
    beta: float = 1.0,
    *,
    order: int = 2,
) -> Array:
    r"""Exact jet of ``t -> soft_inside(emit + t emit_dir, rule + t rule_dir, beta)``.

    ``emit`` / ``emit_dir`` are ``(L, R)`` and ``rule`` / ``rule_dir`` are ``(num_rules,)``;
    the returned ``(order + 1,)`` jet's ``c_1`` is the directional gradient of the CKY inside
    partition and ``2 c_2`` its directional curvature -- pinned to backend autodiff of
    :func:`omnibias.struct.jax.soft_inside`.
    """
    spec = build_chart(grammar, int(emit.shape[0]))
    weight_jets: list[Array] = []
    for s in spec.edge_specs:
        if s[0] == "emit":
            i, a = int(s[1]), int(s[2])
            weight_jets.append(_linear_jet(emit[i, a], emit_dir[i, a], order))
        else:
            r = int(s[1])
            weight_jets.append(_linear_jet(rule[r], rule_dir[r], order))
    return hypergraph_lse_jet(spec.graph, weight_jets, beta, order=order)


def eisner_lse_jet(
    arc: Array,
    direction: Array,
    beta: float = 1.0,
    *,
    order: int = 2,
) -> Array:
    r"""Exact jet of ``t -> soft_eisner(arc + t * direction, beta)``.

    ``arc`` / ``direction`` are ``(n + 1, n + 1)`` arc-score matrices; the returned
    ``(order + 1,)`` jet's ``c_1`` is the directional gradient of the projective-parse
    partition and ``2 c_2`` its directional curvature -- pinned to backend autodiff of
    :func:`omnibias.struct.jax.soft_eisner`.
    """
    spec: EisnerSpec = eisner_hypergraph(int(arc.shape[0]) - 1)
    zero = jnp.zeros((), dtype=arc.dtype)
    weight_jets: list[Array] = []
    for s in spec.edge_specs:
        if s[0] == "arc":
            h, m = int(s[1]), int(s[2])
            weight_jets.append(_linear_jet(arc[h, m], direction[h, m], order))
        else:
            weight_jets.append(_const_jet(zero, order))
    return hypergraph_lse_jet(spec.graph, weight_jets, beta, order=order)


def dp_value_jet(
    problem: ChainTrellis | DAG,
    direction: Array,
    beta: float = 1.0,
    *,
    order: int = 2,
    weights: Array | None = None,
) -> Array:
    r"""Dispatch the exact DP-value jet by problem type.

    * :class:`ChainTrellis` -- pulls ``emissions`` / ``transitions`` / ``start`` from the
      problem; ``direction`` perturbs the emissions ``(T, S)``.
    * :class:`DAG` -- requires the finite edge ``weights`` ``(n, n)``; ``direction``
      perturbs them.
    """
    if isinstance(problem, ChainTrellis):
        return chain_lse_jet(
            jnp.asarray(problem.emissions),
            jnp.asarray(problem.transitions),
            jnp.asarray(direction),
            beta,
            order=order,
            start=jnp.asarray(problem.start),
        )
    if weights is None:
        raise ValueError("dp_value_jet on a DAG requires the edge `weights` (n, n)")
    return dag_lse_jet(jnp.asarray(weights), jnp.asarray(direction), problem, beta, order=order)


__all__ = [
    "chain_lse_jet",
    "cky_lse_jet",
    "dag_lse_jet",
    "dp_value_jet",
    "eisner_lse_jet",
    "hypergraph_lse_jet",
    "lse2_jet",
]

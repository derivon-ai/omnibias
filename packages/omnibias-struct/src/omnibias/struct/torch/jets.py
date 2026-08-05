# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact higher-order jets of a soft-DP value through the unrolled recursion (torch).

This is the ``delta -> 0`` founding bias collapse tower differentiating the ``beta -> inf``
relaxation to **all** orders (not just the first-order marginal). A univariate Taylor jet
is propagated along a perturbation direction through every ``lse_beta`` reduction of the
DP: each pairwise combine ``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`` has its
whole jet from the beta-tempered ``softplus`` tower (``softplus^(n) = sigma^(n-1)`` in
:mod:`omnibias.core`) via :func:`omnibias.torch.jet.compose_jet`, and multi-way reductions
fold the exact associative pairwise jet. No autodiff and no finite differences -- the jet
coefficients are closed form.

The returned jet is ``(order + 1,)`` Taylor coefficients ``c_k = f^(k)(0) / k!`` of
``t -> V(params + t * direction)``: ``c_0`` is the value, ``c_1`` is the directional
gradient ``grad . direction``, and ``2 c_2 = direction^T H direction`` is the directional
curvature. Bit-identical twin of :mod:`omnibias.struct.jax.jets`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.struct._core.eisner import EisnerSpec, eisner_hypergraph
from omnibias.struct._core.parse import BinaryGrammar, build_chart
from omnibias.struct._core.semiring import Hypergraph
from omnibias.struct._core.trellis import DAG, ChainTrellis
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.jet import compose_jet
from torch import Tensor


def _softplus_tower(u0: Tensor, order: int) -> Tensor:
    r"""Derivative tower ``[softplus(u0), softplus'(u0), ..., softplus^(order)(u0)]``."""
    spec = get_activation("softplus")
    fastpath = spec.fastpath
    if fastpath is None:  # defensive: softplus always ships its closed-form tower
        raise NotImplementedError("softplus is missing its closed-form fastpath tower")
    rows = [spec.forward(u0)]
    for k in range(1, order + 1):
        rows.append(fastpath(u0, k))
    return torch.stack(rows, dim=0)


def lse2_jet(a: Tensor, b: Tensor, beta: float = 1.0, order: int = 2) -> Tensor:
    r"""Jet of ``t -> lse_beta(a(t), b(t))`` from two input jets ``a``, ``b``.

    ``a`` and ``b`` are ``(order + 1, ...)`` Taylor-coefficient jets sharing a trailing
    shape. Uses ``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`` and composes the
    ``softplus`` tower through :func:`omnibias.torch.jet.compose_jet`. Generalises
    :func:`omnibias.struct.torch.pairwise_lse_jet` (which fixes ``a`` constant).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    u = beta * (b - a)
    tower = _softplus_tower(u[0], order)
    sp = compose_jet(u, tower) / beta
    result: Tensor = a + sp
    return result


def _lse_reduce_jet(jets: list[Tensor], beta: float, order: int) -> Tensor:
    r"""Fold the exact pairwise :func:`lse2_jet` over a list (``lse`` is associative)."""
    acc = jets[0]
    for nxt in jets[1:]:
        acc = lse2_jet(acc, nxt, beta, order)
    return acc


def _const_jet(value: Tensor, order: int) -> Tensor:
    rows = [value] + [torch.zeros_like(value) for _ in range(order)]
    return torch.stack(rows, dim=0)


def _linear_jet(value: Tensor, slope: Tensor, order: int) -> Tensor:
    rows = [value]
    if order >= 1:
        rows.append(slope)
    rows.extend(torch.zeros_like(value) for _ in range(order - 1))
    return torch.stack(rows[: order + 1], dim=0)


def chain_lse_jet(
    emissions: Tensor,
    transitions: Tensor,
    direction: Tensor,
    beta: float = 1.0,
    *,
    order: int = 2,
    start: Tensor | None = None,
) -> Tensor:
    r"""Exact jet of ``t -> soft_viterbi(emissions + t * direction, transitions, beta)``.

    ``emissions`` and ``direction`` are ``(T, S)``; ``transitions`` is ``(S, S)`` (held
    constant); ``start`` is ``(S,)`` or ``None``. Returns the ``(order + 1,)`` jet of the
    soft Viterbi value along ``direction`` in emission space. ``2 * jet[2]`` is the
    directional curvature ``direction^T H direction`` of the value.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    n_steps, n_states = int(emissions.shape[0]), int(emissions.shape[1])
    start_t = torch.zeros(n_states, dtype=emissions.dtype) if start is None else start
    # alpha jet: (order + 1, S)
    alpha_cols = [
        _linear_jet(start_t[s] + emissions[0, s], direction[0, s], order) for s in range(n_states)
    ]
    alpha = torch.stack(alpha_cols, dim=1)  # (order + 1, S)
    for t in range(1, n_steps):
        new_cols = []
        for s in range(n_states):
            reduced = _lse_reduce_jet(
                [alpha[:, sp] + _const_jet(transitions[sp, s], order) for sp in range(n_states)],
                beta,
                order,
            )
            new_cols.append(reduced + _linear_jet(emissions[t, s], direction[t, s], order))
        alpha = torch.stack(new_cols, dim=1)
    return _lse_reduce_jet([alpha[:, s] for s in range(n_states)], beta, order)


def dag_lse_jet(
    weights: Tensor,
    direction: Tensor,
    dag: DAG,
    beta: float = 1.0,
    *,
    order: int = 2,
) -> Tensor:
    r"""Exact jet of ``t -> soft_shortest_path(weights + t * direction, dag, beta)``.

    ``weights`` and ``direction`` are ``(n, n)`` edge tensors (only the ``dag`` edge
    entries matter). Returns the ``(order + 1,)`` jet of the softmin cost along
    ``direction``. The score of edge ``(u, v)`` is ``-weight``, so the value is
    ``-alpha[sink]`` in the max convention.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    alpha: list[Tensor | None] = [None] * dag.num_nodes
    alpha[dag.source] = _const_jet(torch.zeros((), dtype=weights.dtype), order)
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


def _sum_jets(jets: Sequence[Tensor]) -> Tensor:
    r"""Add a list of ``(order + 1, ...)`` jets coefficient-wise (the semiring edge product)."""
    acc = jets[0]
    for nxt in jets[1:]:
        acc = acc + nxt
    return acc


def hypergraph_lse_jet(
    graph: Hypergraph,
    weight_jets: Sequence[Tensor],
    beta: float = 1.0,
    *,
    order: int = 2,
) -> Tensor:
    r"""Exact jet of the ``lse_beta`` inside value of a :class:`Hypergraph` at its ``root``.

    ``weight_jets`` is a length-``graph.num_edges`` sequence of ``(order + 1,)`` scalar jets,
    one per hyperedge (typically a :func:`_linear_jet` of a perturbed edge weight). The inside
    recursion ``inside[v] = lse_beta_e (weight[e] + sum_tails inside)`` is a
    ``+``/``lse_beta`` fold, so every step is either a jet sum (:func:`_sum_jets`) or the exact
    associative pairwise :func:`lse2_jet` fold (:func:`_lse_reduce_jet`) -- the whole jet is
    closed form, no autodiff. This is the driver the CKY / Eisner jets lift to (mirroring
    :func:`omnibias.struct.torch.semiring_value`).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if len(weight_jets) != graph.num_edges:
        raise ValueError(
            f"weight_jets must have length num_edges={graph.num_edges}, got {len(weight_jets)}"
        )
    node_jets: list[Tensor | None] = [None] * graph.num_nodes
    for v in range(graph.num_nodes):
        edge_jets: list[Tensor] = []
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
    emit: Tensor,
    rule: Tensor,
    emit_dir: Tensor,
    rule_dir: Tensor,
    beta: float = 1.0,
    *,
    order: int = 2,
) -> Tensor:
    r"""Exact jet of ``t -> soft_inside(emit + t emit_dir, rule + t rule_dir, beta)``.

    ``emit`` / ``emit_dir`` are ``(L, R)`` and ``rule`` / ``rule_dir`` are ``(num_rules,)``;
    the returned ``(order + 1,)`` jet's ``c_1`` is the directional gradient of the CKY inside
    partition and ``2 c_2`` its directional curvature -- pinned to backend autodiff of
    :func:`omnibias.struct.torch.soft_inside`.
    """
    spec = build_chart(grammar, int(emit.shape[0]))
    weight_jets: list[Tensor] = []
    for s in spec.edge_specs:
        if s[0] == "emit":
            i, a = int(s[1]), int(s[2])
            weight_jets.append(_linear_jet(emit[i, a], emit_dir[i, a], order))
        else:
            r = int(s[1])
            weight_jets.append(_linear_jet(rule[r], rule_dir[r], order))
    return hypergraph_lse_jet(spec.graph, weight_jets, beta, order=order)


def eisner_lse_jet(
    arc: Tensor,
    direction: Tensor,
    beta: float = 1.0,
    *,
    order: int = 2,
) -> Tensor:
    r"""Exact jet of ``t -> soft_eisner(arc + t * direction, beta)``.

    ``arc`` / ``direction`` are ``(n + 1, n + 1)`` arc-score matrices; the returned
    ``(order + 1,)`` jet's ``c_1`` is the directional gradient of the projective-parse
    partition and ``2 c_2`` its directional curvature -- pinned to backend autodiff of
    :func:`omnibias.struct.torch.soft_eisner`.
    """
    spec: EisnerSpec = eisner_hypergraph(int(arc.shape[0]) - 1)
    zero = torch.zeros((), dtype=arc.dtype, device=arc.device)
    weight_jets: list[Tensor] = []
    for s in spec.edge_specs:
        if s[0] == "arc":
            h, m = int(s[1]), int(s[2])
            weight_jets.append(_linear_jet(arc[h, m], direction[h, m], order))
        else:
            weight_jets.append(_const_jet(zero, order))
    return hypergraph_lse_jet(spec.graph, weight_jets, beta, order=order)


def dp_value_jet(
    problem: ChainTrellis | DAG,
    direction: Tensor,
    beta: float = 1.0,
    *,
    order: int = 2,
    weights: Tensor | None = None,
) -> Tensor:
    r"""Dispatch the exact DP-value jet by problem type.

    * :class:`ChainTrellis` -- pulls ``emissions`` / ``transitions`` / ``start`` from the
      problem; ``direction`` perturbs the emissions ``(T, S)``.
    * :class:`DAG` -- requires the finite edge ``weights`` ``(n, n)``; ``direction``
      perturbs them.
    """
    if isinstance(problem, ChainTrellis):
        return chain_lse_jet(
            torch.as_tensor(problem.emissions),
            torch.as_tensor(problem.transitions),
            torch.as_tensor(direction),
            beta,
            order=order,
            start=torch.as_tensor(problem.start),
        )
    if weights is None:
        raise ValueError("dp_value_jet on a DAG requires the edge `weights` (n, n)")
    return dag_lse_jet(torch.as_tensor(weights), torch.as_tensor(direction), problem, beta, order=order)


__all__ = [
    "chain_lse_jet",
    "cky_lse_jet",
    "dag_lse_jet",
    "dp_value_jet",
    "eisner_lse_jet",
    "hypergraph_lse_jet",
    "lse2_jet",
]

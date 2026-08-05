# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable CKY inside / inside-outside parsing on the semiring driver (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.parse` (float64). Lifts a
:class:`~omnibias.struct._core.parse.BinaryGrammar` onto the shared hypergraph driver
(:func:`omnibias.struct.torch.semiring_value` / ``semiring_marginals``) so parsing needs no
bespoke recursion: :func:`soft_inside` is the ``lse_beta`` inside partition over all parse
trees (``-> best-parse score`` as ``beta -> inf``), and :func:`inside_outside` reads off the
closed-form **span / rule marginals** -- the exact gradient of ``soft_inside`` w.r.t. the
emission and rule scores, which the tests pin equal to ``autograd``.

The ``beta -> inf`` reduction is the relaxation axis; the ``delta -> 0`` tower differentiates
it exactly -- the same two axes as every other layer here.
"""

from __future__ import annotations

import torch
from omnibias.struct._core.parse import BinaryGrammar, ChartSpec, build_chart
from omnibias.struct.torch.semiring import semiring_marginals, semiring_value
from torch import Tensor


def _edge_weights(spec: ChartSpec, emit: Tensor, rule: Tensor) -> Tensor:
    rows: list[Tensor] = []
    for s in spec.edge_specs:
        if s[0] == "emit":
            rows.append(emit[int(s[1]), int(s[2])])
        else:
            rows.append(rule[int(s[1])])
    return torch.stack(rows)


def soft_inside(grammar: BinaryGrammar, emit: Tensor, rule: Tensor, beta: float = 1.0) -> Tensor:
    r"""Soft inside partition ``beta^-1 log sum_trees exp(beta score)`` of a CKY chart.

    ``emit`` is ``(L, R)`` lexical scores and ``rule`` is ``(num_rules,)`` binary-rule
    scores (both learnable). Differentiable in both; ``-> best-parse score`` as
    ``beta -> inf``. The score of a tree is the sum of its lexical + rule scores.
    """
    spec = build_chart(grammar, int(emit.shape[0]))
    return semiring_value(spec.graph, _edge_weights(spec, emit, rule), beta)


def inside_outside(
    grammar: BinaryGrammar, emit: Tensor, rule: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    r"""Closed-form ``(emit_marginals, rule_marginals)`` -- the gradient of :func:`soft_inside`.

    ``emit_marginals`` ``(L, R)`` is the expected lexical usage (``d soft_inside / d emit``)
    and ``rule_marginals`` ``(num_rules,)`` the expected rule usage. Inside-outside via the
    tower softmax; equal to ``autograd`` of :func:`soft_inside`.
    """
    spec = build_chart(grammar, int(emit.shape[0]))
    mu = semiring_marginals(spec.graph, _edge_weights(spec, emit, rule), beta)
    emit_marg = torch.zeros_like(emit)
    rule_marg = torch.zeros_like(rule)
    for e, s in enumerate(spec.edge_specs):
        if s[0] == "emit":
            emit_marg[int(s[1]), int(s[2])] = emit_marg[int(s[1]), int(s[2])] + mu[e]
        else:
            rule_marg[int(s[1])] = rule_marg[int(s[1])] + mu[e]
    return emit_marg, rule_marg


def span_marginals(
    grammar: BinaryGrammar, emit: Tensor, rule: Tensor, beta: float = 1.0
) -> Tensor:
    r"""Span marginals ``P_beta(nonterminal A spans [i, j))`` as an ``(R, L + 1, L + 1)`` tensor.

    ``[A, i, j]`` is the probability that item ``(A, i, j)`` is used by a parse; the start
    symbol over the whole sentence has marginal ``1``. Summing the edge marginals into each
    chart node.
    """
    spec = build_chart(grammar, int(emit.shape[0]))
    mu = semiring_marginals(spec.graph, _edge_weights(spec, emit, rule), beta)
    length_l, r_nt = int(emit.shape[0]), grammar.num_nonterminals
    out = torch.zeros((r_nt, length_l + 1, length_l + 1), dtype=emit.dtype, device=emit.device)
    node_span = {node: span for span, node in spec.node_of.items()}
    for e in range(spec.graph.num_edges):
        a, i, j = node_span[spec.graph.edges[e].head]
        out[a, i, j] = out[a, i, j] + mu[e]
    return out


def soft_inside_batched(
    grammar: BinaryGrammar, emit: Tensor, rule: Tensor, beta: float = 1.0
) -> Tensor:
    r"""Batched :func:`soft_inside` -> ``(B,)`` for ``emit`` ``(B, L, R)``, ``rule`` ``(B, num_rules)``.

    Shared ``grammar`` (so the chart hypergraph is identical across the batch); maps the
    inside partition over the leading batch axis with ``torch.func.vmap`` -- bit-identical to
    looping :func:`soft_inside`.
    """
    from torch.func import vmap

    def fwd(e: Tensor, r: Tensor) -> Tensor:
        return soft_inside(grammar, e, r, beta)

    out: Tensor = vmap(fwd)(emit, rule)
    return out


__all__ = ["inside_outside", "soft_inside", "soft_inside_batched", "span_marginals"]

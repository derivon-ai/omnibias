# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable CKY inside / inside-outside parsing on the semiring driver (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.parse` (float64 -- enable
``jax_enable_x64``). Lifts a :class:`~omnibias.struct._core.parse.BinaryGrammar` onto the
shared hypergraph driver: :func:`soft_inside` is the ``lse_beta`` inside partition over all
parse trees and :func:`inside_outside` reads off the closed-form span / rule marginals (the
exact gradient of ``soft_inside``, pinned equal to ``autograd``).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.struct._core.parse import BinaryGrammar, ChartSpec, build_chart
from omnibias.struct.jax.semiring import semiring_marginals, semiring_value


def _edge_weights(spec: ChartSpec, emit: Array, rule: Array) -> Array:
    rows: list[Array] = []
    for s in spec.edge_specs:
        if s[0] == "emit":
            rows.append(emit[int(s[1]), int(s[2])])
        else:
            rows.append(rule[int(s[1])])
    return jnp.stack(rows)


def soft_inside(grammar: BinaryGrammar, emit: Array, rule: Array, beta: float = 1.0) -> Array:
    r"""Soft inside partition ``beta^-1 log sum_trees exp(beta score)`` of a CKY chart.

    ``emit`` is ``(L, R)`` lexical scores and ``rule`` ``(num_rules,)`` binary-rule scores.
    Differentiable in both; ``-> best-parse score`` as ``beta -> inf``.
    """
    spec = build_chart(grammar, int(emit.shape[0]))
    return semiring_value(spec.graph, _edge_weights(spec, emit, rule), beta)


def inside_outside(
    grammar: BinaryGrammar, emit: Array, rule: Array, beta: float = 1.0
) -> tuple[Array, Array]:
    r"""Closed-form ``(emit_marginals, rule_marginals)`` -- the gradient of :func:`soft_inside`."""
    spec = build_chart(grammar, int(emit.shape[0]))
    mu = semiring_marginals(spec.graph, _edge_weights(spec, emit, rule), beta)
    emit_marg = jnp.zeros_like(emit)
    rule_marg = jnp.zeros_like(rule)
    for e, s in enumerate(spec.edge_specs):
        if s[0] == "emit":
            emit_marg = emit_marg.at[int(s[1]), int(s[2])].add(mu[e])
        else:
            rule_marg = rule_marg.at[int(s[1])].add(mu[e])
    return emit_marg, rule_marg


def span_marginals(grammar: BinaryGrammar, emit: Array, rule: Array, beta: float = 1.0) -> Array:
    r"""Span marginals ``P_beta(nonterminal A spans [i, j))`` as an ``(R, L + 1, L + 1)`` array."""
    spec = build_chart(grammar, int(emit.shape[0]))
    mu = semiring_marginals(spec.graph, _edge_weights(spec, emit, rule), beta)
    length_l, r_nt = int(emit.shape[0]), grammar.num_nonterminals
    out = jnp.zeros((r_nt, length_l + 1, length_l + 1), dtype=emit.dtype)
    node_span = {node: span for span, node in spec.node_of.items()}
    for e in range(spec.graph.num_edges):
        a, i, j = node_span[spec.graph.edges[e].head]
        out = out.at[a, i, j].add(mu[e])
    return out


def soft_inside_batched(
    grammar: BinaryGrammar, emit: Array, rule: Array, beta: float = 1.0
) -> Array:
    r"""Batched :func:`soft_inside` -> ``(B,)`` for ``emit`` ``(B, L, R)``, ``rule`` ``(B, num_rules)`` (``jax.vmap``)."""
    import jax

    out: Array = jax.vmap(lambda e, r: soft_inside(grammar, e, r, beta))(emit, rule)
    return out


__all__ = ["inside_outside", "soft_inside", "soft_inside_batched", "span_marginals"]

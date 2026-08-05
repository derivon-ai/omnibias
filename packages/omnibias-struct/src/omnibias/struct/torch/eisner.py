# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Eisner projective dependency parsing on the semiring driver (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.eisner` (float64). Lifts Eisner's
complete/incomplete span DP onto the shared hypergraph driver
(:func:`omnibias.struct.torch.semiring_value` / ``semiring_marginals``): :func:`soft_eisner`
is the ``lse_beta`` partition over all projective trees (``-> best projective parse`` as
``beta -> inf``), and :func:`eisner_marginals` reads off the closed-form **arc marginals**
``P_beta(arc h -> m)`` -- the exact gradient of ``soft_eisner`` w.r.t. the arc-score matrix,
pinned equal to ``autograd``. The two axes stay apart: ``beta -> inf`` is the relaxation,
``delta -> 0`` differentiates it exactly.
"""

from __future__ import annotations

import torch
from omnibias.struct._core.eisner import EisnerSpec, eisner_hypergraph
from omnibias.struct.torch.semiring import semiring_marginals, semiring_value
from torch import Tensor


def _edge_weights(spec: EisnerSpec, arc: Tensor) -> Tensor:
    zero = torch.zeros((), dtype=arc.dtype, device=arc.device)
    rows: list[Tensor] = []
    for s in spec.edge_specs:
        rows.append(arc[int(s[1]), int(s[2])] if s[0] == "arc" else zero)
    return torch.stack(rows)


def soft_eisner(arc: Tensor, beta: float = 1.0) -> Tensor:
    r"""Soft projective-parse partition ``beta^-1 log sum_trees exp(beta score)`` of ``arc``.

    ``arc`` is the ``(n + 1, n + 1)`` arc-score matrix (``arc[h, m]`` = head ``h`` -> modifier
    ``m``; row/column ``0`` is the ``ROOT``). Differentiable in ``arc``; ``-> best projective
    tree score`` as ``beta -> inf``. The score of a tree is the sum of its arc scores.
    """
    spec = eisner_hypergraph(int(arc.shape[0]) - 1)
    return semiring_value(spec.graph, _edge_weights(spec, arc), beta)


def eisner_marginals(arc: Tensor, beta: float = 1.0) -> Tensor:
    r"""Closed-form arc marginals ``P_beta(arc h -> m)`` as an ``(n + 1, n + 1)`` matrix.

    Equals ``d soft_eisner / d arc`` (the exact gradient); each modifier column ``m >= 1``
    sums to ``1`` (every word has exactly one head). Inside-outside via the tower softmax;
    equal to ``autograd`` of :func:`soft_eisner`. As ``beta -> inf`` it concentrates on the
    best projective tree's arcs.
    """
    spec = eisner_hypergraph(int(arc.shape[0]) - 1)
    mu = semiring_marginals(spec.graph, _edge_weights(spec, arc), beta)
    out = torch.zeros_like(arc)
    for e, s in enumerate(spec.edge_specs):
        if s[0] == "arc":
            out[int(s[1]), int(s[2])] = out[int(s[1]), int(s[2])] + mu[e]
    return out


def soft_eisner_batched(arc: Tensor, beta: float = 1.0) -> Tensor:
    r"""Batched :func:`soft_eisner` -> ``(B,)`` for ``arc`` ``(B, n + 1, n + 1)`` (via ``torch.func.vmap``)."""
    from torch.func import vmap

    def fwd(a: Tensor) -> Tensor:
        return soft_eisner(a, beta)

    out: Tensor = vmap(fwd)(arc)
    return out


__all__ = ["eisner_marginals", "soft_eisner", "soft_eisner_batched"]

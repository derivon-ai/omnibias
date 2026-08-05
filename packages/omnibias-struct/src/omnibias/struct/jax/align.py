# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Needleman-Wunsch sequence alignment (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.align` (float64; needs ``jax_enable_x64``).
Global alignment is a longest-path DAG, so this *reuses the shared shortest-path substrate*:
assemble a weight matrix ``W[u, v] = -score(edge)`` (differentiable in the substitution
matrix ``sub`` and gap penalty ``gap``) then call
:func:`omnibias.struct.jax.soft_shortest_path`. The ``beta -> inf`` softmax anneals to the
NW optimum; the ``delta -> 0`` tower gives closed-form parameter gradients
(:func:`soft_align_marginals`), equal to ``jax.grad``. Do not conflate the two axes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.struct._core.align import AlignmentLattice, build_gotoh_dag, build_local_dag
from omnibias.struct._core.trellis import DAG
from omnibias.struct.jax.soft_dp import soft_shortest_path, soft_shortest_path_marginals


def _assemble(
    a: object, b: object, sub: Array, gap: Array
) -> tuple[Array, DAG, dict[tuple[int, int], tuple[str, int, int]], AlignmentLattice]:
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    lattice = AlignmentLattice(ai.shape[0], bj.shape[0])
    dag, labels = lattice.build_dag()
    n_nodes = lattice.num_nodes
    weight = jnp.zeros((n_nodes, n_nodes), dtype=sub.dtype)
    for (u, v), (kind, i, j) in labels.items():
        val = -sub[int(ai[i]), int(bj[j])] if kind == "sub" else -gap
        weight = weight.at[u, v].set(val)
    return weight, dag, labels, lattice


def soft_align(a: object, b: object, sub: Array, gap: Array, beta: float = 1.0) -> Array:
    r"""Soft global-alignment score ``beta^-1 log sum_paths exp(beta score)`` of ``a`` vs ``b``.

    ``sub`` is a ``(K, K)`` substitution-score matrix and ``gap`` a scalar gap penalty (both
    learnable). Differentiable in ``sub`` and ``gap``; ``-> Needleman-Wunsch optimum`` as
    ``beta -> inf``.
    """
    weight, dag, _, _ = _assemble(a, b, sub, jnp.asarray(gap, dtype=sub.dtype))
    return -soft_shortest_path(weight, dag, beta)


def soft_align_marginals(
    a: object, b: object, sub: Array, gap: Array, beta: float = 1.0
) -> tuple[Array, Array]:
    r"""Closed-form parameter gradients ``(d/d sub, d/d gap)`` of :func:`soft_align`.

    Forward-backward edge marginals ``xi`` from the tower softmax, folded back onto the
    parameters: ``grad_sub[p, q]`` is the expected number of ``(p, q)`` substitutions and
    ``grad_gap`` the expected number of gaps. Equal to ``jax.grad`` of :func:`soft_align`.
    """
    gap_a = jnp.asarray(gap, dtype=sub.dtype)
    weight, dag, labels, _ = _assemble(a, b, sub, gap_a)
    xi = soft_shortest_path_marginals(weight, dag, beta)
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    grad_sub = jnp.zeros_like(sub)
    grad_gap = jnp.zeros((), dtype=sub.dtype)
    for (u, v), (kind, i, j) in labels.items():
        if kind == "sub":
            grad_sub = grad_sub.at[int(ai[i]), int(bj[j])].add(xi[u, v])
        else:
            grad_gap = grad_gap + xi[u, v]
    return grad_sub, grad_gap


# ---------------------------------------------------------------------------
# Smith-Waterman local alignment (free start / end on the shortest-path substrate)
# ---------------------------------------------------------------------------


def _assemble_local(
    a: object, b: object, sub: Array, gap: Array
) -> tuple[Array, DAG, dict[tuple[int, int], tuple[str, int, int]]]:
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    dag, labels = build_local_dag(ai.shape[0], bj.shape[0])
    weight = jnp.zeros((dag.num_nodes, dag.num_nodes), dtype=sub.dtype)
    for (u, v), (kind, i, j) in labels.items():
        if kind == "sub":
            weight = weight.at[u, v].set(-sub[int(ai[i]), int(bj[j])])
        elif kind == "gap":
            weight = weight.at[u, v].set(-gap)
    return weight, dag, labels


def soft_local_align(a: object, b: object, sub: Array, gap: Array, beta: float = 1.0) -> Array:
    r"""Soft Smith-Waterman local-alignment score ``beta^-1 log sum_paths exp(beta score)``.

    Free start / end 0-edges on the shared shortest-path substrate, so ``-> the
    Smith-Waterman local optimum`` as ``beta -> inf`` (with the empty alignment as the
    score-``0`` floor). Differentiable in the substitution matrix ``sub`` and gap penalty
    ``gap``.
    """
    weight, dag, _ = _assemble_local(a, b, sub, jnp.asarray(gap, dtype=sub.dtype))
    value: Array = -soft_shortest_path(weight, dag, beta)
    return value


def soft_local_align_marginals(
    a: object, b: object, sub: Array, gap: Array, beta: float = 1.0
) -> tuple[Array, Array]:
    r"""Closed-form parameter gradients ``(d/d sub, d/d gap)`` of :func:`soft_local_align`."""
    gap_a = jnp.asarray(gap, dtype=sub.dtype)
    weight, dag, labels = _assemble_local(a, b, sub, gap_a)
    xi = soft_shortest_path_marginals(weight, dag, beta)
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    grad_sub = jnp.zeros_like(sub)
    grad_gap = jnp.zeros((), dtype=sub.dtype)
    for (u, v), (kind, i, j) in labels.items():
        if kind == "sub":
            grad_sub = grad_sub.at[int(ai[i]), int(bj[j])].add(xi[u, v])
        elif kind == "gap":
            grad_gap = grad_gap + xi[u, v]
    return grad_sub, grad_gap


# ---------------------------------------------------------------------------
# Gotoh affine gaps (3-state M / Ix / Iy lattice)
# ---------------------------------------------------------------------------


def _assemble_gotoh(
    a: object, b: object, sub: Array, gap_open: Array, gap_extend: Array
) -> tuple[Array, DAG, dict[tuple[int, int], tuple[str, int, int]]]:
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    dag, labels = build_gotoh_dag(ai.shape[0], bj.shape[0])
    weight = jnp.zeros((dag.num_nodes, dag.num_nodes), dtype=sub.dtype)
    for (u, v), (kind, i, j) in labels.items():
        if kind == "sub":
            weight = weight.at[u, v].set(-sub[int(ai[i]), int(bj[j])])
        elif kind == "open":
            weight = weight.at[u, v].set(-(gap_open + gap_extend))
        elif kind == "extend":
            weight = weight.at[u, v].set(-gap_extend)
    return weight, dag, labels


def soft_gotoh(
    a: object, b: object, sub: Array, gap_open: Array, gap_extend: Array, beta: float = 1.0
) -> Array:
    r"""Soft Gotoh affine-gap alignment score ``beta^-1 log sum_paths exp(beta score)``.

    A gap of length ``L`` scores ``gap_open + L * gap_extend`` (both learnable). Differentiable
    in ``sub`` / ``gap_open`` / ``gap_extend``; ``-> the Gotoh optimum`` as ``beta -> inf``.
    """
    weight, dag, _ = _assemble_gotoh(
        a, b, sub, jnp.asarray(gap_open, dtype=sub.dtype), jnp.asarray(gap_extend, dtype=sub.dtype)
    )
    value: Array = -soft_shortest_path(weight, dag, beta)
    return value


def soft_gotoh_marginals(
    a: object, b: object, sub: Array, gap_open: Array, gap_extend: Array, beta: float = 1.0
) -> tuple[Array, Array, Array]:
    r"""Closed-form gradients ``(d/d sub, d/d gap_open, d/d gap_extend)`` of :func:`soft_gotoh`.

    Each gap-open edge scores ``gap_open + gap_extend`` (so it feeds both gradients) and each
    gap-extend edge scores ``gap_extend``; folding the forward-backward edge marginals back
    gives the expected substitution / open / (open + extend) counts, equal to ``jax.grad``.
    """
    open_a = jnp.asarray(gap_open, dtype=sub.dtype)
    extend_a = jnp.asarray(gap_extend, dtype=sub.dtype)
    weight, dag, labels = _assemble_gotoh(a, b, sub, open_a, extend_a)
    xi = soft_shortest_path_marginals(weight, dag, beta)
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    grad_sub = jnp.zeros_like(sub)
    grad_open = jnp.zeros((), dtype=sub.dtype)
    grad_extend = jnp.zeros((), dtype=sub.dtype)
    for (u, v), (kind, i, j) in labels.items():
        if kind == "sub":
            grad_sub = grad_sub.at[int(ai[i]), int(bj[j])].add(xi[u, v])
        elif kind == "open":
            grad_open = grad_open + xi[u, v]
            grad_extend = grad_extend + xi[u, v]
        elif kind == "extend":
            grad_extend = grad_extend + xi[u, v]
    return grad_sub, grad_open, grad_extend


def soft_align_batched(a: object, b: object, sub: Array, gap: Array, beta: float = 1.0) -> Array:
    r"""Batched :func:`soft_align` -> ``(B,)`` for ``sub`` ``(B, K, K)`` (shared sequences ``a``, ``b``).

    ``gap`` may be a shared scalar or a per-example ``(B,)`` vector. Maps :func:`soft_align`
    over the leading batch axis with ``jax.vmap`` -- bit-identical to looping.
    """
    import jax

    gap_a = jnp.asarray(gap, dtype=sub.dtype)
    gap_dim = 0 if gap_a.ndim == 1 else None

    def fwd(s: Array, g: Array) -> Array:
        return soft_align(a, b, s, g, beta)

    out: Array = jax.vmap(fwd, in_axes=(0, gap_dim))(sub, gap_a)
    return out


__all__ = [
    "soft_align",
    "soft_align_batched",
    "soft_align_marginals",
    "soft_gotoh",
    "soft_gotoh_marginals",
    "soft_local_align",
    "soft_local_align_marginals",
]

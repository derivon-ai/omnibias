# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable non-projective matrix-tree dependency parsing (JAX).

Bit-identical twin of :mod:`omnibias.struct.torch.mtt` (float64). By Tutte's directed
Matrix-Tree Theorem the partition over the exponentially many spanning arborescences of the
arc graph is a single determinant, so :func:`soft_matrix_tree` is an **exact** soft value
``log det L(beta) / beta`` (no ``lse_beta`` relaxation of a ``max`` -- the determinant sums the
trees exactly) and :func:`matrix_tree_marginals` reads the closed-form arc marginals off
``L^{-1}`` (Koo et al. 2007), pinned equal to ``autograd``. ``beta -> inf`` is still the
temperature axis (``-> `` the Chu-Liu/Edmonds maximum arborescence); the difference from the
rest of the package is that here it is exact at finite ``beta``. Column-max stabilised so the
determinant / inverse stay finite at large ``beta``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _laplacian_tilde(arc: Array, beta: float) -> tuple[Array, Array, Array]:
    r"""Column-scaled Kirchhoff Laplacian ``(L_tilde, c, tw)`` of ``exp(beta * arc)``.

    ``c[m]`` is the per-modifier max log-weight pulled out for stability; ``tw[h, m]`` are the
    scaled weights (``0`` where ``h == m`` or ``m == 0``); ``L_tilde = diag(colsum tw) - tw``
    restricted to the word block.
    """
    n = int(arc.shape[0]) - 1
    lw = beta * arc
    eye = jnp.eye(n + 1, dtype=bool)
    valid = ~eye
    neg = jnp.full_like(lw, -jnp.inf)
    c = jnp.where(valid, lw, neg).max(axis=0)[1:]  # (n,), over heads h != m, m = 1..n
    shifted = jnp.where(valid[:, 1:], lw[:, 1:] - c[jnp.newaxis, :], neg[:, 1:])
    tw = jnp.exp(shifted)  # (n + 1, n); invalid entries -> 0
    colsum = tw.sum(axis=0)  # (n,)
    aword = tw[1:, :]  # (n, n), word-head block (zero diagonal)
    ltilde = jnp.diag(colsum) - aword
    return ltilde, c, tw


def soft_matrix_tree(arc: Array, beta: float = 1.0) -> Array:
    r"""Exact non-projective soft value ``log det L(beta) / beta`` of the arc-score matrix.

    ``arc`` is the ``(n + 1, n + 1)`` arc-score matrix (``arc[h, m]`` = head ``h`` -> modifier
    ``m``; row/column ``0`` is the ROOT wall). Differentiable in ``arc``; the determinant is
    the *exact* sum over all spanning arborescences, and ``-> `` the maximum-arborescence
    score as ``beta -> inf``.
    """
    ltilde, c, _ = _laplacian_tilde(arc, beta)
    _sign, logabsdet = jnp.linalg.slogdet(ltilde)
    value: Array = (logabsdet + c.sum()) / beta
    return value


def matrix_tree_marginals(arc: Array, beta: float = 1.0) -> Array:
    r"""Closed-form arc marginals ``P_beta(h -> m)`` via ``L^{-1}`` as an ``(n + 1, n + 1)`` matrix.

    Equals ``d soft_matrix_tree / d arc`` (the exact gradient); each modifier column ``m >= 1``
    sums to ``1`` (every word takes exactly one head). As ``beta -> inf`` it concentrates on
    the maximum arborescence's arcs.
    """
    ltilde, _c, tw = _laplacian_tilde(arc, beta)
    binv = jnp.linalg.inv(ltilde)
    diag_b = jnp.diagonal(binv)
    root_tilde = tw[0, :]
    aword = tw[1:, :]
    p_root = root_tilde * diag_b  # (n,)
    p_word = aword * (diag_b[jnp.newaxis, :] - binv.T)  # (n, n)
    out = jnp.zeros_like(arc)
    out = out.at[0, 1:].set(p_root)
    out = out.at[1:, 1:].set(p_word)
    return out


__all__ = ["matrix_tree_marginals", "soft_matrix_tree"]

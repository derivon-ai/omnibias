# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable non-projective matrix-tree dependency parsing (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.mtt` (float64). By Tutte's directed
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

import torch
from torch import Tensor


def _laplacian_tilde(arc: Tensor, beta: float) -> tuple[Tensor, Tensor, Tensor]:
    r"""Column-scaled Kirchhoff Laplacian ``(L_tilde, c, tw)`` of ``exp(beta * arc)``.

    ``c[m]`` is the per-modifier max log-weight pulled out for stability; ``tw[h, m]`` are the
    scaled weights (``0`` where ``h == m`` or ``m == 0``); ``L_tilde = diag(colsum tw) - tw``
    restricted to the word block.
    """
    n = int(arc.shape[0]) - 1
    lw = beta * arc
    eye = torch.eye(n + 1, dtype=torch.bool, device=arc.device)
    valid = ~eye
    neg = torch.full_like(lw, -float("inf"))
    c = torch.where(valid, lw, neg).max(dim=0).values[1:]  # (n,), over heads h != m, m = 1..n
    shifted = torch.where(valid[:, 1:], lw[:, 1:] - c.unsqueeze(0), neg[:, 1:])
    tw = torch.exp(shifted)  # (n + 1, n); invalid entries -> 0
    colsum = tw.sum(dim=0)  # (n,)
    aword = tw[1:, :]  # (n, n), word-head block (zero diagonal)
    ltilde = torch.diag(colsum) - aword
    return ltilde, c, tw


def soft_matrix_tree(arc: Tensor, beta: float = 1.0) -> Tensor:
    r"""Exact non-projective soft value ``log det L(beta) / beta`` of the arc-score matrix.

    ``arc`` is the ``(n + 1, n + 1)`` arc-score matrix (``arc[h, m]`` = head ``h`` -> modifier
    ``m``; row/column ``0`` is the ROOT wall). Differentiable in ``arc``; the determinant is
    the *exact* sum over all spanning arborescences, and ``-> `` the maximum-arborescence
    score as ``beta -> inf``.
    """
    ltilde, c, _ = _laplacian_tilde(arc, beta)
    _sign, logabsdet = torch.linalg.slogdet(ltilde)
    value: Tensor = (logabsdet + c.sum()) / beta
    return value


def matrix_tree_marginals(arc: Tensor, beta: float = 1.0) -> Tensor:
    r"""Closed-form arc marginals ``P_beta(h -> m)`` via ``L^{-1}`` as an ``(n + 1, n + 1)`` matrix.

    Equals ``d soft_matrix_tree / d arc`` (the exact gradient); each modifier column ``m >= 1``
    sums to ``1`` (every word takes exactly one head). As ``beta -> inf`` it concentrates on
    the maximum arborescence's arcs.
    """
    ltilde, _c, tw = _laplacian_tilde(arc, beta)
    binv = torch.linalg.inv(ltilde)
    diag_b = torch.diagonal(binv)
    root_tilde = tw[0, :]
    aword = tw[1:, :]
    p_root = root_tilde * diag_b  # (n,)
    p_word = aword * (diag_b.unsqueeze(0) - binv.t())  # (n, n)
    out = torch.zeros_like(arc)
    out[0, 1:] = p_root
    out[1:, 1:] = p_word
    return out


__all__ = ["matrix_tree_marginals", "soft_matrix_tree"]

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-form jets of a soft tree / arrangement (product of sigmoid gates).

Per gate ``g_j = sigma(beta (w_j . x - t_j))`` via :func:`layer_jet_mv`; per leaf
membership by the Leibniz product (:func:`jet_multiply`); output the linear
combination of leaves. This is exact for the **soft** surrogate (the
``delta -> 0`` jet register). Hardening remains the ``beta -> inf`` temperature
axis and is not claimed as a jet of the discontinuous truth.

Does not rewire PINN :class:`PartitionedField` (that path stays autodiff).
"""

from __future__ import annotations

import numpy as np
import torch
from omnibias.tab._core.jet import TreeJet
from omnibias.tab._core.params import TabParams, leaf_code_matrix
from omnibias.torch.jet import jet_to_tower, layer_jet, mlp_jet
from omnibias.torch.jet_mv import identity_jet, jet_multiply, jet_partials, layer_jet_mv, mlp_jet_mv
from torch import Tensor

_DTYPE = torch.float64


def _complement_jet(g: Tensor) -> Tensor:
    return torch.cat([(1.0 - g[:1]), -g[1:]], dim=0)


def _path_jet(x0: Tensor, v: Tensor, order: int) -> Tensor:
    rows = [x0]
    if order >= 1:
        rows.append(v)
    rows.extend(torch.zeros_like(x0) for _ in range(max(0, order - 1)))
    return torch.stack(rows[: order + 1], dim=0)


def _multiply_1d(a: Tensor, b: Tensor) -> Tensor:
    n = int(a.shape[0])
    out = []
    for i in range(n):
        acc = a[0] * b[i]
        for k in range(1, i + 1):
            acc = acc + a[k] * b[i - k]
        out.append(acc)
    return torch.stack(out, dim=0)


def _gate_jet_mv(
    x_jet: Tensor, w: Tensor, t: Tensor, beta: float, dim: int, order: int
) -> Tensor:
    W = (float(beta) * w).reshape(1, -1)
    b = (-float(beta) * t).reshape(1)
    return layer_jet_mv(x_jet, W, b, "sigmoid", dim, order).reshape(x_jet.shape[0])


def _gate_jet_dir(
    x_jet: Tensor, w: Tensor, t: Tensor, beta: float, order: int
) -> Tensor:
    W = (float(beta) * w).reshape(1, -1)
    b = (-float(beta) * t).reshape(1)
    return layer_jet(x_jet, W, b, "sigmoid", order).reshape(x_jet.shape[0])


def _leaf_mu_mv(
    gates: list[Tensor], code: Tensor, dim: int, order: int
) -> Tensor:
    acc: Tensor | None = None
    for j, g in enumerate(gates):
        fac = g if float(code[j]) > 0.5 else _complement_jet(g)
        acc = fac if acc is None else jet_multiply(acc, fac, dim, order)
    assert acc is not None
    return acc


def _leaf_mu_dir(gates: list[Tensor], code: Tensor) -> Tensor:
    acc: Tensor | None = None
    for j, g in enumerate(gates):
        fac = g if float(code[j]) > 0.5 else _complement_jet(g)
        acc = fac if acc is None else _multiply_1d(acc, fac)
    assert acc is None or acc is not None
    assert acc is not None
    return acc


def _additive_mlp_layers(
    W: Tensor, t: Tensor, leaves: Tensor, b0: Tensor, beta: float
) -> list[tuple[Tensor, Tensor, str | None]]:
    """``Linear -> sigmoid -> Linear`` weights matching ``to_additive_sequential``."""
    b = float(beta)
    W1 = b * W[:, 0, :]
    b1 = -b * t[:, 0]
    u = leaves[:, 1, :] - leaves[:, 0, :]
    W2 = u.transpose(0, 1)
    b2 = b0 + leaves[:, 0, :].sum(dim=0)
    return [(W1, b1, "sigmoid"), (W2, b2, None)]


def _tree_jet_one_product(
    W: Tensor,
    t: Tensor,
    leaves: Tensor,
    b0: Tensor,
    x0: Tensor,
    beta: float,
    order: int,
) -> Tensor:
    """Leibniz product-of-sigmoids jet (any depth)."""
    dim = int(x0.shape[0])
    x_jet = identity_jet(x0, order)
    k = int(leaves.shape[-1])
    m = int(x_jet.shape[0])
    out = x0.new_zeros(m, k)
    out[0] = b0
    n_trees, depth = int(W.shape[0]), int(W.shape[1])
    codes = torch.as_tensor(leaf_code_matrix(depth), dtype=x0.dtype, device=x0.device)
    n_leaves = int(codes.shape[0])
    for tree in range(n_trees):
        gates = [
            _gate_jet_mv(x_jet, W[tree, j], t[tree, j], beta, dim, order)
            for j in range(depth)
        ]
        for ell in range(n_leaves):
            mu = _leaf_mu_mv(gates, codes[ell], dim, order)
            out = out + mu.unsqueeze(-1) * leaves[tree, ell]
    return out


def _tree_jet_one(
    W: Tensor,
    t: Tensor,
    leaves: Tensor,
    b0: Tensor,
    x0: Tensor,
    beta: float,
    order: int,
) -> Tensor:
    if int(W.shape[1]) == 1:
        return mlp_jet_mv(x0, _additive_mlp_layers(W, t, leaves, b0, beta), order)
    return _tree_jet_one_product(W, t, leaves, b0, x0, beta, order)


def _tree_jet_dir_one_product(
    W: Tensor,
    t: Tensor,
    leaves: Tensor,
    b0: Tensor,
    x0: Tensor,
    v: Tensor,
    beta: float,
    order: int,
) -> Tensor:
    x_jet = _path_jet(x0, v, order)
    k = int(leaves.shape[-1])
    out = x0.new_zeros(order + 1, k)
    out[0] = b0
    n_trees, depth = int(W.shape[0]), int(W.shape[1])
    codes = torch.as_tensor(leaf_code_matrix(depth), dtype=x0.dtype, device=x0.device)
    n_leaves = int(codes.shape[0])
    for tree in range(n_trees):
        gates = [
            _gate_jet_dir(x_jet, W[tree, j], t[tree, j], beta, order)
            for j in range(depth)
        ]
        for ell in range(n_leaves):
            mu = _leaf_mu_dir(gates, codes[ell])
            out = out + mu.unsqueeze(-1) * leaves[tree, ell]
    return out


def _tree_jet_dir_one(
    W: Tensor,
    t: Tensor,
    leaves: Tensor,
    b0: Tensor,
    x0: Tensor,
    v: Tensor,
    beta: float,
    order: int,
) -> Tensor:
    if int(W.shape[1]) == 1:
        return mlp_jet(x0, v, _additive_mlp_layers(W, t, leaves, b0, beta), order)
    return _tree_jet_dir_one_product(W, t, leaves, b0, x0, v, beta, order)


def _as_arrays(
    params: TabParams,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(params.W, dtype=_DTYPE),
        torch.as_tensor(params.t, dtype=_DTYPE),
        torch.as_tensor(params.leaves, dtype=_DTYPE),
        torch.as_tensor(params.b0, dtype=_DTYPE),
    )


def extract_tree_jet(
    params: TabParams,
    X: np.ndarray,
    *,
    beta: float,
    max_order: int,
    output_index: int = 0,
) -> TreeJet:
    r"""Multivariate jet of the **soft** tree at each row of ``X``.

    Returns a :class:`TreeJet` (FieldJet layout). Exact for the soft surrogate,
    not the ``beta -> inf`` hardened tree.
    """
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=_DTYPE)
    W, t, leaves, b0 = _as_arrays(params)
    order = int(max_order)
    rows = []
    for i in range(int(Xt.shape[0])):
        jet = _tree_jet_one(W, t, leaves, b0, Xt[i], float(beta), order)
        rows.append(jet[:, int(output_index)] if jet.ndim == 2 else jet)
    stacked = torch.stack(rows, dim=1)
    dim = int(Xt.shape[1])
    raw = jet_partials(stacked, dim, order)
    partials = {alpha: val.detach().cpu().numpy() for alpha, val in raw.items()}
    names = tuple(f"x{i}" for i in range(dim))
    return TreeJet(
        X=np.asarray(X, dtype=np.float64),
        order=order,
        partials=partials,
        var_names=names,
    )


def extract_arrangement_jet(
    W: np.ndarray,
    t: np.ndarray,
    cell_logits: np.ndarray,
    X: np.ndarray,
    *,
    beta: float,
    max_order: int,
    output_index: int = 0,
) -> TreeJet:
    r"""Multivariate jet of one arrangement (a single product-of-sigmoids tree)."""
    Ww = np.asarray(W, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    cell = np.asarray(cell_logits, dtype=np.float64)
    if cell.ndim == 1:
        cell = cell.reshape(-1, 1)
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab._core.params import TabParams as _TP

    cfg = SoftTreeConfig(
        n_features=int(Ww.shape[1]),
        n_trees=1,
        depth=int(Ww.shape[0]),
        n_outputs=int(cell.shape[-1]),
        task="regression",
    )
    params = _TP(cfg, Ww.reshape(1, *Ww.shape), tt.reshape(1, -1), cell.reshape(1, *cell.shape), np.zeros(cell.shape[-1]))
    return extract_tree_jet(params, X, beta=beta, max_order=max_order, output_index=output_index)


def extract_tree_jet_directional(
    params: TabParams,
    X: np.ndarray,
    v: np.ndarray,
    *,
    beta: float,
    max_order: int,
    output_index: int = 0,
) -> np.ndarray:
    r"""Directional derivative tower ``(order+1, n)`` of the soft tree along ``v``."""
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=_DTYPE)
    vt = torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=_DTYPE)
    if vt.ndim == 1:
        vt = vt.expand(Xt.shape[0], -1)
    W, t, leaves, b0 = _as_arrays(params)
    order = int(max_order)
    rows = []
    for i in range(int(Xt.shape[0])):
        jet = _tree_jet_dir_one(W, t, leaves, b0, Xt[i], vt[i], float(beta), order)
        col = jet[:, int(output_index)] if jet.ndim == 2 else jet
        rows.append(jet_to_tower(col))
    return torch.stack(rows, dim=1).detach().cpu().numpy()


def sequential_mlp_jet(
    seq: torch.nn.Sequential,
    x0: Tensor,
    v: Tensor,
    order: int,
) -> Tensor:
    r"""Directional jet of a depth-1 ``Linear -> Sigmoid -> Linear`` sequential."""
    from omnibias.torch.jet import mlp_jet

    lin1, _, lin2 = seq[0], seq[1], seq[2]
    return mlp_jet(
        x0,
        v,
        [
            (lin1.weight, lin1.bias, "sigmoid"),
            (lin2.weight, lin2.bias, None),
        ],
        int(order),
    )


__all__ = [
    "TreeJet",
    "extract_arrangement_jet",
    "extract_tree_jet",
    "extract_tree_jet_directional",
    "sequential_mlp_jet",
]

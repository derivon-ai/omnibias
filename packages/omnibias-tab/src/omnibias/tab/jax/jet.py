# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""JAX twin of :mod:`omnibias.tab.torch.jet` (product-of-sigmoids tree jets).

Bit-identical to the torch kernels at float64 (parity ``~1e-9``). Exact for the
**soft** surrogate; hardening is the ``beta -> inf`` axis.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from omnibias.jax.jet import jet_to_tower, layer_jet, mlp_jet
from omnibias.jax.jet_mv import identity_jet, jet_multiply, jet_partials, layer_jet_mv, mlp_jet_mv
from omnibias.tab._core.jet import TreeJet
from omnibias.tab._core.params import TabParams, leaf_code_matrix


def _complement_jet(g: Any) -> Any:
    return jnp.concatenate([(1.0 - g[:1]), -g[1:]], axis=0)


def _path_jet(x0: Any, v: Any, order: int) -> Any:
    rows = [x0]
    if order >= 1:
        rows.append(v)
    zeros = jnp.zeros_like(x0)
    rows.extend(zeros for _ in range(max(0, order - 1)))
    return jnp.stack(rows[: order + 1], axis=0)


def _multiply_1d(a: Any, b: Any) -> Any:
    n = int(a.shape[0])
    out = []
    for i in range(n):
        acc = a[0] * b[i]
        for k in range(1, i + 1):
            acc = acc + a[k] * b[i - k]
        out.append(acc)
    return jnp.stack(out, axis=0)


def _gate_jet_mv(x_jet: Any, w: Any, t: Any, beta: float, dim: int, order: int) -> Any:
    W = (float(beta) * w).reshape((1, -1))
    b = (-float(beta) * t).reshape((1,))
    return layer_jet_mv(x_jet, W, b, "sigmoid", dim, order).reshape((x_jet.shape[0],))


def _gate_jet_dir(x_jet: Any, w: Any, t: Any, beta: float, order: int) -> Any:
    W = (float(beta) * w).reshape((1, -1))
    b = (-float(beta) * t).reshape((1,))
    return layer_jet(x_jet, W, b, "sigmoid", order).reshape((x_jet.shape[0],))


def _leaf_mu_mv(gates: list[Any], code: np.ndarray, dim: int, order: int) -> Any:
    acc = None
    for j, g in enumerate(gates):
        fac = g if float(code[j]) > 0.5 else _complement_jet(g)
        acc = fac if acc is None else jet_multiply(acc, fac, dim, order)
    assert acc is not None
    return acc


def _leaf_mu_dir(gates: list[Any], code: np.ndarray) -> Any:
    acc = None
    for j, g in enumerate(gates):
        fac = g if float(code[j]) > 0.5 else _complement_jet(g)
        acc = fac if acc is None else _multiply_1d(acc, fac)
    assert acc is not None
    return acc


def _additive_mlp_layers(
    W: Any, t: Any, leaves: Any, b0: Any, beta: float
) -> list[tuple[Any, Any, str | None]]:
    b = float(beta)
    W1 = b * W[:, 0, :]
    b1 = -b * t[:, 0]
    u = leaves[:, 1, :] - leaves[:, 0, :]
    W2 = u.T
    b2 = b0 + leaves[:, 0, :].sum(axis=0)
    return [(W1, b1, "sigmoid"), (W2, b2, None)]


def _tree_jet_one_product(
    W: Any, t: Any, leaves: Any, b0: Any, x0: Any, beta: float, order: int
) -> Any:
    dim = int(x0.shape[0])
    x_jet = identity_jet(x0, order)
    k = int(leaves.shape[-1])
    m = int(x_jet.shape[0])
    out = jnp.zeros((m, k), dtype=x0.dtype)
    out = out.at[0].set(b0)
    n_trees, depth = int(W.shape[0]), int(W.shape[1])
    codes = np.asarray(leaf_code_matrix(depth))
    n_leaves = int(codes.shape[0])
    for tree in range(n_trees):
        gates = [
            _gate_jet_mv(x_jet, W[tree, j], t[tree, j], beta, dim, order)
            for j in range(depth)
        ]
        for ell in range(n_leaves):
            mu = _leaf_mu_mv(gates, codes[ell], dim, order)
            out = out + mu[:, None] * leaves[tree, ell]
    return out


def _tree_jet_one(
    W: Any, t: Any, leaves: Any, b0: Any, x0: Any, beta: float, order: int
) -> Any:
    if int(W.shape[1]) == 1:
        return mlp_jet_mv(x0, _additive_mlp_layers(W, t, leaves, b0, beta), order)
    return _tree_jet_one_product(W, t, leaves, b0, x0, beta, order)


def extract_tree_jet(
    params: TabParams,
    X: np.ndarray,
    *,
    beta: float,
    max_order: int,
    output_index: int = 0,
) -> TreeJet:
    Xt = jnp.asarray(np.asarray(X, dtype=np.float64))
    W = jnp.asarray(params.W)
    t = jnp.asarray(params.t)
    leaves = jnp.asarray(params.leaves)
    b0 = jnp.asarray(params.b0)
    order = int(max_order)
    rows = []
    for i in range(int(Xt.shape[0])):
        jet = _tree_jet_one(W, t, leaves, b0, Xt[i], float(beta), order)
        rows.append(jet[:, int(output_index)] if jet.ndim == 2 else jet)
    stacked = jnp.stack(rows, axis=1)
    dim = int(Xt.shape[1])
    raw = jet_partials(stacked, dim, order)
    partials = {alpha: np.asarray(val) for alpha, val in raw.items()}
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
    params = _TP(
        cfg,
        Ww.reshape(1, *Ww.shape),
        tt.reshape(1, -1),
        cell.reshape(1, *cell.shape),
        np.zeros(cell.shape[-1]),
    )
    return extract_tree_jet(
        params, X, beta=beta, max_order=max_order, output_index=output_index
    )


def _tree_jet_dir_one_product(
    W: Any,
    t: Any,
    leaves: Any,
    b0: Any,
    x0: Any,
    v: Any,
    beta: float,
    order: int,
) -> Any:
    x_jet = _path_jet(x0, v, order)
    k = int(leaves.shape[-1])
    out = jnp.zeros((order + 1, k), dtype=x0.dtype)
    out = out.at[0].set(b0)
    n_trees, depth = int(W.shape[0]), int(W.shape[1])
    codes = np.asarray(leaf_code_matrix(depth))
    n_leaves = int(codes.shape[0])
    for tree in range(n_trees):
        gates = [
            _gate_jet_dir(x_jet, W[tree, j], t[tree, j], beta, order)
            for j in range(depth)
        ]
        for ell in range(n_leaves):
            mu = _leaf_mu_dir(gates, codes[ell])
            out = out + mu[:, None] * leaves[tree, ell]
    return out


def _tree_jet_dir_one(
    W: Any,
    t: Any,
    leaves: Any,
    b0: Any,
    x0: Any,
    v: Any,
    beta: float,
    order: int,
) -> Any:
    if int(W.shape[1]) == 1:
        return mlp_jet(x0, v, _additive_mlp_layers(W, t, leaves, b0, beta), order)
    return _tree_jet_dir_one_product(W, t, leaves, b0, x0, v, beta, order)


def extract_tree_jet_directional(
    params: TabParams,
    X: np.ndarray,
    v: np.ndarray,
    *,
    beta: float,
    max_order: int,
    output_index: int = 0,
) -> np.ndarray:
    Xt = jnp.asarray(np.asarray(X, dtype=np.float64))
    vt = jnp.asarray(np.asarray(v, dtype=np.float64))
    if vt.ndim == 1:
        vt = jnp.broadcast_to(vt, Xt.shape)
    W = jnp.asarray(params.W)
    t = jnp.asarray(params.t)
    leaves = jnp.asarray(params.leaves)
    b0 = jnp.asarray(params.b0)
    order = int(max_order)
    rows = []
    for i in range(int(Xt.shape[0])):
        out = _tree_jet_dir_one(W, t, leaves, b0, Xt[i], vt[i], float(beta), order)
        col = out[:, int(output_index)] if out.ndim == 2 else out
        rows.append(jet_to_tower(col))
    return np.asarray(jnp.stack(rows, axis=1))


__all__ = [
    "extract_arrangement_jet",
    "extract_tree_jet",
    "extract_tree_jet_directional",
]

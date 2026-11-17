# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Residual losses of separable networks on tensor-product grids, without visiting the grid points.

A separable network ``u(x) = sum_r prod_i phi_i(x_i)_r`` has one network
``phi_i: R -> R^R`` per input axis, of any activation and depth.  Each term
``coeff * prod_i a_i(x_i) d^(k_i) / dx_i^(k_i)`` of a linear operator maps it to a
rank-``R`` canonical-polyadic (CP) tensor on a grid ``X_1 x ... x X_D``, so the
residual ``L u - f`` with a separable target ``f(x) = sum_q prod_i F_i[q, x_i]`` is a
CP tensor of rank ``J = T R + Q`` with factor matrices ``A_i`` of shape ``(N_i, J)``.
With product weights ``q(x) = prod_i q_i(x_i)``

    sum_x q(x) (L u(x) - f(x))^2 = sum_(j, j') prod_i (A_i^T diag(q_i) A_i)[j, j'],

the CP-norm identity of CP alternating least squares.  The cost is
``O(D N J^2)`` time and ``O(D J^2)`` memory instead of ``O(N^D J)``.  The axis
networks enter only through their derivative tables on the axis nodes, which
:func:`axis_derivative_tables` computes by Taylor mode for any network.  For
``tanh`` and ``sin`` MLPs with parameters stacked over the axes,
:func:`mlp_axis_tables` computes the tables of every axis together: one batched
product per layer over all Taylor orders and the activation's exact series
recurrence.  All axes are evaluated together in the loss: shorter axes are
padded to the longest one with zero weight (exact).
:func:`separable_grid_gauss_newton` returns the Gauss-Newton product
``v -> J^T Q J v`` of the same loss from mixed Gram matrices of the residual and
its Jacobian-vector product, also without visiting the grid points.

Scope and limits.  The operator must be linear with constant or separable
coefficients, the target separable (or zero), and the points a tensor grid.
The sum subtracts terms as large as ``S^2`` with
``S = sum_j prod_i ||sqrt(|q_i|) A_i[:, j]||``, so its absolute rounding error is
about ``eps S^2`` whatever the loss value: relative accuracy degrades like
``eps S^2 / loss`` near a fit, the square of the explicit residual's.  Nothing
here is a convergence statement for any PDE.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import jet as jet_module

__all__ = [
    "SeparableTerm",
    "axis_derivative_tables",
    "mlp_axis_tables",
    "separable_grid_gauss_newton",
    "separable_grid_loss",
]

_jet: Callable[..., Any] = jet_module.jet


class SeparableTerm(NamedTuple):
    """One term ``coeff * prod_i a_i(x_i) d^(orders[i]) / dx_i^(orders[i])`` of a separable operator.

    ``factors`` is ``None`` (constant coefficient) or one row index per axis into the
    ``coefficients`` tables of :func:`separable_grid_loss`.  Terms are static
    configuration, not traced arguments.
    """

    coeff: float
    orders: tuple[int, ...]
    factors: tuple[int, ...] | None = None


def axis_derivative_tables(fn: Callable[[Array], Array], x: Array, order: int) -> Array:
    """``(order + 1, N, R)`` derivatives ``d^k fn / dx^k`` (``k = 0..order``) on the nodes ``x`` (``(N,)``).

    ``fn`` maps the ``(N,)`` nodes to ``(N, R)`` features row by row (row ``n``
    depends on ``x[n]`` only), as a per-axis network applied to a batch does.  One
    Taylor-mode pass then yields every derivative row.
    """
    if int(order) != order or order < 0:
        raise ValueError("order must be a non-negative integer")
    x = jnp.asarray(x)
    if x.ndim != 1 or x.shape[0] == 0:
        raise ValueError("nodes must be a non-empty one-dimensional array")
    if order == 0:
        p = fn(x)
        series: list[Array] = []
    else:
        p, series = _jet(fn, (x,), ([jnp.ones_like(x)] + [jnp.zeros_like(x)] * (int(order) - 1),))
    if p.ndim != 2 or p.shape[0] != x.shape[0]:
        raise ValueError("fn must map (N,) nodes to (N, R) features")
    return jnp.stack([p, *series])


def _derivative_product(z: list[Array], s: list[Array], k: int) -> Array:
    """``(1/k) sum_(j=1..k) j z_j s_(k-j)``: the order-``k`` coefficient of ``integral(z' s)``."""
    acc = z[1] * s[k - 1]
    for j in range(2, k + 1):
        acc = acc + j * z[j] * s[k - j]
    return acc / k


def _square(y: list[Array], k: int) -> Array:
    """Order-``k`` coefficient of ``y^2``."""
    acc = y[k // 2] * y[k // 2] if k % 2 == 0 else jnp.zeros_like(y[0])
    for i in range((k + 1) // 2):
        acc = acc + 2.0 * y[i] * y[k - i]
    return acc


def _tanh_series(z: list[Array]) -> list[Array]:
    """Normalized Taylor coefficients of ``tanh`` of the series ``z``, from ``y' = (1 - y^2) z'``."""
    y = [jnp.tanh(z[0])]
    sq = [y[0] * y[0]]
    for k in range(1, len(z)):
        y.append(z[k] - _derivative_product(z, sq, k))
        sq.append(_square(y, k))
    return y


def _sin_series(z: list[Array]) -> list[Array]:
    """Normalized Taylor coefficients of ``sin`` of the series ``z``, from ``s' = c z'`` and ``c' = -s z'``."""
    s, c = [jnp.sin(z[0])], [jnp.cos(z[0])]
    for k in range(1, len(z)):
        s.append(_derivative_product(z, c, k))
        c.append(-_derivative_product(z, s, k))
    return s


_SERIES: dict[str, Callable[[list[Array]], list[Array]]] = {
    "tanh": _tanh_series,
    "sin": _sin_series,
}


def mlp_axis_tables(
    layers: Sequence[tuple[Array, Array]],
    nodes: Array,
    order: int,
    *,
    activation: str = "tanh",
) -> Array:
    """``(D, order + 1, N, R)`` derivative tables of ``D`` scalar-input MLPs, one per axis.

    ``layers[l] = (W, b)`` with ``W`` of shape ``(D, n_l, n_(l+1))`` and ``b`` of
    shape ``(D, n_(l+1))``: axis ``i`` evaluates
    ``phi_i(x) = W_L[i]^T act(... act(W_1[i]^T x + b_1[i]) ...) + b_L[i]`` with
    ``n_1 = 1``, ``activation`` (``"tanh"`` or ``"sin"``) on every hidden layer
    and an affine last layer.  ``nodes`` is ``(D, N)``; pad shorter axes and give
    the padding zero weight in :func:`separable_grid_loss`.  Entry ``[i, k]``
    equals ``axis_derivative_tables(phi_i, nodes[i], order)[k]``; the truncated
    series of all orders share one batched product per layer.
    """
    if int(order) != order or order < 0:
        raise ValueError("order must be a non-negative integer")
    if activation not in _SERIES:
        raise ValueError(f"activation must be one of {sorted(_SERIES)}")
    x = jnp.asarray(nodes)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("nodes must have shape (D, N) with N >= 1")
    params = [(jnp.asarray(W), jnp.asarray(b)) for W, b in layers]
    if not params:
        raise ValueError("at least one layer is required")
    dim, width = x.shape[0], 1
    for W, b in params:
        if W.ndim != 3 or W.shape[:2] != (dim, width) or b.shape != (dim, W.shape[2]):
            raise ValueError(
                f"layer weights must have shape ({dim}, {width}, n) with biases ({dim}, n)"
            )
        width = W.shape[2]
    K = int(order)
    zeros = jnp.zeros_like(x)
    series = [x, jnp.ones_like(x), *[zeros] * (K - 1)][: K + 1]
    h = jnp.stack(series, axis=1)[..., None]
    act = _SERIES[activation]
    for i, (W, b) in enumerate(params):
        z = jnp.einsum("dkni,dio->dkno", h, W)
        if i < len(params) - 1:
            # the bias joins the order-zero coefficient on its way into the series: the stack is not rewritten to hold it
            h = jnp.stack(act([z[:, 0] + b[:, None, :], *[z[:, k] for k in range(1, K + 1)]]), axis=1)
        else:
            h = z.at[:, 0].add(b[:, None, :])
    scale = jnp.asarray([math.factorial(k) for k in range(K + 1)], dtype=h.dtype)
    return h * scale[None, :, None, None]


def _validate(
    tables: Sequence[Array],
    terms: Sequence[SeparableTerm],
    coefficients: Sequence[Array] | None,
) -> tuple[list[Array], tuple[SeparableTerm, ...]]:
    tabs = [jnp.asarray(t) for t in tables]
    if not tabs:
        raise ValueError("at least one axis table is required")
    if any(t.ndim != 3 or t.shape[1] == 0 for t in tabs):
        raise ValueError("each table must have shape (K_i + 1, N_i, R) with N_i >= 1")
    if len({t.shape[2] for t in tabs}) != 1:
        raise ValueError("every axis table must have the same rank R")
    if not terms:
        raise ValueError("at least one operator term is required")
    dim = len(tabs)
    out = []
    for term in terms:
        coeff, orders, factors = term
        orders = tuple(int(k) for k in orders)
        if len(orders) != dim:
            raise ValueError(f"term orders must have one entry per axis ({dim})")
        for i, k in enumerate(orders):
            if k < 0:
                raise ValueError("derivative orders must be non-negative")
            if k >= tabs[i].shape[0]:
                raise ValueError(
                    f"axis {i} table holds orders up to {tabs[i].shape[0] - 1}, term needs {k}"
                )
        if factors is not None:
            if coefficients is None:
                raise ValueError("a term with coefficient factors needs the coefficients tables")
            factors = tuple(int(u) for u in factors)
            if len(factors) != dim:
                raise ValueError(f"term factors must have one entry per axis ({dim})")
            if any(not 0 <= u < coefficients[i].shape[0] for i, u in enumerate(factors)):
                raise ValueError("coefficient factor index out of range")
        out.append(SeparableTerm(float(coeff), orders, factors))
    return tabs, tuple(out)


def _axis_arrays(
    arrays: Sequence[Array] | None, name: str, tabs: list[Array], rows: bool, dtype: jnp.dtype
) -> list[Array] | None:
    if arrays is None:
        return None
    if len(arrays) != len(tabs):
        raise ValueError(f"expected {len(tabs)} {name} arrays (one per axis), got {len(arrays)}")
    out = [jnp.asarray(a, dtype=dtype) for a in arrays]
    for a, t in zip(out, tabs, strict=True):
        if (a.ndim != 2 or a.shape[1] != t.shape[1]) if rows else a.shape != t.shape[1:2]:
            raise ValueError(f"{name} arrays must have shape {'(M, N_i)' if rows else '(N_i,)'}")
    return out


def separable_grid_loss(
    tables: Sequence[Array],
    terms: Sequence[SeparableTerm],
    *,
    weights: Sequence[Array] | None = None,
    target: Sequence[Array] | None = None,
    coefficients: Sequence[Array] | None = None,
) -> Array:
    """``0.5 * sum_x q(x) (sum_t L_t u(x) - f(x))^2`` over the grid of the table nodes.

    ``tables[i]`` is ``(K_i + 1, N_i, R)``: the axis-``i`` features and their
    derivatives of orders ``1..K_i`` on its ``N_i`` nodes (see
    :func:`axis_derivative_tables`).  ``weights[i]`` is ``(N_i,)``; the default
    ``1 / N_i`` gives the mean over grid points.  ``target`` is ``None`` (zero) or
    one ``(Q, N_i)`` array per axis with ``f(x) = sum_q prod_i target[i][q, x_i]``.
    ``coefficients[i]`` is ``(U_i, N_i)``; a term with ``factors`` is multiplied by
    ``prod_i coefficients[i][factors[i], x_i]``.  Differentiable in every array
    argument by ordinary autodiff; the grid itself is never materialized.
    """
    factors, qs = _residual_factors(tables, terms, weights, target, coefficients)
    n = max(a.shape[0] for a in factors)
    A = jnp.stack([jnp.pad(a, ((0, n - a.shape[0]), (0, 0))) for a in factors])
    q = jnp.stack([jnp.pad(w, (0, n - w.shape[0])) for w in qs])
    G = jnp.einsum("inj,in,ink->ijk", A, q, A)
    return 0.5 * jnp.sum(jnp.prod(G, axis=0))


def _residual_factors(
    tables: Sequence[Array],
    terms: Sequence[SeparableTerm],
    weights: Sequence[Array] | None,
    target: Sequence[Array] | None,
    coefficients: Sequence[Array] | None,
) -> tuple[list[Array], list[Array]]:
    """Unweighted ``(N_i, J)`` CP factors of the residual and the ``(N_i,)`` axis weights."""
    tabs, terms_t = _validate(tables, terms, coefficients)
    dtype = jnp.result_type(*tabs)
    tabs = [t.astype(dtype) for t in tabs]
    qs = _axis_arrays(weights, "weight", tabs, False, dtype)
    fs = _axis_arrays(target, "target", tabs, True, dtype)
    cs = _axis_arrays(coefficients, "coefficient", tabs, True, dtype)
    if fs is not None and len({f.shape[0] for f in fs}) != 1:
        raise ValueError("target factors must have the same number of terms Q on every axis")
    if qs is None:
        qs = [jnp.full(t.shape[1:2], 1.0 / t.shape[1], dtype=dtype) for t in tabs]
    factors = []
    for i, tab in enumerate(tabs):
        cols = []
        for term in terms_t:
            col = tab[term.orders[i]]
            if term.factors is not None:
                assert cs is not None
                col = cs[i][term.factors[i]][:, None] * col
            cols.append(term.coeff * col if i == 0 else col)
        if fs is not None:
            cols.append(-fs[i].T if i == 0 else fs[i].T)
        factors.append(jnp.concatenate(cols, axis=1))
    return factors, qs


def _hadamard(mats: Sequence[Array], like: Array) -> Array:
    out = jnp.ones_like(like)
    for m in mats:
        out = out * m
    return out


def separable_grid_gauss_newton(
    tables_fn: Callable[[Any], Sequence[Array]],
    params: Any,
    terms: Sequence[SeparableTerm],
    *,
    weights: Sequence[Array] | None = None,
    target: Sequence[Array] | None = None,
    coefficients: Sequence[Array] | None = None,
) -> tuple[Array, Any, Callable[[Any], Any]]:
    """Loss, gradient and Gauss-Newton product ``v -> J^T Q J v`` of :func:`separable_grid_loss`.

    ``tables_fn(params)`` returns the axis tables (for example
    ``lambda p: mlp_axis_tables(p, nodes, order)``); ``J`` is the Jacobian of the
    grid residual with respect to ``params`` and ``Q`` the diagonal of grid weights.
    The residual is a rank-``J`` CP tensor and ``J v`` a rank-``D J`` one whose
    axis-``a`` factor is ``A_a`` except ``Adot_a = (dA_a/dparams) v`` in block ``a``,
    so ``J^T Q J v = grad <r, stop_gradient(J v)>_Q`` needs only the ``J x J`` Gram
    matrices ``A_b^T diag(q_b) A_b`` (shared by every product) and
    ``A_d^T diag(q_d) Adot_d``.  ``tables_fn`` is linearized once; each product is one
    linearized and one transposed table map plus ``O(D N J^2 + D^3 J^2)`` work
    instead of ``O(N^D J)``.  The returned product is a linear function of a pytree
    shaped like ``params`` and can be used inside ``jax.jit`` with iterative solvers.
    The loss keeps the rounding limit stated in the module notes; the product does
    not involve the target.
    """

    def factor_map(p: Any) -> tuple[list[Array], list[Array]]:
        return _residual_factors(tables_fn(p), terms, weights, target, coefficients)

    A, lin, qs = jax.linearize(factor_map, params, has_aux=True)
    lin_t = jax.linear_transpose(lin, params)
    dim = len(A)
    Aq = [q[:, None] * a for q, a in zip(qs, A, strict=True)]
    gram = [aq.T @ a for aq, a in zip(Aq, A, strict=True)]
    rest = [_hadamard([gram[b] for b in range(dim) if b != a], gram[0]) for a in range(dim)]
    pair = {
        (a, d): _hadamard([gram[b] for b in range(dim) if b not in (a, d)], gram[0])
        for a in range(dim)
        for d in range(dim)
        if d != a
    }
    loss = 0.5 * jnp.sum(gram[0] * rest[0])
    (grad,) = lin_t([aq @ m for aq, m in zip(Aq, rest, strict=True)])

    def gnvp(v: Any) -> Any:
        Ad = lin(v)
        mixed = [aq.T @ ad for aq, ad in zip(Aq, Ad, strict=True)]
        cot = []
        for a in range(dim):
            s = jnp.zeros_like(gram[0])
            for d in range(dim):
                if d != a:
                    s = s + mixed[d] * pair[(a, d)]
            cot.append(qs[a][:, None] * (Ad[a] @ rest[a] + A[a] @ s.T))
        (out,) = lin_t(cot)
        return out

    return loss, grad, gnvp

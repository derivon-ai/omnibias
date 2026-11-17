# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Residual losses of one-layer ridge operators on tensor-product grids, without visiting the grid points.

For ``sin``, ``cos`` and ``exp`` every feature is an exponential of the ridge
variable: ``e^(lam (w.x + b)) = e^(lam b) prod_i e^(lam w_i x_i)`` with
``lam = i`` (``sin``, ``cos``) or ``lam = 1`` (``exp``).  A constant-coefficient
operator multiplies the feature by its symbol ``p(lam w)``.  On a grid
``X_1 x ... x X_D`` with product weights ``q(x) = prod_i q_i(x_i)`` the weighted
sum of ``(L u - f)^2`` is therefore an exact ``H x H`` quadratic form whose
kernel is the Hadamard product of ``D`` one-dimensional Gram matrices
``E_i diag(q_i) E_i^*`` with ``E_i[h, j] = e^(lam w_(h,i) x_(i,j))``.  The cost is
``O(D H^2 N)`` time and ``O(D H^2)`` memory instead of ``O(N^D H)``.  All axes are
evaluated together: shorter axes are padded to the longest one with zero weight
(exact), and for ``sin`` and ``cos`` both one-dimensional Gram matrices of every
axis are blocks of one batched real ``2H x 2H`` product of the stacked cosine and
sine tables.

Separable variable coefficients ``a_t(x) = prod_i C_i[k_(t,i), x_i]`` are extra
per-axis weights of the same Gram matrices.  Terms with the same coefficient are
merged into one of ``K`` groups and every axis needs the Grams of its ``U_i``
distinct factors, ``O(U^2 D H^2 N + K^2 D H^2)`` in total.  A target given by its
values on every grid point enters only the cross term
``sum_x q(x) f(x) e^(lam w_h.x)``, contracted one axis at a time in about
``4 K H N^D`` flops (no transcendental per grid point).

Scope and limits.  The points must form a tensor grid, and coefficients must be
separable (expand others first).  The quadratic form subtracts
terms of size ``S^2`` with ``S = sum_h |c_h p(lam w_h)|``, so its absolute
rounding error is about ``eps S^2`` whatever the loss value: relative accuracy
degrades like ``eps S^2 / loss`` near a fit, the square of the per-point
residual's ``eps S / |r|``.  Nothing here is a convergence statement for any PDE.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.jax.experimental.closed_ridge import _validate, derivative_closed_system
from omnibias.jax.experimental.tanh_ridge import RidgeTerm

import jax.numpy as jnp
from jax import Array

__all__ = ["closed_ridge_grid_loss"]

_LAMBDA = {"sin": 1j, "cos": 1j, "exp": 1.0}


def _symbol(W: Array, terms: tuple[RidgeTerm, ...], lam: complex | float) -> Array:
    """``p(lam w_h) = sum_t coeff_t (lam^2 |w_h|^2)^k`` or ``coeff_t (lam w_(h,i))^n``."""
    base = W * lam
    out = jnp.zeros(W.shape[:1], dtype=base.dtype)
    for term in terms:
        if term.kind == "laplace":
            out = out + term.coeff * jnp.sum(base * base, axis=1) ** term.order
        else:
            out = out + term.coeff * base[:, term.axis] ** term.order
    return out


def _grid_terms(terms: Sequence[RidgeTerm], dim: int) -> tuple[RidgeTerm, ...]:
    """Validated terms; order 0 (either kind) is the identity, with symbol 1."""
    if not terms:
        raise ValueError("at least one operator term is required")
    out: list[RidgeTerm] = []
    for term in terms:
        if term.order == 0 and term.kind in ("laplace", "coordinate"):
            out.append(RidgeTerm("laplace", 0, float(term.coeff)))
        else:
            out.extend(_validate([term], dim))
    return tuple(out)


def _axes(
    nodes: Sequence[Array],
    weights: Sequence[Array] | None,
    target: Sequence[Array] | None,
    dim: int,
    dtype: jnp.dtype,
) -> tuple[list[Array], list[Array], list[Array] | None]:
    if len(nodes) != dim:
        raise ValueError(f"expected {dim} node arrays (one per input dimension), got {len(nodes)}")
    xs = [jnp.asarray(n, dtype=dtype) for n in nodes]
    if any(x.ndim != 1 or x.shape[0] == 0 for x in xs):
        raise ValueError("each node array must be one-dimensional and non-empty")
    if weights is None:
        qs = [jnp.full(x.shape, 1.0 / x.shape[0], dtype=dtype) for x in xs]
    else:
        if len(weights) != dim:
            raise ValueError(f"expected {dim} weight arrays, got {len(weights)}")
        qs = [jnp.asarray(w, dtype=dtype) for w in weights]
        if any(q.shape != x.shape for q, x in zip(qs, xs, strict=True)):
            raise ValueError("each weight array must match its node array")
    if target is None:
        return xs, qs, None
    if len(target) != dim:
        raise ValueError(f"expected {dim} target factor arrays, got {len(target)}")
    fs = [jnp.asarray(t, dtype=dtype) for t in target]
    n_terms = fs[0].shape[0] if fs[0].ndim == 2 else -1
    if any(f.ndim != 2 or f.shape != (n_terms, x.shape[0]) for f, x in zip(fs, xs, strict=True)):
        raise ValueError("target factors must have shape (Q, N_i) with the same Q on every axis")
    return xs, qs, fs


def _stack(arrays: Sequence[Array], n: int) -> Array:
    """Stack along a new leading axis, zero-padding the last axis to length ``n``."""
    return jnp.stack([jnp.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])]) for a in arrays])


def _coefficient_groups(
    terms: tuple[RidgeTerm, ...],
    coefficients: Sequence[Array],
    term_factors: Sequence[Sequence[int]],
    xs: list[Array],
    dtype: jnp.dtype,
) -> tuple[list[Array], list[tuple[RidgeTerm, ...]], tuple[tuple[int, ...], ...]]:
    """Validated factor tables, terms grouped by coefficient and the static ``(K, D)`` factor index map."""
    dim = len(xs)
    if len(coefficients) != dim:
        raise ValueError(f"expected {dim} coefficient factor arrays, got {len(coefficients)}")
    cs = [jnp.asarray(cf, dtype=dtype) for cf in coefficients]
    if any(
        cf.ndim != 2 or cf.shape[0] == 0 or cf.shape[1] != x.shape[0]
        for cf, x in zip(cs, xs, strict=True)
    ):
        raise ValueError("coefficient factors must have shape (U_i, N_i) with U_i >= 1")
    if len(term_factors) != len(terms):
        raise ValueError(
            f"expected one factor tuple per term ({len(terms)}), got {len(term_factors)}"
        )
    keys: dict[tuple[int, ...], list[RidgeTerm]] = {}
    for term, factors in zip(terms, term_factors, strict=True):
        key = tuple(int(k) for k in factors)
        if len(key) != dim:
            raise ValueError(f"each term needs {dim} factor indices, got {len(key)}")
        if any(not 0 <= k < cf.shape[0] for k, cf in zip(key, cs, strict=True)):
            raise ValueError(
                f"factor indices {key} out of range for {[cf.shape[0] for cf in cs]} factors"
            )
        keys.setdefault(key, []).append(term)
    return cs, [tuple(g) for g in keys.values()], tuple(keys)


def _pair_product(M: Array, idx: tuple[tuple[int, ...], ...]) -> Array:
    """``prod_i M[i, idx[:, i], :, idx[:, i], :]`` for per-axis blocks ``M`` of shape ``(D, U, H, U, H)``."""
    out = None
    for i in range(M.shape[0]):
        col = jnp.asarray([row[i] for row in idx])
        block = M[i][col][:, :, col]
        out = block if out is None else out * block
    assert out is not None
    return out


def _target_transform(
    W: Array, xs: list[Array], ws: list[Array], F: Array, lam: complex | float
) -> Array:
    """``T_h = sum_x prod_i ws_i(x_i) F[x] e^(lam w_h.x)``: one GEMM for axis 0, per-unit contractions after."""
    H, dim = W.shape
    Fq = F
    for i, w in enumerate(ws):
        Fq = Fq * w.reshape([1] * i + [w.shape[0]] + [1] * (dim - i - 1))
    t0 = W[:, :1] * xs[0][None, :]
    rest = F.shape[1:]
    if lam == 1.0:
        Y = (jnp.exp(t0) @ Fq.reshape(xs[0].shape[0], -1)).reshape((H, *rest))
    else:
        Y2 = (jnp.concatenate([jnp.cos(t0), jnp.sin(t0)]) @ Fq.reshape(xs[0].shape[0], -1)).reshape(
            (2 * H, *rest)
        )
        Y = Y2[:H] + 1j * Y2[H:]
    for i in range(1, dim):
        Y = jnp.einsum("hn,hn...->h...", jnp.exp(lam * W[:, i : i + 1] * xs[i][None, :]), Y)
    return Y


def _target_values(target_values: Array, xs: list[Array], dtype: jnp.dtype) -> Array:
    F = jnp.asarray(target_values, dtype=dtype)
    shape = tuple(x.shape[0] for x in xs)
    if F.shape != shape:
        raise ValueError(f"target_values must have the grid shape {shape}, got {F.shape}")
    return F


def _values_cross(
    W: Array,
    amps: Array,
    xs: list[Array],
    group_weights: list[list[Array]],
    F: Array,
    lam: complex | float,
    activation: str,
) -> Array:
    """``sum_x q f L u`` for group amplitudes ``amps`` ``(K, H)`` and per-group axis weights."""
    cross = jnp.zeros((), dtype=amps.dtype)
    for amp, ws in zip(amps, group_weights, strict=True):
        cross = cross + amp @ _target_transform(W, xs, ws, F, lam)
    return cross if activation == "exp" else jnp.imag(cross)


def _weighted_sum_sq(F: Array, qs: list[Array]) -> Array:
    out = F * F
    for i, w in enumerate(qs):
        out = out * w.reshape([1] * i + [w.shape[0]] + [1] * (len(qs) - i - 1))
    return jnp.sum(out)


def _varcoef_loss(
    W: Array,
    b: Array,
    c: Array,
    xs: list[Array],
    qs: list[Array],
    fs: list[Array] | None,
    F: Array | None,
    cs: list[Array],
    groups: list[tuple[RidgeTerm, ...]],
    idx: tuple[tuple[int, ...], ...],
    activation: str,
) -> Array:
    lam = _LAMBDA[activation]
    H, dim = W.shape
    n = max(x.shape[0] for x in xs)
    U = max(cf.shape[0] for cf in cs)
    q = _stack(qs, n)
    C = jnp.stack([jnp.pad(cf, [(0, U - cf.shape[0]), (0, n - cf.shape[1])]) for cf in cs])
    t = W.T[:, :, None] * _stack(xs, n)[:, None, :]
    B = jnp.exp(t) if activation == "exp" else jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=1)
    R = B.shape[1]
    Ba = (C[:, :, None, :] * B[:, None, :, :]).reshape(dim, U * R, n)
    Baq = Ba * q[:, None, :]
    G = jnp.einsum("dan,dbn->dab", Baq, Ba).reshape(dim, U, R, U, R)
    if activation == "exp":
        amps = jnp.stack([c * _symbol(W, g, lam) * jnp.exp(b) for g in groups])
        uu = jnp.einsum("kh,khlg,lg->", amps, _pair_product(G, idx), amps)
    else:
        kappa = 1.0 if activation == "sin" else 1j
        amps = jnp.stack([kappa * c * _symbol(W, g, lam) * jnp.exp(1j * b) for g in groups])
        CC, CSt, SC, SS = (
            G[..., :H, :, :H],
            G[..., :H, :, H:],
            G[..., H:, :, :H],
            G[..., H:, :, H:],
        )
        Mm = _pair_product((CC + SS) + 1j * (SC - CSt), idx)
        Mp = _pair_product((CC - SS) + 1j * (SC + CSt), idx)
        uu = 0.5 * jnp.real(jnp.einsum("kh,khlg,lg->", amps, Mm, jnp.conj(amps))) - 0.5 * jnp.real(
            jnp.einsum("kh,khlg,lg->", amps, Mp, amps)
        )
    if F is not None:
        group_weights = [[qs[i] * cs[i][k] for i, k in enumerate(row)] for row in idx]
        cross = _values_cross(W, amps, xs, group_weights, F, lam, activation)
        return 0.5 * (uu - 2.0 * cross + _weighted_sum_sq(F, qs))
    if fs is None:
        return 0.5 * uu
    Ft = _stack(fs, n)
    P = jnp.einsum("dan,dqn->daq", Baq, Ft).reshape(dim, U, R, -1)
    if activation != "exp":
        P = P[:, :, :H] + 1j * P[:, :, H:]
    proj = P[0][jnp.asarray([row[0] for row in idx])]
    for i in range(1, dim):
        proj = proj * P[i][jnp.asarray([row[i] for row in idx])]
    cross = jnp.einsum("kh,khq->", amps, proj)
    cross = cross if activation == "exp" else jnp.imag(cross)
    ff = jnp.prod(jnp.einsum("dqn,dpn->dqp", Ft * q[:, None, :], Ft), axis=0)
    return 0.5 * (uu - 2.0 * cross + jnp.sum(ff))


def closed_ridge_grid_loss(
    W: Array,
    b: Array,
    c: Array,
    nodes: Sequence[Array],
    terms: Sequence[RidgeTerm],
    *,
    weights: Sequence[Array] | None = None,
    target: Sequence[Array] | None = None,
    target_values: Array | None = None,
    coefficients: Sequence[Array] | None = None,
    term_factors: Sequence[Sequence[int]] | None = None,
    activation: str = "sin",
) -> Array:
    """``0.5 * sum_x q(x) (sum_t a_t(x) coeff_t L_t u(x) - f(x))^2`` over the grid ``nodes[0] x ... x nodes[D-1]``.

    ``W`` is ``(H, D)`` and ``b``, ``c`` are ``(H,)`` for ``u(x) = sum_h c_h act(w_h . x + b_h)``.
    ``nodes[i]`` and ``weights[i]`` are ``(N_i,)``; the default weights ``1 / N_i`` give the mean over
    grid points.  The target is zero by default; ``target`` gives it in separable form, one ``(Q, N_i)``
    array per axis with ``f(x) = sum_q prod_i target[i][q, x_i]``, and ``target_values`` by its values
    on the grid, shape ``(N_0, ..., N_(D-1))``.  ``coefficients`` (one ``(U_i, N_i)`` array of factor
    values per axis) with ``term_factors`` (one tuple of ``D`` row indices per term) gives term ``t``
    the coefficient ``a_t(x) = prod_i coefficients[i][term_factors[t][i], x_i]``; without them
    ``a_t = 1``.  A term of order 0 is the identity.  Differentiable in ``W``, ``b`` and ``c`` by ordinary autodiff; the grid itself is
    never materialized.
    """
    if activation not in _LAMBDA:
        derivative_closed_system(activation)
        raise ValueError(f"{activation!r} is not an exponential-family activation (sin, cos, exp)")
    terms_t = _grid_terms(terms, W.shape[1])
    if target is not None and target_values is not None:
        raise ValueError(
            "give the target either as separable factors (target) or as grid values (target_values)"
        )
    if (coefficients is None) != (term_factors is None):
        raise ValueError("coefficients and term_factors must be given together")
    dtype = jnp.result_type(W, b, c)
    xs, qs, fs = _axes(nodes, weights, target, W.shape[1], dtype)
    F = None if target_values is None else _target_values(target_values, xs, dtype)
    if coefficients is not None and term_factors is not None:
        cs, groups, idx = _coefficient_groups(terms_t, coefficients, term_factors, xs, dtype)
        return _varcoef_loss(W, b, c, xs, qs, fs, F, cs, groups, idx, activation)
    lam = _LAMBDA[activation]
    H = W.shape[0]
    n = max(x.shape[0] for x in xs)
    q = _stack(qs, n)
    t = W.T[:, :, None] * _stack(xs, n)[:, None, :]
    if activation == "exp":
        amp = c * _symbol(W, terms_t, lam) * jnp.exp(b)
        E = jnp.exp(t)
        Eq = E * q[:, None, :]
        uu = amp @ (jnp.prod(jnp.einsum("dhn,dkn->dhk", Eq, E), axis=0) @ amp)
        if F is not None:
            cross = _values_cross(W, amp[None], xs, [qs], F, lam, activation)
            return 0.5 * (uu - 2.0 * cross + _weighted_sum_sq(F, qs))
        if fs is None:
            return 0.5 * uu
        Ft = _stack(fs, n)
        proj = jnp.prod(jnp.einsum("dhn,dqn->dhq", Eq, Ft), axis=0)
        ff = jnp.prod(jnp.einsum("dqn,dpn->dqp", Ft * q[:, None, :], Ft), axis=0)
        return 0.5 * (uu - 2.0 * (amp @ jnp.sum(proj, axis=1)) + jnp.sum(ff))
    amp = c * _symbol(W, terms_t, lam) * jnp.exp(1j * b)
    if activation == "cos":
        amp = 1j * amp
    CS = jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=1)
    CSq = CS * q[:, None, :]
    G = jnp.einsum("dhn,dkn->dhk", CSq, CS)
    CC, CSt, SC, SS = G[:, :H, :H], G[:, :H, H:], G[:, H:, :H], G[:, H:, H:]
    Mm = jnp.prod((CC + SS) + 1j * (SC - CSt), axis=0)
    Mp = jnp.prod((CC - SS) + 1j * (SC + CSt), axis=0)
    uu = 0.5 * jnp.real(amp @ (Mm @ jnp.conj(amp))) - 0.5 * jnp.real(amp @ (Mp @ amp))
    if F is not None:
        cross = _values_cross(W, amp[None], xs, [qs], F, lam, activation)
        return 0.5 * (uu - 2.0 * cross + _weighted_sum_sq(F, qs))
    if fs is None:
        return 0.5 * uu
    Ft = _stack(fs, n)
    P = jnp.einsum("dhn,dqn->dhq", CSq, Ft)
    proj = jnp.prod(P[:, :H] + 1j * P[:, H:], axis=0)
    ff = jnp.prod(jnp.einsum("dqn,dpn->dqp", Ft * q[:, None, :], Ft), axis=0)
    return 0.5 * (uu - 2.0 * jnp.imag(amp @ jnp.sum(proj, axis=1)) + jnp.sum(ff))

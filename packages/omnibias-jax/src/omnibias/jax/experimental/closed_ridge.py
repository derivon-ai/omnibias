# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Order-independent operator programs for ridge fields with derivative-closed activations.

An activation is *derivative-closed* when its registered differential system
(:mod:`omnibias.core.jet_systems`) is linear with constant coefficients:
``v' = C v`` and ``sigma = a . v``.  ``sin`` and ``cos`` (``v = (sin, cos)``)
and ``exp`` (``v = (exp,)``) qualify; ``tanh``, ``sigmoid`` and ``gaussian``
do not.  Then ``sigma^(m) = (a C^m) . v`` exactly, and every
constant-coefficient operator of the one-hidden-layer field
``u(x) = sum_h c_h sigma(w_h . x + b_h)`` collapses to ``r = len(v)`` feature
columns:

``sum_t coeff_t L_t u(x) = sum_h c_h sum_k A_(h,k) v_k(z_h)``,
``A_h = sum_t g_t(w_h) (a C^(m_t))``

with ``g_t = coeff_t |w_h|^(2k)`` for ``Delta^k`` and ``coeff_t w_(h,i)^n`` for
``d_i^n``.  The ``(B, H)`` work of :func:`closed_ridge_operator` and of the
reverse rule in :func:`closed_ridge_residual_loss` is therefore independent of
derivative order and number of terms; only ``O(H)`` metric work grows with
them.  For ``sin`` the rows ``a C^m`` are the phase cycle
``(sin, cos, -sin, -cos)[m mod 4]``.

Scope and limits.  This is exact algebra for one hidden layer of a
derivative-closed activation, not a deep-network or general activation method.
The rows ``a C^m`` are exact rationals; the floating-point program can still
overflow for large metrics (``|w|^(2k)``) or large ``exp`` arguments, and
cancellation in the residual is not removed.  Nothing here is a convergence
statement for any PDE.  See ``docs/performance.md`` for measured regimes.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache

from omnibias.core.jet_systems import get_system
from omnibias.jax.experimental.tanh_ridge import (
    RidgeTerm,
    _derivative_order,
    _metric,
    _readout,
)

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["closed_ridge_operator", "closed_ridge_residual_loss", "derivative_closed_system"]

MAX_DERIVATIVE_ORDER = 64
_STATE_FNS = {"sin": jnp.sin, "cos": jnp.cos, "exp": jnp.exp}
# The rematerialised reverse rule measured up to 3x slower with the unrolled projection at D = 8.
_UNROLL_MAX_D = 6


@lru_cache(maxsize=32)
def derivative_closed_system(
    activation: str,
) -> tuple[tuple[str, ...], tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    """``(states, C, a)`` with ``v' = C v`` and ``sigma = a . v``; ``ValueError`` if not derivative-closed."""
    try:
        spec = get_system(activation)
    except NotImplementedError as exc:
        raise ValueError(f"no differential system registered for {activation!r}") from exc
    r = len(spec.states)

    def linear_row(poly) -> tuple[Fraction, ...]:  # type: ignore[no-untyped-def]
        row = [Fraction(0)] * r
        for coeff, powers in poly:
            if powers[0] != 0 or sum(powers[1:]) != 1:
                raise ValueError(
                    f"activation {activation!r} is not derivative-closed: its differential system is not "
                    "linear with constant coefficients"
                )
            row[list(powers[1:]).index(1)] += coeff
        return tuple(row)

    C = tuple(linear_row(p) for p in spec.rhs)
    a = linear_row(spec.output)
    missing = [
        s
        for s, used in zip(spec.states, _used_states(C, a), strict=True)
        if used and s not in _STATE_FNS
    ]
    if missing:
        raise ValueError(f"no evaluator for states {missing} of {activation!r}")
    return spec.states, C, a


def _used_states(C, a) -> list[bool]:  # type: ignore[no-untyped-def]
    r = len(a)
    used = [x != 0 for x in a]
    for _ in range(r):
        used = [used[k] or any(used[j] and C[j][k] != 0 for j in range(r)) for k in range(r)]
    return used


def _rows(activation: str, n_max: int) -> list[tuple[Fraction, ...]]:
    """``[a C^0, a C^1, ..., a C^n_max]`` (exact)."""
    _, C, a = derivative_closed_system(activation)
    r = len(a)
    rows = [a]
    for _ in range(n_max):
        prev = rows[-1]
        rows.append(
            tuple(sum((prev[j] * C[j][k] for j in range(r)), Fraction(0)) for k in range(r))
        )
    return rows


def _validate(terms: Sequence[RidgeTerm], dim: int) -> tuple[RidgeTerm, ...]:
    if not terms:
        raise ValueError("at least one operator term is required")
    out = []
    for term in terms:
        kind, order, coeff, axis = term
        if kind not in ("laplace", "coordinate"):
            raise ValueError(f"unknown term kind {kind!r}")
        if int(order) != order or order < 1:
            raise ValueError("term order must be a positive integer")
        t = RidgeTerm(kind, int(order), float(coeff), int(axis))
        if _derivative_order(t) > MAX_DERIVATIVE_ORDER:
            raise ValueError(
                f"derivative order {_derivative_order(t)} exceeds the supported maximum {MAX_DERIVATIVE_ORDER}"
            )
        if kind == "coordinate" and not 0 <= axis < dim:
            raise ValueError("coordinate axis out of range")
        out.append(t)
    return tuple(out)


def _amplitudes(
    W: Array, terms: tuple[RidgeTerm, ...], rows: list[tuple[Fraction, ...]], shift: int
) -> dict[int, Array]:
    """``{k: sum_t g_t(W) rows[m_t + shift][k]}`` over the state columns that are not identically zero."""
    out: dict[int, Array] = {}
    gs = [_metric(t, W) for t in terms]
    for k in range(len(rows[0])):
        acc: Array | None = None
        for t, g in zip(terms, gs, strict=True):
            w = rows[_derivative_order(t) + shift][k]
            if w:
                term = float(w) * g
                acc = term if acc is None else acc + term
        if acc is not None:
            out[k] = acc
    return out


def _project(x: Array, W: Array, b: Array) -> Array:
    if W.shape[1] > _UNROLL_MAX_D:
        return x @ W.T + b
    z = b[None, :] + x[:, 0:1] * W[None, :, 0]
    for d in range(1, W.shape[1]):
        z = z + x[:, d : d + 1] * W[None, :, d]
    return z


def _states(z: Array, names: tuple[str, ...], needed: set[int]) -> dict[int, Array]:
    return {k: _STATE_FNS[names[k]](z) for k in sorted(needed)}


def _psi(A: dict[int, Array], v: dict[int, Array]) -> Array:
    out: Array | None = None
    for k, amp in A.items():
        term = amp[None, :] * v[k]
        out = term if out is None else out + term
    assert out is not None
    return out


def closed_ridge_operator(
    W: Array,
    b: Array,
    c: Array,
    x: Array,
    terms: Sequence[RidgeTerm],
    activation: str = "sin",
) -> Array:
    """Operator value ``sum_t coeff_t L_t u(x)`` for each row of ``x`` (shape ``(B,)``).

    ``W`` is ``(H, D)``, ``b`` and ``c`` are ``(H,)`` and ``x`` is ``(B, D)``.
    Differentiable by ordinary autodiff; use :func:`closed_ridge_residual_loss`
    for the dedicated gradient rule.
    """
    terms_t = _validate(terms, W.shape[1])
    names, _, _ = derivative_closed_system(activation)
    rows = _rows(activation, max(_derivative_order(t) for t in terms_t))
    A = _amplitudes(W, terms_t, rows, 0)
    if not A:
        return jnp.zeros(x.shape[:1], dtype=jnp.result_type(x, W))
    v = _states(_project(x, W, b), names, set(A))
    return jnp.sum(_psi(A, v) * c[None, :], axis=1)


@lru_cache(maxsize=128)
def _make_loss(terms: tuple[RidgeTerm, ...], activation: str):  # type: ignore[no-untyped-def]
    names, _, _ = derivative_closed_system(activation)
    rows = _rows(activation, max(_derivative_order(t) for t in terms) + 1)

    def forward(W: Array, b: Array, c: Array, x: Array, target: Array) -> Array:
        A = _amplitudes(W, terms, rows, 0)
        if not A:
            return -target
        psi = _psi(A, _states(_project(x, W, b), names, set(A)))
        return _readout(psi, c, W.shape[1]) - target

    @jax.custom_vjp
    def loss_fn(W: Array, b: Array, c: Array, x: Array, target: Array) -> Array:
        r = forward(W, b, c, x, target)
        return 0.5 * jnp.mean(r * r)

    def fwd(W: Array, b: Array, c: Array, x: Array, target: Array):  # type: ignore[no-untyped-def]
        r = forward(W, b, c, x, target)
        return 0.5 * jnp.mean(r * r), (W, b, c, x, r)

    def bwd(res, gbar):  # type: ignore[no-untyped-def]
        W, b, c, x, r = res
        A = _amplitudes(W, terms, rows, 0)
        Ad = _amplitudes(W, terms, rows, 1)
        v = _states(
            _project(x, W, b), names, set(A) | set(Ad)
        )  # rematerialised: no (B, H) array is saved
        rbar = gbar * r / r.shape[0]
        Wbar = jnp.zeros_like(W)
        bbar = jnp.zeros_like(b)
        cbar = jnp.zeros_like(c)
        if Ad:
            zbar = rbar[:, None] * (_psi(Ad, v) * c[None, :])
            bbar = jnp.sum(zbar, axis=0)
            Wbar = zbar.T @ x
        # one (H,) column sum per state column, shared by every term (the order-independent part)
        col = {k: rbar @ v[k] for k in A}
        for k, amp in A.items():
            cbar = cbar + amp * col[k]
        q = jnp.sum(W * W, axis=-1)
        for term in terms:
            row = rows[_derivative_order(term)]
            s: Array | None = None
            for k, w in enumerate(row):
                if w:
                    part = float(w) * col[k]
                    s = part if s is None else s + part
            if s is None:
                continue
            sc = s * c
            if term.kind == "laplace":
                Wbar = (
                    Wbar + (term.coeff * term.order * 2.0 * q ** (term.order - 1) * sc)[:, None] * W
                )
            else:
                unit = jnp.zeros_like(W).at[:, term.axis].set(1.0)
                Wbar = (
                    Wbar
                    + (term.coeff * term.order * W[:, term.axis] ** (term.order - 1) * sc)[:, None]
                    * unit
                )
        return Wbar, bbar, cbar, None, None

    loss_fn.defvjp(fwd, bwd)
    return loss_fn


def closed_ridge_residual_loss(
    W: Array,
    b: Array,
    c: Array,
    x: Array,
    target: Array,
    terms: Sequence[RidgeTerm],
    activation: str = "sin",
) -> Array:
    """``0.5 * mean((sum_t coeff_t L_t u(x) - target)^2)`` with a rematerialising, order-independent reverse rule.

    Gradients flow into ``W``, ``b`` and ``c``; ``x`` and ``target`` are treated as constants.
    """
    terms_t = _validate(terms, W.shape[1])
    derivative_closed_system(activation)
    return _make_loss(terms_t, activation)(W, b, c, x, target)  # type: ignore[no-any-return]

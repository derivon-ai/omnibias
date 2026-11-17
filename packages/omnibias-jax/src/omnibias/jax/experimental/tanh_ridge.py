# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Opt-in value/gradient program for high-order operators of a tanh ridge field.

For the one-hidden-layer field ``u(x) = sum_h c_h tanh(w_h . x + b_h)`` every
constant-coefficient term of the supported operators has an exact ridge form

``Delta^k u    = sum_h c_h |w_h|^(2k) phi_(2k)(z_h)``
``d_i^n u      = sum_h c_h w_(h,i)^n phi_n(z_h)``

with ``z = w . x + b`` and ``phi_n = tanh^(n)``.  :func:`tanh_ridge_operator`
evaluates ``phi_1 ... phi_N`` from one ``tanh`` and exact integer polynomials in
``s = sech^2``.  :func:`tanh_ridge_residual_loss` wraps the mean-square operator
residual in a hand-written reverse rule: the backward pass recomputes ``z`` (no
``(B, H)`` residual is stored) and keeps the live ``|w|^(2k)`` / ``w_i^n``
metric term, so gradients into ``W``, ``b`` and ``c`` are complete.

Scope and limits.  The program is exact algebra for ``tanh`` ridge features,
not a general activation or deep-network method.  Derivative orders ``2..8`` are
accepted; orders ``2..6`` are the measured and oracle-qualified range.  The
default front end forms ``s`` as ``1 - tanh(z)^2``, which loses relative accuracy
in the tails; ``accurate_tails=True`` forms ``s`` from ``exp(-2|z|)`` at a
measured cost.  Neither choice is a convergence or stability statement for any
PDE.  See ``docs/performance.md`` for the measured regimes and the baselines that
match or beat this program.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["RidgeTerm", "tanh_ridge_operator", "tanh_ridge_residual_loss"]

MAX_DERIVATIVE_ORDER = 8
_UNROLL_MAX_D = 8  # unrolled multiply-add projection for small input dimension


class RidgeTerm(NamedTuple):
    """One term ``coeff * L`` of a ridge operator.

    ``kind="laplace"`` is the Laplace power ``Delta^order`` (derivative order
    ``2 * order``); ``kind="coordinate"`` is ``d^order / d x_axis^order``.
    """

    kind: Literal["laplace", "coordinate"]
    order: int
    coeff: float = 1.0
    axis: int = 0


# --------------------------------------------------------------- exact derivative polynomials


def _mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _der(a: list[int]) -> list[int]:
    return [i * a[i] for i in range(1, len(a))] or [0]


def _add(a: list[int], b: list[int]) -> list[int]:
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def _strip(a: list[int]) -> list[int]:
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


@lru_cache(maxsize=1)
def _tables() -> dict[int, tuple[int, ...]]:
    """``tanh^(n) = (t if n even) * s * f_n(s)`` with integer ``f_n`` (low to high degree).

    With ``t' = s``, ``s' = -2 t s`` and ``t^2 = 1 - s``:
    odd ``n -> n + 1``: ``f' = -2 (f + s f_s)``; even ``n -> n + 1``: ``f' = (3 s - 2) f - 2 s (1 - s) f_s``.
    """
    f: dict[int, list[int]] = {1: [1]}
    for n in range(1, MAX_DERIVATIVE_ORDER + 1):
        cur = f[n]
        if n % 2 == 1:
            f[n + 1] = _strip([-2 * x for x in _add(cur, _mul([0, 1], _der(cur)))])
        else:
            f[n + 1] = _strip(_add(_mul([-2, 3], cur), _mul(_mul([0, -2], [1, -1]), _der(cur))))
    return {n: tuple(c) for n, c in f.items()}


def _front(z: Array, accurate_tails: bool) -> tuple[Array, Array]:
    t = jnp.tanh(z)
    if not accurate_tails:
        return t, 1.0 - t * t
    e = jnp.exp(-2.0 * jnp.abs(z))
    r = 1.0 / (1.0 + e)
    return t, 4.0 * e * r * r


def _phi(z: Array, n_max: int, accurate_tails: bool) -> list[Array]:
    """``[tanh^(1)(z), ..., tanh^(n_max)(z)]`` sharing one ``tanh`` and exact integer polynomials in ``s``."""
    t, s = _front(z, accurate_tails)
    table = _tables()
    out: list[Array] = []
    for n in range(1, n_max + 1):
        coeffs = table[n]
        acc = jnp.full_like(s, float(coeffs[-1]))
        for c in reversed(coeffs[:-1]):
            acc = acc * s + float(c)
        val = s * acc
        out.append(val * t if n % 2 == 0 else val)
    return out


# ------------------------------------------------------------------------ operator program


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
        if _derivative_order(term) > MAX_DERIVATIVE_ORDER:
            raise ValueError(
                f"derivative order {_derivative_order(term)} exceeds the supported maximum {MAX_DERIVATIVE_ORDER}"
            )
        if kind == "coordinate" and not 0 <= axis < dim:
            raise ValueError("coordinate axis out of range")
        out.append(RidgeTerm(kind, int(order), float(coeff), int(axis)))
    return tuple(out)


def _derivative_order(term: RidgeTerm) -> int:
    return 2 * term.order if term.kind == "laplace" else term.order


def _project(x: Array, W: Array, b: Array) -> Array:
    """``z = x W^T + b``; small ``D`` uses fused multiply-adds so ``z`` need not round-trip through memory."""
    if W.shape[1] > _UNROLL_MAX_D:
        return x @ W.T + b
    z = b[None, :] + x[:, 0:1] * W[None, :, 0]
    for d in range(1, W.shape[1]):
        z = z + x[:, d : d + 1] * W[None, :, d]
    return z


def _metric(term: RidgeTerm, W: Array) -> Array:
    if term.kind == "laplace":
        return term.coeff * jnp.sum(W * W, axis=-1) ** term.order
    return term.coeff * W[:, term.axis] ** term.order


def _psi(
    W: Array, phi: list[Array], terms: tuple[RidgeTerm, ...], with_derivative: bool = False
) -> tuple[Array, Array | None]:
    """``psi = sum_t g_t phi_(m_t)`` and, when requested, ``dpsi/dz = sum_t g_t phi_(m_t + 1)``."""
    psi = jnp.zeros_like(phi[0])
    dpsi = jnp.zeros_like(phi[0]) if with_derivative else None
    for term in terms:
        m = _derivative_order(term)
        g = _metric(term, W)
        psi = psi + g * phi[m - 1]
        if dpsi is not None:
            dpsi = dpsi + g * phi[m]
    return psi, dpsi


def tanh_ridge_operator(
    W: Array,
    b: Array,
    c: Array,
    x: Array,
    terms: Sequence[RidgeTerm],
    *,
    accurate_tails: bool = False,
) -> Array:
    """Operator value ``sum_t coeff_t L_t u(x)`` for each row of ``x`` (shape ``(B,)``).

    ``W`` is ``(H, D)``, ``b`` and ``c`` are ``(H,)`` and ``x`` is ``(B, D)``.
    Differentiable by ordinary autodiff; use :func:`tanh_ridge_residual_loss`
    for the dedicated gradient rule.
    """
    terms_t = _validate(terms, W.shape[1])
    z = _project(x, W, b)
    phi = _phi(z, max(_derivative_order(t) for t in terms_t), accurate_tails)
    psi, _ = _psi(W, phi, terms_t)
    return jnp.sum(psi * c[None, :], axis=1)


def _readout(psi: Array, c: Array, dim: int) -> Array:
    """``psi @ c``: one matrix-vector product for small ``D``, a fused multiply-reduce otherwise (measured per ``D`` regime)."""
    return psi @ c if dim <= _UNROLL_MAX_D else jnp.sum(psi * c[None, :], axis=1)


@lru_cache(maxsize=128)
def _make_loss(terms: tuple[RidgeTerm, ...], accurate_tails: bool):  # type: ignore[no-untyped-def]
    nmax = max(_derivative_order(t) for t in terms)

    def parts(
        W: Array, b: Array, x: Array, extra: int = 0
    ) -> tuple[list[Array], list[Array], Array]:
        phi = _phi(_project(x, W, b), nmax + extra, accurate_tails)
        gs = [_metric(t, W) for t in terms]
        psi = jnp.zeros_like(phi[0])
        for term, g in zip(terms, gs, strict=False):
            psi = psi + g * phi[_derivative_order(term) - 1]
        return phi, gs, psi

    @jax.custom_vjp
    def loss_fn(W: Array, b: Array, c: Array, x: Array, target: Array) -> Array:
        _, _, psi = parts(W, b, x)
        r = _readout(psi, c, W.shape[1]) - target
        return 0.5 * jnp.mean(r * r)

    def fwd(W: Array, b: Array, c: Array, x: Array, target: Array):  # type: ignore[no-untyped-def]
        _, _, psi = parts(W, b, x)
        r = _readout(psi, c, W.shape[1]) - target
        return 0.5 * jnp.mean(r * r), (W, b, c, x, r)

    def bwd(res, gbar):  # type: ignore[no-untyped-def]
        W, b, c, x, r = res
        phi, gs, _ = parts(
            W, b, x, 1
        )  # rematerialised: no (B, H) array is saved by the forward pass
        rbar = gbar * r / r.shape[0]
        dz = jnp.zeros_like(phi[0])
        for term, g in zip(terms, gs, strict=False):
            dz = dz + (c * g)[None, :] * phi[_derivative_order(term)]
        zbar = rbar[:, None] * dz
        bbar = jnp.sum(zbar, axis=0)
        Wbar = zbar.T @ x
        cbar = jnp.zeros_like(c)
        # live metric term: d g_t / dW_h contracted with colsum_b(rbar * phi_{m_t}) * c_h; the (H,) column sums are formed once per term
        q = jnp.sum(W * W, axis=-1)
        for term, g in zip(terms, gs, strict=False):
            colraw = rbar @ phi[_derivative_order(term) - 1]
            cbar = cbar + g * colraw
            col = colraw * c
            if term.kind == "laplace":
                Wbar = (
                    Wbar
                    + (term.coeff * term.order * 2.0 * q ** (term.order - 1) * col)[:, None] * W
                )
            else:
                unit = jnp.zeros_like(W).at[:, term.axis].set(1.0)
                Wbar = (
                    Wbar
                    + (term.coeff * term.order * W[:, term.axis] ** (term.order - 1) * col)[:, None]
                    * unit
                )
        return Wbar, bbar, cbar, None, None

    loss_fn.defvjp(fwd, bwd)
    return loss_fn


def tanh_ridge_residual_loss(
    W: Array,
    b: Array,
    c: Array,
    x: Array,
    target: Array,
    terms: Sequence[RidgeTerm],
    *,
    accurate_tails: bool = False,
) -> Array:
    """``0.5 * mean((sum_t coeff_t L_t u(x) - target)^2)`` with a rematerialising reverse rule.

    Gradients flow into ``W``, ``b`` and ``c``; ``x`` and ``target`` are treated as constants.
    """
    terms_t = _validate(terms, W.shape[1])
    return _make_loss(terms_t, bool(accurate_tails))(W, b, c, x, target)  # type: ignore[no-any-return]

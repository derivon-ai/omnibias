# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form fractional derivative of an analytic function (jax).

Bit-identical twin of :mod:`omnibias.fractional.torch.ops.analytic`. Given the
Taylor jet ``a_k = f^(k)(a)/k!`` of a function about a terminal ``a``, the
fractional derivative of order ``alpha`` is

.. math::

    {}_a D_x^{\alpha} f(x)
        = \sum_{k} a_k \,\frac{\Gamma(k+1)}{\Gamma(k+1-\alpha)}\,(x-a)^{k-\alpha},

with ``t = x - a >= 0``. It is closed form on the analytic-function class (no
grid, no history), differentiable in both the order ``alpha`` (through
``gammaln``) and the coefficients ``a_k``. See the torch twin for the full
Riemann-Liouville / Caputo, terminal, branch-point, and integer-order notes.
The gamma ratio uses ``gammaln`` + an explicit real sign (not
``jax.scipy.special.gamma``) so the two backends match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid as jax_sigmoid
from jax.scipy.special import gammaln

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from omnibias.jax.activations import JaxActivationSpec

    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | str | None]


def _gamma_ratio(k: Array, alpha: Array) -> Array:
    r"""Real gamma ratio ``Gamma(k+1) / Gamma(k+1-alpha)`` (see the torch twin)."""
    y = (k + 1.0) - alpha
    log_mag = gammaln(k + 1.0) - gammaln(y)
    parity = jnp.remainder(jnp.ceil(-y), 2.0)
    sign = jnp.where(y > 0, jnp.ones_like(y), 1.0 - 2.0 * parity)
    return sign * jnp.exp(log_mag)


def fractional_derivative(
    jet: Array,
    x: Array | float,
    *,
    alpha: float | Array,
    a: float = 0.0,
    kind: str = "riemann_liouville",
) -> Array:
    r"""Closed-form fractional derivative of an analytic function from its jet.

    See :func:`omnibias.fractional.torch.ops.analytic.fractional_derivative` for
    the full parameter semantics; this is the bit-identical JAX twin (output
    shape ``(*feat, *x.shape)``, differentiable in ``alpha`` and the jet).
    """
    if kind not in ("riemann_liouville", "caputo"):
        raise ValueError(
            f"kind must be 'riemann_liouville' or 'caputo', got {kind!r}"
        )
    if jet.ndim < 1:
        raise ValueError("jet must have a leading order axis, shape (N+1, ...)")

    n1 = jet.shape[0]
    a_t = jnp.asarray(alpha, dtype=jet.dtype)

    k = jnp.arange(n1, dtype=jet.dtype)
    ratio = _gamma_ratio(k, a_t)
    if kind == "caputo":
        ratio = ratio * (k >= jnp.ceil(a_t)).astype(jet.dtype)

    x_t = jnp.asarray(x, dtype=jet.dtype)
    t = x_t - a

    fdims = jet.ndim - 1
    pdims = t.ndim
    shape_k = (n1,) + (1,) * fdims + (1,) * pdims
    ratio_e = ratio.reshape(shape_k)
    exps_e = (k - a_t).reshape(shape_k)
    jet_e = jet.reshape(jet.shape + (1,) * pdims)
    t_e = t.reshape((1,) * (1 + fdims) + t.shape)

    powers = t_e**exps_e
    # Masked / vanishing terms (ratio == 0, e.g. Caputo below ceil(alpha) or a
    # Gamma pole) contribute exactly 0 even at the terminal, where ``t**exps``
    # is ``0**negative = inf`` and ``0 * inf`` would otherwise be ``nan``.
    terms = jnp.where(ratio_e == 0.0, 0.0, jet_e * ratio_e * powers)
    out: Array = terms.sum(axis=0)
    return out


def _patch_weights(term: Array, x: Array, blend: float) -> Array:
    r"""Patch-selection weights ``(M, *x)`` summing to 1 (see the torch twin)."""
    m = int(term.shape[0])
    if m == 1:
        return jnp.ones((1, *x.shape), dtype=x.dtype)
    if blend > 0.0:
        bnds = term[1:].reshape((m - 1,) + (1,) * x.ndim)
        p = jax_sigmoid((x[None] - bnds) / blend)  # (M-1, *x)
        w0 = 1.0 - p[:1]
        wlast = p[-1:]
        if m == 2:
            return jnp.concatenate([w0, wlast], axis=0)
        return jnp.concatenate([w0, p[:-1] - p[1:], wlast], axis=0)
    idx = jnp.searchsorted(term, x.reshape(-1), side="right").reshape(x.shape) - 1
    idx = jnp.clip(idx, 0, m - 1)
    rng = jnp.arange(m).reshape((m,) + (1,) * x.ndim)
    return (idx[None] == rng).astype(x.dtype)


def piecewise_fractional_derivative(
    jets: Array,
    terminals: Sequence[float] | Array,
    x: Array | float,
    *,
    alpha: float | Array,
    kind: str = "riemann_liouville",
    blend: float = 0.0,
    gap: float = 1e-6,
) -> Array:
    r"""Piecewise-analytic (multi-terminal) closed-form fractional derivative (jax twin).

    See :func:`omnibias.fractional.torch.ops.analytic.piecewise_fractional_derivative`
    for the full semantics (short-memory restart at each terminal, hard vs smooth
    ``blend`` patch selection, ``gap`` safe-clamp). Bit-identical to the torch twin;
    differentiable in ``alpha`` and the jets.
    """
    if jets.ndim < 2:
        raise ValueError("jets must have shape (M, N+1, *feat)")
    term_list = [float(t) for t in terminals]
    m = jets.shape[0]
    if len(term_list) != m:
        raise ValueError(f"expected {m} terminals (one per jet), got {len(term_list)}")
    if any(term_list[i] >= term_list[i + 1] for i in range(m - 1)):
        raise ValueError("terminals must be strictly increasing")

    x_t = jnp.asarray(x, dtype=jets.dtype)
    if float(jnp.min(x_t)) < term_list[0] - 1e-9:
        raise ValueError(
            f"every x must be >= the first terminal {term_list[0]}, got min {float(jnp.min(x_t))}"
        )
    feat = jets.ndim - 2

    parts = [
        fractional_derivative(
            jets[i], jnp.maximum(x_t, term_list[i] + gap), alpha=alpha, a=term_list[i], kind=kind
        )
        for i in range(m)
    ]
    stack = jnp.stack(parts, axis=0)  # (M, *feat, *x)

    term_t = jnp.asarray(term_list, dtype=x_t.dtype)
    weights = _patch_weights(term_t, x_t, blend)  # (M, *x)
    w_e = weights.reshape((m,) + (1,) * feat + tuple(x_t.shape))
    out: Array = (w_e * stack).sum(axis=0)
    return out


def mlp_fractional_derivative(
    x0: Array,
    v: Array,
    layers: Sequence[LayerSpec],
    t: Array | float,
    *,
    alpha: float | Array,
    order: int,
    kind: str = "riemann_liouville",
) -> Array:
    r"""Closed-form directional fractional derivative of a deep MLP (jax twin).

    See :func:`omnibias.fractional.torch.ops.analytic.mlp_fractional_derivative`.
    Requires the ``omnibias-jax`` package (imported lazily).
    """
    from omnibias.jax.jet import mlp_jet

    jet = mlp_jet(x0, v, layers, order)
    return fractional_derivative(jet, t, alpha=alpha, a=0.0, kind=kind)


__all__ = [
    "fractional_derivative",
    "mlp_fractional_derivative",
    "piecewise_fractional_derivative",
]

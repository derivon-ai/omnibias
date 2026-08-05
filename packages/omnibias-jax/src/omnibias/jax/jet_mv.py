# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact multivariate (multi-index) jets via Faà di Bruno (jax).

Bit-identical twin of :mod:`omnibias.torch.jet_mv`. This is the full
``D``-variable generalisation of the directional jets in
:mod:`omnibias.jax.jet`: instead of a single Taylor series in the path
parameter ``t``, we carry the *whole* truncated multivariate Taylor expansion of
a deep network around a base point ``x0``,

.. math::

    f(x_0 + \delta) = \sum_{|\alpha| \le N} c_\alpha\, \delta^\alpha,
    \qquad c_\alpha = \frac{D^\alpha f(x_0)}{\alpha!},

so a single forward pass yields *every* mixed partial derivative up to total
order ``N`` (the gradient, the full Hessian, third-order tensors, ...).

Representation
--------------
A multivariate jet is a dense array of shape ``(M, ...)`` where
``M = num_multi_indices(dim, order)`` and row ``i`` holds the Taylor coefficient
``c_alpha`` for ``alpha = multi_indices(dim, order)[i]``. The trailing axes are
hidden units / output components, exactly as for the directional jet.

Kernel
------
Affine maps act per coefficient (the map is degree-1 in ``delta``); the bias hits
the constant coefficient only. Activations use the same shifted-power identity as
the directional kernel,

.. math::

    \sigma(u) = \sum_{k=0}^{N} \frac{\sigma^{(k)}(u_0)}{k!}\,(u - u_0)^k,

but the series powers ``(u - u_0)^k`` are built with *multivariate* truncated
polynomial multiplication (:func:`omnibias.core.multi_index.multiply_table`)
rather than 1-D convolution. The exact derivative tower ``sigma^(k)(u_0)`` comes
from the omnibias activation fast paths, so the composition is exact and needs
neither nested autodiff nor finite differences.

.. important::

    **Bit-parity requires 64-bit JAX.** As with :mod:`omnibias.jax.jet`, the
    "bit-identical twin" guarantee against :mod:`omnibias.torch.jet_mv` holds in
    double precision only. JAX silently truncates arrays to ``float32`` unless
    64-bit mode is enabled *before* the first JAX import::

        import jax
        jax.config.update("jax_enable_x64", True)

    (equivalently ``JAX_ENABLE_X64=1``). Without it the jet is still internally
    consistent, but cross-backend comparisons match only to ``float32``
    tolerance, not bit-for-bit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from omnibias.core.multi_index import (
    index_position,
    multi_index_factorial,
    multi_indices,
    multiply_table,
    num_multi_indices,
)
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.jet import _sigma_tower, affine_jet

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | str | None]


def compose_jet_mv(
    u_jet: Array, sigma_tower: Array, dim: int, order: int
) -> Array:
    """Compose an elementwise activation onto a multivariate jet: ``b = sigma(u)``.

    Parameters
    ----------
    u_jet
        Pre-activation jet, shape ``(M, ...)`` with
        ``M = num_multi_indices(dim, order)`` (rows in the canonical
        :func:`~omnibias.core.multi_index.multi_indices` order).
    sigma_tower
        Derivative tower of the activation at the constant coefficient
        ``u_jet[0]``: ``sigma_tower[k] = sigma^(k)(u_jet[0])`` for
        ``k = 0..order`` (shape ``(order + 1, ...)``).
    dim, order
        Number of input variables and the truncation total order.

    Returns
    -------
    Array
        Jet of ``sigma(u)`` of shape ``(M, ...)``.
    """
    u_jet = jnp.asarray(u_jet)
    sigma_tower = jnp.asarray(sigma_tower)
    m = num_multi_indices(dim, order)
    if u_jet.shape[0] != m:
        raise ValueError(
            f"u_jet has {u_jet.shape[0]} rows but dim={dim}, order={order} "
            f"requires {m}"
        )
    if sigma_tower.shape[0] != order + 1:
        raise ValueError(
            f"sigma_tower order {sigma_tower.shape[0] - 1} must equal jet order "
            f"{order}"
        )

    table = multiply_table(dim, order)
    zero = jnp.zeros_like(u_jet[0])
    # w = u - u0 has no constant term (row 0 is zero).
    w = [zero] + [u_jet[g] for g in range(1, m)]
    # p holds the coefficients of w^k; start at w^0 = 1 (only the constant row).
    p = [jnp.ones_like(u_jet[0])] + [zero for _ in range(m - 1)]
    result = [sigma_tower[0] * p[0]] + [zero for _ in range(m - 1)]
    fact = 1.0
    for k in range(1, order + 1):
        fact *= k
        # p <- truncated multivariate product (p * w) via the Cauchy table.
        acc = [zero for _ in range(m)]
        for g, a, b in table:
            acc[g] = acc[g] + p[a] * w[b]
        p = acc
        dk = sigma_tower[k] / fact
        for g in range(m):
            result[g] = result[g] + dk * p[g]
    return jnp.stack(result, axis=0)


def jet_multiply(a: Array, b: Array, dim: int, order: int) -> Array:
    r"""Truncated product of two multivariate jets: ``jet(a) * jet(b)``.

    Bit-identical twin of :func:`omnibias.torch.jet_mv.jet_multiply`. Given the
    jets of two analytic functions about the *same* base point, returns the jet of
    their pointwise product via the multivariate Cauchy product

    .. math::

        (a\,b)_\gamma = \sum_{\alpha + \beta = \gamma} a_\alpha\, b_\beta,

    truncated at total order ``order`` (the same
    :func:`~omnibias.core.multi_index.multiply_table` that drives
    :func:`compose_jet_mv`). Both ``a`` and ``b`` carry
    ``M = num_multi_indices(dim, order)`` leading rows; any trailing shape
    broadcasts, so a scalar mask ``(M,)`` multiplies a vector field ``(M, C)``
    component-wise. This is the jet-level Leibniz rule and is what keeps a
    product ansatz such as ``u = g + b * net`` closed-form to arbitrary order.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    m = num_multi_indices(dim, order)
    if a.shape[0] != m:
        raise ValueError(
            f"a has {a.shape[0]} rows but dim={dim}, order={order} requires {m}"
        )
    if b.shape[0] != m:
        raise ValueError(
            f"b has {b.shape[0]} rows but dim={dim}, order={order} requires {m}"
        )
    table = multiply_table(dim, order)
    sample = a[0] * b[0]
    out = [jnp.zeros_like(sample) for _ in range(m)]
    for g, al, be in table:
        out[g] = out[g] + a[al] * b[be]
    return jnp.stack(out, axis=0)


def jet_reciprocal(u_jet: Array, dim: int, order: int) -> Array:
    r"""Jet of the reciprocal ``1 / u`` from the jet of ``u``.

    Bit-identical twin of :func:`omnibias.torch.jet_mv.jet_reciprocal`. ``1/u`` is
    analytic wherever ``u(x_0) != 0`` and its derivative tower is elementary,

    .. math::

        \frac{d^k}{du^k}\,u^{-1} = (-1)^k\, k!\, u^{-(k+1)},

    so the reciprocal composes through the *same* :func:`compose_jet_mv` kernel as
    any activation -- it is simply a tower this module supplies itself rather than
    one the activation dictionary carries. This is what makes a *rational* map (a
    normalisation, a softmax denominator, a quotient ansatz) closed form at
    arbitrary order instead of a nested-autodiff fallback.

    A vanishing ``u_jet[0]`` is not an error: the result is then infinite or
    ``nan`` in the usual floating-point way, matching division elsewhere.
    """
    u_jet = jnp.asarray(u_jet)
    m = num_multi_indices(dim, order)
    if u_jet.shape[0] != m:
        raise ValueError(
            f"u_jet has {u_jet.shape[0]} rows but dim={dim}, order={order} "
            f"requires {m}"
        )
    inv = 1.0 / u_jet[0]
    rows = []
    power = inv
    coeff = 1.0
    for k in range(order + 1):
        if k > 0:
            power = power * inv
            coeff = -coeff * k
        rows.append(coeff * power)
    return compose_jet_mv(u_jet, jnp.stack(rows, axis=0), dim, order)


def jet_exp(u_jet: Array, dim: int, order: int) -> Array:
    """Jet of ``exp(u)`` from the jet of ``u``.

    A thin wrapper over :func:`compose_jet_mv` with the tower of the registered
    ``"exp"`` activation (``exp^(k) = exp`` at every order), so the exponential is
    not special-cased -- it is one more entry of the shared dictionary.
    """
    u_jet = jnp.asarray(u_jet)
    tower = _sigma_tower(get_activation("exp"), u_jet[0], order)
    return compose_jet_mv(u_jet, tower, dim, order)


def jet_softmax(s_jet: Array, dim: int, order: int) -> Array:
    r"""Jet of ``softmax(s)`` over the trailing axis, from the jet of the scores.

    Bit-identical twin of :func:`omnibias.torch.jet_mv.jet_softmax`. Softmax is the
    first genuinely *non-elementwise* map in this module: every output couples to
    every score through the shared denominator. It still factors into primitives
    that are each exact --

    .. math::

        \mathrm{softmax}(s)_j = e^{s_j} \cdot \Big(\sum_i e^{s_i}\Big)^{-1},

    an :func:`jet_exp`, a row-wise sum (linear, so it acts per coefficient), a
    :func:`jet_reciprocal`, and a :func:`jet_multiply` -- so the composite jet is
    exact to machine precision at arbitrary order, with no nested autodiff.

    ``s_jet`` has shape ``(M, ..., n)``; the softmax is taken over the last axis.
    The constant coefficient is max-shifted before exponentiating, exactly as a
    stable softmax implementation does. The shift is held constant for the
    gradient and is identical across the axis, so it cancels in the mathematics
    and only removes overflow.
    """
    s_jet = jnp.asarray(s_jet)
    m = num_multi_indices(dim, order)
    if s_jet.shape[0] != m:
        raise ValueError(
            f"s_jet has {s_jet.shape[0]} rows but dim={dim}, order={order} "
            f"requires {m}"
        )
    if s_jet.ndim < 2:
        raise ValueError(
            f"s_jet must carry a trailing softmax axis, got shape {s_jet.shape}"
        )
    shift = jnp.max(jax.lax.stop_gradient(s_jet[0]), axis=-1, keepdims=True)
    centred = jnp.concatenate([(s_jet[0] - shift)[None, ...], s_jet[1:]], axis=0)
    e_jet = jet_exp(centred, dim, order)
    total = jnp.sum(e_jet, axis=-1, keepdims=True)  # linear: acts per coefficient
    return jet_multiply(e_jet, jet_reciprocal(total, dim, order), dim, order)


def jet_attention(
    q_jet: Array,
    keys: Array,
    values: Array,
    dim: int,
    order: int,
    beta: float | Array = 1.0,
) -> Array:
    r"""Jet of dot-product attention ``softmax(beta q K^T) V`` w.r.t. the *inputs*.

    Bit-identical twin of :func:`omnibias.torch.jet_mv.jet_attention`.
    :mod:`omnibias.hopfield` already differentiates the log-sum-exp core in closed
    form, but with respect to the **scores**. That is the wrong variable for a PDE:
    a residual needs ``d/dx``. Pushing the query jet through the block supplies the
    missing coordinate story -- the value agrees with
    :func:`omnibias.hopfield.jax.ops.attention` and every mixed partial ``D^alpha``
    of the attention output comes out of the same single jet.

    Parameters
    ----------
    q_jet
        Jet of the query, shape ``(M, ..., d_key)``.
    keys, values
        Memory of shape ``(n, d_key)`` and ``(n, d_val)``; constant in the input
        (they are network parameters, not functions of ``x``), so they enter as two
        affine maps around the softmax.
    dim, order
        Input-variable count and truncation total order.
    beta
        Inverse temperature. An array is allowed, which keeps a *trainable*
        temperature closed form in exactly the same way.
    """
    q_jet = jnp.asarray(q_jet)
    keys = jnp.asarray(keys)
    values = jnp.asarray(values)
    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError(
            f"keys and values must be 2-D, got {keys.shape} and {values.shape}"
        )
    if keys.shape[0] != values.shape[0]:
        raise ValueError(
            f"keys and values must share the memory axis, got {keys.shape[0]} "
            f"keys and {values.shape[0]} values"
        )
    if q_jet.shape[-1] != keys.shape[-1]:
        raise ValueError(
            f"query width {q_jet.shape[-1]} != key width {keys.shape[-1]}"
        )
    scores = affine_jet(q_jet, keys * beta)
    weights = jet_softmax(scores, dim, order)
    return affine_jet(weights, jnp.swapaxes(values, -2, -1))


def affine_jet_mv(z_jet: Array, W: Array, b: Array | None = None) -> Array:
    """Push a multivariate jet through an affine map ``u = W z + b``.

    Identical per-coefficient action to :func:`omnibias.jax.affine_jet`: the map
    is degree-1 in the perturbation, so it acts row-wise with the bias added to
    the constant coefficient (row 0) only.
    """
    return affine_jet(z_jet, W, b)


def layer_jet_mv(
    z_jet: Array,
    W: Array,
    b: Array | None,
    spec: JaxActivationSpec | str,
    dim: int,
    order: int,
) -> Array:
    """Push a multivariate jet through one ``sigma(W z + b)`` layer."""
    z_jet = jnp.asarray(z_jet)
    resolved = get_activation(spec)
    u_jet = affine_jet(z_jet, W, b)
    sigma_tower = _sigma_tower(resolved, u_jet[0], order)
    return compose_jet_mv(u_jet, sigma_tower, dim, order)


def identity_jet(x0: Array, order: int) -> Array:
    """Multivariate jet of the identity map ``x(delta) = x0 + delta``.

    Returns shape ``(M, D)`` where ``D = x0.shape[-1]``: row 0 (the constant
    coefficient) is ``x0`` and each unit-multi-index row ``e_i`` is the standard
    basis vector ``e_i`` (since ``d x_j / d delta_i = delta_{ij}``). All other
    rows are zero. This is the seed for :func:`mlp_jet_mv`.
    """
    x0 = jnp.asarray(x0)
    if x0.ndim != 1:
        raise ValueError(f"x0 must be a 1-D vector, got shape {x0.shape}")
    dim = int(x0.shape[0])
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    pos = index_position(dim, order)
    m = num_multi_indices(dim, order)
    rows = [jnp.zeros((dim,), dtype=x0.dtype) for _ in range(m)]
    rows[pos[(0,) * dim]] = x0
    if order >= 1:
        eye = jnp.eye(dim, dtype=x0.dtype)
        for i in range(dim):
            e = tuple(1 if j == i else 0 for j in range(dim))
            rows[pos[e]] = eye[i]
    return jnp.stack(rows, axis=0)


def mlp_jet_mv(
    x0: Array,
    layers: Sequence[tuple[Array, Array | None, JaxActivationSpec | str | None]],
    order: int,
) -> Array:
    """Exact multivariate Taylor jet of a deep MLP around ``x0``.

    Parameters
    ----------
    x0
        Base point, shape ``(D,)``.
    layers
        Sequence of ``(W, b, spec)``. ``spec=None`` is a pure affine readout;
        otherwise the layer computes ``sigma(W z + b)``.
    order
        Truncation total order ``N``; the returned jet has
        ``num_multi_indices(D, N)`` rows.

    Returns
    -------
    Array
        Jet of shape ``(M, C)`` whose row ``i`` is the Taylor coefficient
        ``D^alpha f(x0) / alpha!`` for ``alpha = multi_indices(D, N)[i]``. Use
        :func:`jet_partials`, :func:`jet_gradient` or :func:`jet_hessian` to read
        out raw derivatives.
    """
    x0 = jnp.asarray(x0)
    dim = int(x0.shape[-1])
    jet = identity_jet(x0, order)
    for W, b, spec in layers:
        if spec is None:
            jet = affine_jet(jet, W, b)
        else:
            jet = layer_jet_mv(jet, W, b, spec, dim, order)
    return jet


def jet_partials(jet: Array, dim: int, order: int) -> dict[tuple[int, ...], Array]:
    """Convert a jet to raw partial derivatives ``{alpha: D^alpha f(x0)}``.

    Each coefficient row is scaled by ``alpha!`` (``D^alpha f = alpha! c_alpha``).
    """
    jet = jnp.asarray(jet)
    idx = multi_indices(dim, order)
    if jet.shape[0] != len(idx):
        raise ValueError(
            f"jet has {jet.shape[0]} rows but dim={dim}, order={order} "
            f"requires {len(idx)}"
        )
    return {alpha: jet[i] * multi_index_factorial(alpha) for i, alpha in enumerate(idx)}


def jet_gradient(jet: Array, dim: int, order: int) -> Array:
    """Gradient ``d f / d x_i`` from a jet, shape ``(D, ...)``.

    Requires ``order >= 1``.
    """
    if order < 1:
        raise ValueError(f"gradient needs order >= 1, got {order}")
    jet = jnp.asarray(jet)
    pos = index_position(dim, order)
    rows = []
    for i in range(dim):
        e = tuple(1 if j == i else 0 for j in range(dim))
        rows.append(jet[pos[e]])  # alpha! = 1 for a unit multi-index
    return jnp.stack(rows, axis=0)


def jet_hessian(jet: Array, dim: int, order: int) -> Array:
    """Hessian ``d^2 f / d x_i d x_j`` from a jet, shape ``(D, D, ...)``.

    Requires ``order >= 2``. The symmetric entry ``H[i, j]`` is read off the
    coefficient of ``alpha = e_i + e_j`` and rescaled by ``alpha!`` (2 on the
    diagonal, 1 off-diagonal).
    """
    if order < 2:
        raise ValueError(f"hessian needs order >= 2, got {order}")
    jet = jnp.asarray(jet)
    pos = index_position(dim, order)
    rows = []
    for i in range(dim):
        row = []
        for j in range(dim):
            alpha = tuple(
                (1 if k == i else 0) + (1 if k == j else 0) for k in range(dim)
            )
            row.append(jet[pos[alpha]] * multi_index_factorial(alpha))
        rows.append(jnp.stack(row, axis=0))
    return jnp.stack(rows, axis=0)


__all__ = [
    "affine_jet_mv",
    "compose_jet_mv",
    "identity_jet",
    "jet_attention",
    "jet_exp",
    "jet_gradient",
    "jet_hessian",
    "jet_multiply",
    "jet_partials",
    "jet_reciprocal",
    "jet_softmax",
    "layer_jet_mv",
    "mlp_jet_mv",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Divergence of deep MLP vector fields through the input/output projection.

For a vector field ``s(x) = W_L a_n + b_L`` with ``a_l = sigma_l(z_l)``,
``z_l = W_l a_{l-1} + b_l`` and ``a_0 = x``, the Jacobian is
``J = W_L S_n W_n ... W_2 S_1 W_1`` with ``S_l = diag sigma_l'(z_l)``. The
trace is cyclic, so

    div s = tr(J) = tr(S_n W_n ... W_2 S_1 C),    C = W_1 W_L,

where ``C`` has shape ``(H_1, H_n)`` and is formed once per parameter set:
the input dimension ``D`` enters only through that product. With one hidden
layer the divergence is ``diag(C) . sigma_1'(z_1)``; with two it is
``sigma_2'^T (W_2 o C^T) sigma_1'``. Both cost about one forward pass.
Deeper networks carry the ``H_n`` columns of ``C`` through the middle layers,
``O(H^3)`` per point and layer instead of the ``O(D H^2)`` of ``D``
forward-mode directions, and close the trace against the last hidden layer
without forming that product. When ``D`` directions are cheaper, the columns
of ``W_1`` are carried instead.

Everything is ordinary JAX, so parameter gradients, input gradients and
higher derivatives follow from autodiff.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from omnibias.jax.experimental._chain import (
    affine_readout,
    carry_columns,
    columns,
    hidden_slopes,
    pairing_index,
    points_2d,
    readout_value,
    require_layers,
)

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.jax.experimental._chain import LayerSpec

__all__ = ["projected_deep_divergence"]


def _trace(weights: list[Array], slopes: list[Array], WL: Array, index: np.ndarray | None) -> Array:
    """``tr(W_L S_n W_n ... S_1 W_1[:, index])`` per point; ``slopes`` are unit-major ``(H_l, B)``."""
    n = len(weights)
    W1 = columns(weights[0], index)
    if n == 1:
        return jnp.sum(W1 * WL.T, axis=1) @ slopes[0]
    if n == 2:
        E = weights[1] * (W1 @ WL).T
        return jnp.sum((E @ slopes[0]) * slopes[1], axis=0)
    if (n - 2) * WL.shape[1] < (n - 1) * WL.shape[0]:
        T = carry_columns(weights[:-1], slopes[:-1], lambda: W1 @ WL)
        return jnp.einsum("ib,ij,jbi->b", slopes[-1], weights[-1], T)
    T = carry_columns(weights, slopes, W1)
    return jnp.einsum("ji,ibj->b", WL, T)


def projected_deep_divergence(
    x: Array,
    layers: Sequence[LayerSpec],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
) -> tuple[Array, Array]:
    r"""``(s(x), div s(x))`` for a deep MLP vector field.

    Parameters
    ----------
    x : Array of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)``
        ``W`` has shape ``(out, in)``; ``spec=None`` is an affine layer. The
        last layer must be affine.
    coordinates : optional concrete integer indices of shape ``(m,)``: output
        ``j`` is differentiated along input ``coordinates[j]``, for example the
        space coordinates of a field that also takes time. ``None`` pairs
        output ``j`` with input ``j`` and needs ``m = D`` outputs.

    Returns
    -------
    value : Array of shape ``(m,)`` or ``(B, m)``.
    divergence : Array of shape ``()`` or ``(B,)``.
    """
    require_layers(layers)
    a, batched = points_2d(x)
    batch, dim = a.shape
    hidden, WL, bL = affine_readout(layers)
    outputs = WL.shape[0]
    index = pairing_index(coordinates, outputs, dim)
    # Unit-major layout (unit, point): every layer is one GEMM over the leading axis.
    weights, slopes, a = hidden_slopes(a.T, hidden)
    value = readout_value(WL, bL, a, len(hidden))
    if weights:
        divergence = _trace(weights, slopes, WL, index)
    else:
        total = jnp.trace(WL) if index is None else jnp.sum(WL[np.arange(outputs), index])
        divergence = jnp.broadcast_to(total, (batch,))
    if not batched:
        return value[:, 0], divergence[0]
    return value.T, divergence

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jacobian log-determinants of residual MLP blocks through the Sylvester identity.

For a vector field ``s(x) = W_L a_n + b_L`` with ``a_l = sigma_l(z_l)``,
``z_l = W_l a_{l-1} + b_l`` and ``a_0 = x``, the Jacobian is ``J = W_L A W_1``
with ``A = S_n W_n ... W_2 S_1`` and ``S_l = diag sigma_l'(z_l)``. Sylvester's
determinant identity moves ``W_L`` to the other side:

    det(I_D + J) = det(I_{H_n} + A C),    C = W_1 W_L,

so the log-determinant of the residual block ``x + s(x)`` needs the
``H_n x H_n`` matrix ``A C`` per point instead of the ``D x D`` Jacobian. The
``H_n`` columns of ``C`` (formed once per parameter set) are carried through
the hidden layers, ``O(H^3)`` per point and layer, followed by one
``H_n x H_n`` LU factorization instead of a ``D x D`` one. When the field has
fewer outputs than ``H_n``, the smaller Jacobian is formed directly. The
trace of the same matrix is the divergence.

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

__all__ = ["projected_deep_logdet"]


def _block(weights: list[Array], slopes: list[Array], WL: Array, index: np.ndarray | None) -> Array:
    """Per point, ``J[:, index]`` (``m < H_n``) or ``A C`` (otherwise): ``(B, k, k)`` with equal determinants
    of ``I + .``; ``slopes`` are unit-major ``(H_l, B)``."""
    W1 = columns(weights[0], index)
    m, width = WL.shape
    seeds = W1 if m < width else W1 @ WL
    T = carry_columns(weights, slopes, seeds)
    if m < width:
        return jnp.einsum("ji,ibd->bjd", WL, T)
    return jnp.transpose(T, (1, 0, 2))


def projected_deep_logdet(
    x: Array,
    layers: Sequence[LayerSpec],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
) -> tuple[Array, Array, Array]:
    r"""``(s(x), sign, log|det(I + J_s(x))|)`` for a deep MLP vector field.

    Parameters
    ----------
    x : Array of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)``
        ``W`` has shape ``(out, in)``; ``spec=None`` is an affine layer. The
        last layer must be affine.
    coordinates : optional concrete integer indices of shape ``(m,)``: the
        determinant is of ``I + d s / d x[coordinates]``, for example the space
        block of a field that also takes time. ``None`` uses every input and
        needs ``m = D`` outputs.

    Returns
    -------
    value : Array of shape ``(m,)`` or ``(B, m)``.
    sign, logabsdet : Arrays of shape ``()`` or ``(B,)``, as from
        :func:`jax.numpy.linalg.slogdet`.
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
        block = _block(weights, slopes, WL, index)
    else:
        block = jnp.broadcast_to(columns(WL, index), (batch, outputs, outputs))
    sign, logabsdet = jnp.linalg.slogdet(jnp.eye(block.shape[-1], dtype=block.dtype) + block)
    if not batched:
        return value[:, 0], sign[0], logabsdet[0]
    return value.T, sign, logabsdet

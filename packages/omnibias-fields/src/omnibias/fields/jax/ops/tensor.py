# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Divergence of a second-order tensor field (jax).

Bit-identical twin of :mod:`omnibias.fields.torch.ops.tensor`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import derivative

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def tensor_divergence(
    state: FieldState,
    sigma_names: Sequence[Sequence[str]],
) -> Array:
    r"""Row-wise divergence :math:`(\nabla\cdot\sigma)_i = \partial_j \sigma_{ij}`.

    See :func:`omnibias.fields.torch.ops.tensor.tensor_divergence`.
    """
    spatial = state.coordinate_spec.spatial_axes
    d = len(spatial)
    rows = list(sigma_names)
    if len(rows) != d:
        raise ValueError(
            f"tensor_divergence expects a {d}x{d} layout (d = number of spatial "
            f"axes {spatial!r}); got {len(rows)} rows"
        )
    cols: list[Array] = []
    for i, row in enumerate(rows):
        row = list(row)
        if len(row) != d:
            raise ValueError(
                f"row {i} of sigma_names must have {d} entries, got {len(row)}"
            )
        acc: Array | None = None
        for j in range(d):
            term = derivative(state, row[j], axis=spatial[j], order=1)
            acc = term if acc is None else acc + term
        assert acc is not None
        cols.append(acc)
    return jnp.stack(cols, axis=-1)


def tensor_double_dot(a: Array, b: Array) -> Array:
    r"""Frobenius double contraction :math:`A:B = \sum_{ij} A_{ij} B_{ij}`.

    Operates on the trailing two axes of two equally-shaped tensor batches
    (e.g. ``stress : strain_rate`` for viscous dissipation). Returns shape ``(B,)``.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.shape != b.shape:
        raise ValueError(
            f"tensor_double_dot requires equal shapes; got {tuple(a.shape)} "
            f"and {tuple(b.shape)}"
        )
    if a.ndim < 2:
        raise ValueError("tensor_double_dot requires at least 2 trailing axes")
    return (a * b).sum(axis=(-2, -1))


# ----------------------------------------------------------------------
# Batched linear-algebra on the trailing (d, d) block (bit-identical twin of
# :mod:`omnibias.fields.torch.ops.tensor`).
# ----------------------------------------------------------------------
def _require_square(a: Array, fn: str) -> None:
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(
            f"{fn} requires a square trailing block (..., d, d); got shape "
            f"{tuple(a.shape)}"
        )


def tensor_transpose(a: Array) -> Array:
    r"""Transpose the trailing two axes: :math:`A^\top`, shape ``(..., d, d)``."""
    a = jnp.asarray(a)
    _require_square(a, "tensor_transpose")
    return jnp.swapaxes(a, -1, -2)


def tensor_matmul(a: Array, b: Array) -> Array:
    r"""Batched matrix product :math:`(AB)_{ik} = A_{ij} B_{jk}`, shape ``(..., d, d)``."""
    return jnp.matmul(a, b)


def tensor_trace(a: Array) -> Array:
    r"""Trace of the trailing block :math:`\operatorname{tr}A = A_{ii}`, shape ``(...,)``."""
    a = jnp.asarray(a)
    _require_square(a, "tensor_trace")
    return jnp.trace(a, axis1=-2, axis2=-1)


def tensor_determinant(a: Array) -> Array:
    r"""Determinant of the trailing block :math:`\det A`, shape ``(...,)``."""
    a = jnp.asarray(a)
    _require_square(a, "tensor_determinant")
    return jnp.linalg.det(a)


def tensor_inverse(a: Array) -> Array:
    r"""Inverse of the trailing block :math:`A^{-1}`, shape ``(..., d, d)``."""
    a = jnp.asarray(a)
    _require_square(a, "tensor_inverse")
    return jnp.linalg.inv(a)


def tensor_cofactor(a: Array) -> Array:
    r"""Cofactor matrix :math:`\operatorname{cof}A = (\det A)\,A^{-\top}`, shape ``(..., d, d)``.

    Satisfies ``A @ cof(A)^T = det(A) I`` (the adjugate/Cramer identity); used
    for Nanson's formula and the derivative of the Jacobian ``dJ/dF = cof(F)``.
    """
    a = jnp.asarray(a)
    _require_square(a, "tensor_cofactor")
    det = jnp.linalg.det(a)
    inv_t = jnp.swapaxes(jnp.linalg.inv(a), -1, -2)
    return det[..., None, None] * inv_t


__all__ = [
    "tensor_cofactor",
    "tensor_determinant",
    "tensor_divergence",
    "tensor_double_dot",
    "tensor_inverse",
    "tensor_matmul",
    "tensor_trace",
    "tensor_transpose",
]

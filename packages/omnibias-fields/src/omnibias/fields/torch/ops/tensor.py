# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Divergence of a second-order tensor field (torch).

For solid mechanics / elasticity and the conservative form of momentum balance:
given a stress tensor :math:`\\sigma_{ij}`, the row-wise divergence

.. math::

    (\\nabla\\cdot\\sigma)_i = \\partial_j \\sigma_{ij}

is the body-force-balancing vector. This is the flat (Cartesian) divergence; the
covariant version with Christoffel corrections lives in ``omnibias-geometry``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import derivative
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def tensor_divergence(
    state: FieldState,
    sigma_names: Sequence[Sequence[str]],
) -> Tensor:
    r"""Row-wise divergence :math:`(\nabla\cdot\sigma)_i = \partial_j \sigma_{ij}`.

    Parameters
    ----------
    state
        Field state carrying the stress components.
    sigma_names
        A ``(d, d)`` nested sequence of component names laying out
        :math:`\sigma_{ij}` (row ``i`` is the output index; column ``j`` is the
        spatial derivative axis). ``d`` must equal the number of spatial axes.

    Returns
    -------
    Tensor
        The divergence vector of shape ``(B, d)``.
    """
    spatial = state.coordinate_spec.spatial_axes
    d = len(spatial)
    rows = list(sigma_names)
    if len(rows) != d:
        raise ValueError(
            f"tensor_divergence expects a {d}x{d} layout (d = number of spatial "
            f"axes {spatial!r}); got {len(rows)} rows"
        )
    cols: list[Tensor] = []
    for i, row in enumerate(rows):
        row = list(row)
        if len(row) != d:
            raise ValueError(
                f"row {i} of sigma_names must have {d} entries, got {len(row)}"
            )
        acc: Tensor | None = None
        for j in range(d):
            term = derivative(state, row[j], axis=spatial[j], order=1)
            acc = term if acc is None else acc + term
        assert acc is not None
        cols.append(acc)
    return torch.stack(cols, dim=-1)


def tensor_double_dot(a: Tensor, b: Tensor) -> Tensor:
    r"""Frobenius double contraction :math:`A:B = \sum_{ij} A_{ij} B_{ij}`.

    Operates on the trailing two axes of two equally-shaped tensor batches
    (e.g. ``stress : strain_rate`` for viscous dissipation). Returns shape ``(B,)``.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"tensor_double_dot requires equal shapes; got {tuple(a.shape)} "
            f"and {tuple(b.shape)}"
        )
    if a.ndim < 2:
        raise ValueError("tensor_double_dot requires at least 2 trailing axes")
    return (a * b).sum(dim=(-2, -1))


# ----------------------------------------------------------------------
# Batched linear-algebra on the trailing (d, d) block.
# These are plain tensor ops (no FieldState); they underpin the
# finite-strain kinematics in :mod:`omnibias.fields.torch.ops.finite_strain`.
# ----------------------------------------------------------------------
def _require_square(a: Tensor, fn: str) -> None:
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(
            f"{fn} requires a square trailing block (..., d, d); got shape "
            f"{tuple(a.shape)}"
        )


def tensor_transpose(a: Tensor) -> Tensor:
    r"""Transpose the trailing two axes: :math:`A^\top`, shape ``(..., d, d)``."""
    _require_square(a, "tensor_transpose")
    return a.transpose(-1, -2)


def tensor_matmul(a: Tensor, b: Tensor) -> Tensor:
    r"""Batched matrix product :math:`(AB)_{ik} = A_{ij} B_{jk}`, shape ``(..., d, d)``."""
    return torch.matmul(a, b)


def tensor_trace(a: Tensor) -> Tensor:
    r"""Trace of the trailing block :math:`\operatorname{tr}A = A_{ii}`, shape ``(...,)``."""
    _require_square(a, "tensor_trace")
    return torch.diagonal(a, dim1=-2, dim2=-1).sum(dim=-1)


def tensor_determinant(a: Tensor) -> Tensor:
    r"""Determinant of the trailing block :math:`\det A`, shape ``(...,)``."""
    _require_square(a, "tensor_determinant")
    return torch.linalg.det(a)


def tensor_inverse(a: Tensor) -> Tensor:
    r"""Inverse of the trailing block :math:`A^{-1}`, shape ``(..., d, d)``."""
    _require_square(a, "tensor_inverse")
    return torch.linalg.inv(a)


def tensor_cofactor(a: Tensor) -> Tensor:
    r"""Cofactor matrix :math:`\operatorname{cof}A = (\det A)\,A^{-\top}`, shape ``(..., d, d)``.

    Satisfies ``A @ cof(A)^T = det(A) I`` (the adjugate/Cramer identity); used
    for Nanson's formula and the derivative of the Jacobian ``dJ/dF = cof(F)``.
    """
    _require_square(a, "tensor_cofactor")
    det = torch.linalg.det(a)
    inv_t = torch.linalg.inv(a).transpose(-1, -2)
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

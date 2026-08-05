# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hermitian-operator projection helpers (torch backend).

In quantum mechanics every physical observable is represented by a
Hermitian operator :math:`O = O^\dagger`. When the network learns a
matrix-valued field (e.g. a learnable Hamiltonian, a tight-binding
hopping matrix, or a metric), Hermiticity is naturally a *constraint*
rather than something the optimiser hits by chance.

For v0.0.1 we expose two ways to enforce Hermiticity:

- :func:`hermitian_projection` -- a pure-Python helper that returns the
  Hermitian part of a square matrix, :math:`O_H = (O + O^\dagger)/2`.
- :func:`hermiticity_loss` -- a soft :math:`L^2` loss
  :math:`||O - O^\dagger||_F^2`.

A future v0.0.2 will add a :class:`HermitianOperatorField`
:class:`FieldBase` wrapper that takes a network producing the upper
triangular entries and exposes the full Hermitian matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor


def hermitian_projection(matrix: Tensor) -> Tensor:
    r"""Return the Hermitian part of a complex matrix.

    For a complex matrix :math:`O`, the Hermitian part is
    :math:`(O + O^\dagger)/2`. For a real matrix the operation reduces
    to :math:`(O + O^T)/2`.

    Parameters
    ----------
    matrix
        Tensor of shape ``(..., N, N)``. Complex-valued tensors are
        conjugated before transposing; real-valued tensors are simply
        transposed (the conjugate is a no-op).

    Returns
    -------
    Tensor
        Same shape as input. For complex inputs ``out.conj().mH ==
        out`` exactly; for real inputs ``out.T == out``.

    Raises
    ------
    ValueError
        If the last two dimensions don't match.

    Examples
    --------
    >>> import torch
    >>> M = torch.tensor([[1.0, 2.0], [4.0, 3.0]])
    >>> hermitian_projection(M)
    tensor([[1., 3.],
            [3., 3.]])
    """
    if matrix.dim() < 2:
        raise ValueError(
            f"matrix must have at least 2 dimensions; got shape {tuple(matrix.shape)}"
        )
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"last two dims must match for a square matrix; got "
            f"{matrix.shape[-2]} x {matrix.shape[-1]}"
        )
    if matrix.is_complex():
        return 0.5 * (matrix + matrix.conj().transpose(-1, -2))
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def hermiticity_loss(matrix: Tensor) -> Tensor:
    r"""Soft loss :math:`||O - O^\dagger||_F^2 / ||O||_F^2`.

    Returns a non-negative scalar tensor; zero exactly when the input
    is Hermitian.

    Parameters
    ----------
    matrix
        Tensor of shape ``(..., N, N)``.

    Returns
    -------
    Tensor
        Zero-dimensional scalar.
    """
    if matrix.dim() < 2:
        raise ValueError(
            f"matrix must have at least 2 dimensions; got shape {tuple(matrix.shape)}"
        )
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"last two dims must match; got "
            f"{matrix.shape[-2]} x {matrix.shape[-1]}"
        )
    if matrix.is_complex():
        diff = matrix - matrix.conj().transpose(-1, -2)
    else:
        diff = matrix - matrix.transpose(-1, -2)
    num = (diff.abs() ** 2).sum()
    den = (matrix.abs() ** 2).sum()
    eps = torch.finfo(num.dtype if num.is_floating_point() else torch.float32).tiny
    return num / (den + eps)


__all__ = ["hermitian_projection", "hermiticity_loss"]

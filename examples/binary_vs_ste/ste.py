# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The straight-through estimator (STE) baseline -- the arm to beat.

The classic binary-neural-network trick (Hubara et al., 2016): the forward is the
hard ``sign``; the backward *pretends* the nonlinearity was the identity, clipped
to ``|z| <= 1`` (the "hard-tanh" STE). This is exactly the crude special case that
the omnibias closed-form ``beta * tanh'(beta z)`` Riccati surrogate generalises
(STE replaces the smooth bump ``s'(z)`` by the box ``1_{|z|<1}``). It is written
here as an explicit :class:`torch.autograd.Function` so the benchmark compares
like with like: identical hard forward, different backward.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["binarize_ste"]


class _BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor) -> Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(z)
        return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:  # type: ignore[no-untyped-def]
        (z,) = ctx.saved_tensors
        return grad_out * (z.abs() <= 1.0).to(grad_out.dtype)


def binarize_ste(z: Tensor) -> Tensor:
    """Hard ``sign(z)`` in ``{-1, +1}`` (``sign(0)=+1``) with the clipped-identity backward.

    The backward multiplies the upstream gradient by the hard-tanh STE mask
    ``1_{|z| <= 1}`` -- the standard straight-through estimator. Matches the hard
    forward of :func:`omnibias.binary.torch.ops.binarize` so only the backward
    distinguishes this baseline from the omnibias surrogates.
    """
    return _BinarizeSTE.apply(z)

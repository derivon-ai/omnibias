# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Torch smooth-surrogate towers / jets and a curvature-aware quantizer backward.

Bit-identical twin of :mod:`omnibias.binary.jax.ops.surrogate`. These go *beyond*
the straight-through estimator: the hard quantizer backward uses the exact
derivative tower / Taylor jet of the smooth ``tanh(beta z)`` surrogate (one
``tanh`` evaluation, any order, via the Riccati polynomials), so the backward
carries the surrogate's local curvature rather than just its slope.
"""

from __future__ import annotations

import torch
from omnibias.binary.torch.ops.quantize import riccati_tanh_derivative
from omnibias.torch.jet import compose_jet
from torch import Tensor

__all__ = [
    "binarize_curvature",
    "curvature_corrected_slope",
    "surrogate_jet",
    "surrogate_tower",
]


def surrogate_tower(z: Tensor, beta: float | Tensor, order: int) -> Tensor:
    r"""Derivative tower ``[s, s', ..., s^(order)]`` of ``s(z) = tanh(beta z)``.

    Row ``k`` is ``d^k/dz^k tanh(beta z) = beta^k * T_k(tanh(beta z))`` where
    ``T_k`` is the Riccati tanh polynomial; a single ``tanh`` evaluation suffices
    regardless of order. Differentiable in both ``z`` and ``beta``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = torch.as_tensor(z)
    t = torch.tanh(beta * z)
    rows = [t]
    for k in range(1, order + 1):
        rows.append((beta**k) * riccati_tanh_derivative(t, order=k))
    return torch.stack(rows, dim=0)


def surrogate_jet(z: Tensor, beta: float | Tensor, order: int) -> Tensor:
    r"""Taylor jet of ``u -> tanh(beta u)`` at ``u = z`` via :func:`compose_jet`.

    Built by composing the exact ``tanh`` derivative tower onto the affine
    pre-activation jet of ``beta * (z + t)``; because the inner map is affine the
    result equals ``tower_to_jet(surrogate_tower(z, beta, order))``. ``jet[1]`` is
    the surrogate slope (the standard backward), ``jet[2]`` its curvature: this is
    the jet-STE backward signal.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = torch.as_tensor(z)
    t = torch.tanh(beta * z)
    sigma_rows = [t] + [riccati_tanh_derivative(t, order=k) for k in range(1, order + 1)]
    sigma_tower = torch.stack(sigma_rows, dim=0)
    u_rows = [beta * z, torch.ones_like(z) * beta]
    u_rows += [torch.zeros_like(z) for _ in range(order - 1)]
    u_jet = torch.stack(u_rows[: order + 1], dim=0)
    return compose_jet(u_jet, sigma_tower)


def curvature_corrected_slope(
    z: Tensor, beta: float | Tensor, *, window: float | None = None
) -> Tensor:
    r"""Windowed-average surrogate slope ``s'(z) + (h^2/6) s'''(z)`` (``h = window``).

    The point slope ``s'(z)`` is replaced by the average of ``s'`` over a
    symmetric window of half-width ``h`` (default ``h = 1/beta``):

    .. math:: \frac{1}{2h}\int_{-h}^{h} s'(z + u)\,du = s'(z) + \frac{h^2}{6} s'''(z) + O(h^4),

    a better effective gradient through the hard step than the point slope alone.
    Reduces to ``s'(z)`` as ``h -> 0``.
    """
    h = (1.0 / beta) if window is None else window
    tower = surrogate_tower(z, beta, order=3)
    return tower[1] + (h * h / 6.0) * tower[3]


class _BinarizeCurvatureFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, beta: Tensor | float) -> Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(z)
        ctx.beta = beta
        return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor | None, None]:  # type: ignore[no-untyped-def]
        (z,) = ctx.saved_tensors
        return grad_out * curvature_corrected_slope(z, ctx.beta), None


def binarize_curvature(z: Tensor, beta: float = 10.0) -> Tensor:
    """Hard ``sign(z)`` with a 2nd-order, curvature-corrected surrogate backward.

    Identical hard forward to :func:`omnibias.binary.torch.ops.binarize`; the
    backward uses :func:`curvature_corrected_slope` (the windowed-average slope)
    instead of the point slope. ``beta`` is a fixed hyperparameter here.
    """
    return _BinarizeCurvatureFn.apply(z, beta)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Surrogate-gradient *kernels* for a single hard ``sign`` forward.

The **temperature-collapse** view: the forward (the hard ``sign``) is the
``beta -> inf`` limit of *any* smooth sigmoidal CDF -- one gate sharpened into a
0/1 step -- so it is base-independent. (That is the feasibility sense, distinct
from the founding bias collapse, which coalesces ``K`` parallel hyperplanes into
one and yields a derivative.) Base-independence leaves the **backward** a free
choice of probability density (a "nascent delta"): the order-1 rung of the
chosen base's bias-collapse derivative tower. This module exposes that choice
directly: one hard ``sign`` forward, a selectable kernel backward.

For ``u = beta * z`` each kernel is the (unit-peak) density of a smooth step:

==========  ======================================  ===========  ============
kernel      shape ``k(u)``                           tail          smooth CDF
==========  ======================================  ===========  ============
``box``     ``1_{|u| <= 1}``                        compact       hardtanh (STE)
``tanh``    ``1 - tanh(u)^2 = sech^2 u``            exponential   tanh
``logistic````4 s(1 - s)``, ``s = sigmoid(u)``      exponential   logistic
``gaussian````exp(-u^2 / 2)``                       gaussian      probit
``cauchy``  ``1 / (1 + u^2)``                       heavy (1/u^2) arctan
==========  ======================================  ===========  ============

Two magnitude conventions:

* ``normalize="peak"`` (default) -- multiply the upstream gradient by ``k(beta z)``
  (peak height 1). ``beta`` then controls only the *width* of the gradient window,
  decoupling sharpness from magnitude. ``box`` at ``beta = 1`` reproduces the STE.
* ``normalize="exact"`` -- multiply by ``beta * k(beta z)`` (a unit-integral nascent
  delta, the *exact* derivative of the smooth surrogate). With ``tanh`` this equals
  :func:`omnibias.binary.torch.ops.binarize`'s ``beta * tanh'(beta z)`` backward.

``box``'s compact support is exactly why the STE starves units with ``|z| > 1/beta``;
the heavy-tailed ``cauchy`` kernel never sends a zero gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor

KERNELS: tuple[str, ...] = ("box", "tanh", "logistic", "gaussian", "cauchy")
NORMALIZATIONS: tuple[str, ...] = ("peak", "exact")


def kernel_value(name: str, u: Tensor) -> Tensor:
    """Unit-peak surrogate density ``k(u)`` (see module table)."""
    if name == "box":
        return (u.abs() <= 1.0).to(u.dtype)
    if name == "tanh":
        t = torch.tanh(u)
        return 1.0 - t * t
    if name == "logistic":
        s = torch.sigmoid(u)
        return 4.0 * s * (1.0 - s)
    if name == "gaussian":
        return torch.exp(-0.5 * u * u)
    if name == "cauchy":
        return 1.0 / (1.0 + u * u)
    raise ValueError(f"unknown kernel {name!r}; choose from {KERNELS}")


def kernel_deriv(name: str, u: Tensor) -> Tensor:
    """Derivative ``dk/du`` of :func:`kernel_value` (for the learnable-``beta`` path)."""
    if name == "box":
        return torch.zeros_like(u)
    if name == "tanh":
        t = torch.tanh(u)
        return -2.0 * t * (1.0 - t * t)
    if name == "logistic":
        s = torch.sigmoid(u)
        return 4.0 * (1.0 - 2.0 * s) * s * (1.0 - s)
    if name == "gaussian":
        return -u * torch.exp(-0.5 * u * u)
    if name == "cauchy":
        d = 1.0 + u * u
        return -2.0 * u / (d * d)
    raise ValueError(f"unknown kernel {name!r}; choose from {KERNELS}")


class _BinarizeKernel(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx, z: Tensor, beta: Tensor, kernel: str, beta_scale: bool, beta_is_param: bool
    ) -> Tensor:
        ctx.save_for_backward(z, beta)
        ctx.kernel = kernel
        ctx.beta_scale = beta_scale
        ctx.beta_is_param = beta_is_param
        return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None, None, None, None]:  # type: ignore[no-untyped-def]
        z, beta = ctx.saved_tensors
        u = beta * z
        k = kernel_value(ctx.kernel, u)
        slope = beta * k if ctx.beta_scale else k
        grad_z = grad_out * slope
        grad_beta: Tensor | None = None
        if ctx.beta_is_param:
            dk = kernel_deriv(ctx.kernel, u)
            dslope = (k + beta * dk * z) if ctx.beta_scale else (dk * z)
            grad_beta = (grad_out * dslope).sum().reshape(())
        return grad_z, grad_beta, None, None, None


def binarize_kernel(
    z: Tensor,
    beta: float | Tensor = 1.0,
    *,
    kernel: str = "tanh",
    normalize: str = "peak",
) -> Tensor:
    """Hard ``sign(z)`` in ``{-1,+1}`` with a selectable surrogate-kernel backward.

    Parameters
    ----------
    z : Tensor
        Pre-activation (or weight) tensor.
    beta : float or Tensor
        Surrogate bandwidth (inverse window width). Pass a gradient-tracking scalar
        :class:`~torch.Tensor` to learn it.
    kernel : str
        One of :data:`KERNELS`.
    normalize : {"peak", "exact"}
        ``"peak"`` multiplies by ``k(beta z)`` (height 1, width ``~1/beta``);
        ``"exact"`` multiplies by ``beta * k(beta z)`` (the exact smooth-surrogate
        derivative).
    """
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"unknown normalize {normalize!r}; choose from {NORMALIZATIONS}")
    beta_is_param = isinstance(beta, Tensor) and beta.requires_grad
    beta_t = beta if isinstance(beta, Tensor) else torch.as_tensor(beta, dtype=z.dtype, device=z.device)
    beta_t = beta_t.to(dtype=z.dtype, device=z.device)
    return _BinarizeKernel.apply(z, beta_t, kernel, normalize == "exact", beta_is_param)


__all__ = ["KERNELS", "NORMALIZATIONS", "binarize_kernel", "kernel_deriv", "kernel_value"]

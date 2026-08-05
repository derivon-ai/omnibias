# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed convolution layers: :class:`cmbConv1d`, :class:`cmbConv2d`.

Each is a standard ``nn.Conv*d`` followed by an
:class:`OperatorBlock` applied per output channel. The activation
operator role determines the *kind* of feature each kernel is
encouraged to learn:

- ``op="grad"`` + ``base="tanh"`` -> Sobel-like edge kernels.
- ``op="laplacian"`` + ``base="gaussian"`` -> Laplacian-of-Gaussian
  blob kernels.
- ``op="derivative"`` + ``block_kwargs={"derivative_order": n}`` ->
  arbitrary closed-form nth activation derivative kernels.
- ``op="band"`` + ``base="gaussian"`` -> Difference-of-Gaussian
  scale-space kernels.
- ``op="integral"`` + ``base="gaussian"`` -> erf-window scale integrals.
- ``op="identity"`` -> standard activation, drop-in for the base nonlinearity.

The OperatorBlock works on channel-last tensors, so we permute
``(B, C, ...)`` <-> ``(B, ..., C)`` around it.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.torch.activations.registry import ActivationSpec
from omnibias.torch.blocks.operator import OperatorBlock, OpName
from omnibias.torch.fastpath.hermite import gaussian_nth_derivative

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn as nn

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_HALF = 0.5


class _CmbConvBase(nn.Module):
    """Shared logic for cmbConv1d / cmbConv2d."""

    conv: nn.Module

    def __init__(
        self,
        op: OpName,
        base: str | ActivationSpec[Tensor],
        out_channels: int,
        block_kwargs: dict[str, Any] | None,
    ) -> None:
        super().__init__()
        self.block = OperatorBlock(
            op=op,
            base=base,
            channels=out_channels,
            **(block_kwargs or {}),
        )

    def _apply_block_channel_last(self, z: Tensor, channel_dim: int) -> Tensor:
        """Permute (B, C, ...) -> (B, ..., C), apply block, permute back."""
        # Move channel axis to last.
        perm = list(range(z.dim()))
        perm.append(perm.pop(channel_dim))
        z_cl = z.permute(*perm).contiguous()
        out_cl = self.block(z_cl)
        # Inverse permutation.
        inv = [0] * z.dim()
        for i, p in enumerate(perm):
            inv[p] = i
        out: Tensor = out_cl.permute(*inv).contiguous()
        return out


class cmbConv1d(_CmbConvBase):
    """``nn.Conv1d`` + per-channel :class:`OperatorBlock`.

    Layout: input ``(B, in_channels, L)`` -> output ``(B, out_channels, L_out)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        op: OpName = "identity",
        base: str | ActivationSpec[Tensor] = "tanh",
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        bias: bool = True,
        block_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(op=op, base=base, out_channels=out_channels, block_kwargs=block_kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv(x)  # (B, out_channels, L_out)
        return self._apply_block_channel_last(z, channel_dim=1)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"op={self.block.op!r}, base={self.block.base_name!r}"
        )


class cmbConv2d(_CmbConvBase):
    """``nn.Conv2d`` + per-channel :class:`OperatorBlock`.

    Layout: input ``(B, in_channels, H, W)`` -> output ``(B, out_channels, H_out, W_out)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        op: OpName = "identity",
        base: str | ActivationSpec[Tensor] = "gaussian",
        stride: int | tuple[int, int] = 1,
        padding: int | str = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        block_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(op=op, base=base, out_channels=out_channels, block_kwargs=block_kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv(x)  # (B, out_channels, H, W)
        return self._apply_block_channel_last(z, channel_dim=1)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"op={self.block.op!r}, base={self.block.base_name!r}"
        )


def analytic_gaussian_taps(kernel_size: int, sigma: Tensor, order: int = 0) -> Tensor:
    r"""Exact cell-integrated Gaussian / derivative-of-Gaussian kernel taps.

    Returns taps of shape ``(*sigma.shape, kernel_size)`` for a continuous-scale
    Gaussian of standard deviation ``sigma`` (pixel units). Tap ``j`` is the
    *integral* of the (``order``-th derivative of the) unit-area Gaussian over the
    cell ``[j - 1/2, j + 1/2]`` -- exact, anti-aliased *area* sampling rather than
    point sampling:

    .. math::
        w_j^{(0)} = \tfrac12\!\left[\operatorname{erf}\tfrac{j+1/2}{\sigma\sqrt2}
                    - \operatorname{erf}\tfrac{j-1/2}{\sigma\sqrt2}\right],\qquad
        w_j^{(n)} = \frac{g^{(n-1)}(b_j) - g^{(n-1)}(a_j)}{\sigma^{n}\sqrt{2\pi}},

    with ``a_j = (j-1/2)/\sigma``, ``b_j = (j+1/2)/\sigma`` and ``g(u)=e^{-u^2/2}``.
    The ``order=0`` smoother is the ``erf`` antiderivative (the gaussian
    ``spec.integral``); ``order>=1`` uses the closed-form Gaussian derivative
    tower (:func:`gaussian_nth_derivative`), so derivative-of-Gaussian *is* the
    omnibias sigma tower. Differentiable in ``sigma`` -- a learnable continuous
    scale, no resampling.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    offsets = torch.arange(
        -(kernel_size // 2), kernel_size // 2 + 1, dtype=sigma.dtype, device=sigma.device
    )
    sig = sigma.unsqueeze(-1)
    a = (offsets - _HALF) / sig
    b = (offsets + _HALF) / sig
    if order == 0:
        return _HALF * (torch.erf(b / _SQRT_2) - torch.erf(a / _SQRT_2))
    gm = gaussian_nth_derivative(b, order - 1) - gaussian_nth_derivative(a, order - 1)
    return gm * (_INV_SQRT_2PI / sig.pow(order))


class _AnalyticGaussianConvBase(nn.Module):
    """Shared learnable-scale handling for the analytic Gaussian conv layers."""

    sigma: Tensor

    def __init__(self, channels: int, sigma_init: float, learnable_sigma: bool) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be > 0, got {sigma_init}")
        self.channels = channels
        sigma0 = torch.full((channels,), float(sigma_init))
        if learnable_sigma:
            self.sigma = nn.Parameter(sigma0)
        else:
            self.register_buffer("sigma", sigma0)

    def _safe_sigma(self) -> Tensor:
        # Keep sigma strictly positive without killing the gradient at typical scales.
        return self.sigma.clamp_min(1e-3)


class AnalyticGaussianConv1d(_AnalyticGaussianConvBase):
    r"""Depthwise 1-D convolution with an exact, continuous-scale Gaussian kernel.

    The per-channel kernel taps are built in closed form from the learnable scale
    ``sigma`` via :func:`analytic_gaussian_taps` (exact cell integration through
    the ``erf`` antiderivative / Gaussian derivative tower), so the layer is
    anti-aliased, sampling-free, and differentiable end-to-end in ``sigma``.
    ``derivative_order=0`` is a Gaussian blur; ``order=1`` a derivative-of-Gaussian
    edge filter; ``order=2`` a second-derivative (Laplacian-of-Gaussian in 1-D).

    Layout: ``(B, channels, L) -> (B, channels, L_out)`` (depthwise/groups).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        derivative_order: int = 0,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        stride: int = 1,
        padding: int | str = "same",
    ) -> None:
        super().__init__(channels, sigma_init, learnable_sigma)
        self.kernel_size = kernel_size
        self.derivative_order = derivative_order
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        taps = analytic_gaussian_taps(self.kernel_size, self._safe_sigma(), self.derivative_order)
        weight = taps.unsqueeze(1).to(x.dtype)  # (C, 1, K) depthwise
        return F.conv1d(x, weight, stride=self.stride, padding=self.padding, groups=self.channels)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"derivative_order={self.derivative_order}"
        )


class AnalyticGaussianConv2d(_AnalyticGaussianConvBase):
    r"""Depthwise 2-D separable analytic Gaussian / derivative-of-Gaussian conv.

    The isotropic Gaussian is separable, so the 2-D kernel is the outer product of
    two 1-D :func:`analytic_gaussian_taps` profiles (one per axis). ``derivative_order``
    is ``(n_y, n_x)``: ``(0, 0)`` is an isotropic blur, ``(1, 0)`` / ``(0, 1)`` are
    gradient-of-Gaussian filters, and a Laplacian-of-Gaussian is the sum of a
    ``(2, 0)`` and a ``(0, 2)`` layer. Learnable continuous scale ``sigma``.

    Layout: ``(B, channels, H, W) -> (B, channels, H_out, W_out)`` (depthwise).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int],
        *,
        derivative_order: int | tuple[int, int] = 0,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        stride: int | tuple[int, int] = 1,
        padding: int | str = "same",
    ) -> None:
        super().__init__(channels, sigma_init, learnable_sigma)
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        ny, nx = (
            (derivative_order, derivative_order)
            if isinstance(derivative_order, int)
            else derivative_order
        )
        self.kernel_size = (kh, kw)
        self.derivative_order = (ny, nx)
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        sigma = self._safe_sigma()
        kh, kw = self.kernel_size
        ny, nx = self.derivative_order
        ty = analytic_gaussian_taps(kh, sigma, ny)  # (C, Kh)
        tx = analytic_gaussian_taps(kw, sigma, nx)  # (C, Kw)
        kernel2d = ty.unsqueeze(-1) * tx.unsqueeze(-2)  # (C, Kh, Kw)
        weight = kernel2d.unsqueeze(1).to(x.dtype)  # (C, 1, Kh, Kw)
        return F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.channels)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"derivative_order={self.derivative_order}"
        )


__all__ = [
    "AnalyticGaussianConv1d",
    "AnalyticGaussianConv2d",
    "analytic_gaussian_taps",
    "cmbConv1d",
    "cmbConv2d",
]

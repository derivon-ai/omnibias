# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed convolution layers: :class:`cmbConv1D`, :class:`cmbConv2D`.

Each is a standard ``keras.layers.Conv*`` (channels-last) followed by an
:class:`OperatorBlock` applied per output channel. Because Keras conv
layers are channels-last by default, the operator block (which acts on
the last axis) applies directly with no permutation. Mirrors
:mod:`omnibias.torch.blocks.conv`.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.core.spec import ActivationSpec
from omnibias.keras.blocks.operator import OperatorBlock, OpName
from omnibias.keras.fastpath.hermite import gaussian_nth_derivative

import keras
from keras import layers, ops

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_HALF = 0.5


class cmbConv1D(layers.Layer):
    """``keras.layers.Conv1D`` + per-channel :class:`OperatorBlock`.

    Layout: input ``(B, L, in_channels)`` -> output ``(B, L_out, filters)``.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        op: OpName = "identity",
        base: str | ActivationSpec[Any] = "tanh",
        strides: int = 1,
        padding: str = "valid",
        dilation_rate: int = 1,
        use_bias: bool = True,
        block_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self._op = op
        self._base_name = base if isinstance(base, str) else base.name
        self._block_kwargs = block_kwargs or {}
        self.conv = layers.Conv1D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
        )
        self.block = OperatorBlock(
            op=op, base=base, channels=filters, **self._block_kwargs
        )

    def call(self, x: Any) -> Any:
        return self.block(self.conv(x))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "op": self._op,
                "base": self._base_name,
                "block_kwargs": self._block_kwargs,
            }
        )
        return config


class cmbConv2D(layers.Layer):
    """``keras.layers.Conv2D`` + per-channel :class:`OperatorBlock`.

    Layout: input ``(B, H, W, in_channels)`` -> output ``(B, H_out, W_out, filters)``.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, int],
        op: OpName = "identity",
        base: str | ActivationSpec[Any] = "gaussian",
        strides: int | tuple[int, int] = 1,
        padding: str = "valid",
        dilation_rate: int | tuple[int, int] = 1,
        use_bias: bool = True,
        block_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self._op = op
        self._base_name = base if isinstance(base, str) else base.name
        self._block_kwargs = block_kwargs or {}
        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
        )
        self.block = OperatorBlock(
            op=op, base=base, channels=filters, **self._block_kwargs
        )

    def call(self, x: Any) -> Any:
        return self.block(self.conv(x))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "op": self._op,
                "base": self._base_name,
                "block_kwargs": self._block_kwargs,
            }
        )
        return config


def analytic_gaussian_taps(kernel_size: int, sigma: Any, order: int = 0) -> Any:
    r"""Exact cell-integrated Gaussian / derivative-of-Gaussian kernel taps.

    Keras twin of :func:`omnibias.torch.blocks.conv.analytic_gaussian_taps`;
    bit-identical closed form. Returns taps of shape ``(*sigma.shape, kernel_size)``
    for a continuous-scale Gaussian of standard deviation ``sigma`` (pixel units).
    Tap ``j`` is the *integral* of the (``order``-th derivative of the) unit-area
    Gaussian over the cell ``[j - 1/2, j + 1/2]`` (exact, anti-aliased area
    sampling). ``order=0`` uses the ``erf`` antiderivative; ``order>=1`` uses the
    closed-form Gaussian derivative tower (:func:`gaussian_nth_derivative`).
    Differentiable in ``sigma``.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    half = kernel_size // 2
    offsets = ops.cast(ops.arange(-half, half + 1), sigma.dtype)
    sig = ops.expand_dims(sigma, -1)
    a = (offsets - _HALF) / sig
    b = (offsets + _HALF) / sig
    if order == 0:
        return _HALF * (ops.erf(b / _SQRT_2) - ops.erf(a / _SQRT_2))
    gm = gaussian_nth_derivative(b, order - 1) - gaussian_nth_derivative(a, order - 1)
    return gm * (_INV_SQRT_2PI / ops.power(sig, order))


class _AnalyticGaussianConvBase(layers.Layer):
    """Shared learnable-scale handling for the analytic Gaussian conv layers."""

    def __init__(
        self,
        channels: int,
        sigma_init: float,
        learnable_sigma: bool,
        **kwargs: Any,
    ) -> None:
        # Follow the global float policy so float64 test runs are not silently
        # truncated to the default float32 compute dtype on layer ``__call__``.
        kwargs.setdefault("dtype", keras.config.floatx())
        super().__init__(**kwargs)
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be > 0, got {sigma_init}")
        self.channels = channels
        self.sigma_init = float(sigma_init)
        self.learnable_sigma = learnable_sigma
        self.sigma = self.add_weight(
            name="sigma",
            shape=(channels,),
            initializer=keras.initializers.Constant(self.sigma_init),
            trainable=learnable_sigma,
            dtype=keras.config.floatx(),
        )

    def _safe_sigma(self) -> Any:
        # ``max(sigma, 1e-3)`` via native (Python-operator / unary) ops only:
        # ``ops.maximum`` with a scalar routes through Keras float32 promotion and
        # would silently truncate float64 inputs.
        return self.sigma + ops.relu(1e-3 - self.sigma)


class AnalyticGaussianConv1D(_AnalyticGaussianConvBase):
    r"""Depthwise 1-D convolution with an exact, continuous-scale Gaussian kernel.

    Keras twin of :class:`omnibias.torch.blocks.conv.AnalyticGaussianConv1d`. The
    per-channel taps are built in closed form from the learnable scale ``sigma``
    via :func:`analytic_gaussian_taps`, so the layer is anti-aliased, sampling-free
    and differentiable end-to-end in ``sigma``. ``derivative_order=0`` is a Gaussian
    blur; ``1`` a derivative-of-Gaussian; ``2`` a 1-D Laplacian-of-Gaussian.

    Layout: channels-last ``(B, L, channels) -> (B, L_out, channels)``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        derivative_order: int = 0,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        strides: int = 1,
        padding: str = "same",
        **kwargs: Any,
    ) -> None:
        super().__init__(channels, sigma_init, learnable_sigma, **kwargs)
        self.kernel_size = kernel_size
        self.derivative_order = derivative_order
        self.strides = strides
        self.padding = padding

    def call(self, x: Any) -> Any:
        taps = analytic_gaussian_taps(self.kernel_size, self._safe_sigma(), self.derivative_order)
        kernel = ops.cast(ops.expand_dims(ops.transpose(taps), -1), x.dtype)  # (K, C, 1)
        return ops.depthwise_conv(x, kernel, strides=self.strides, padding=self.padding)


class AnalyticGaussianConv2D(_AnalyticGaussianConvBase):
    r"""Depthwise 2-D separable analytic Gaussian / derivative-of-Gaussian conv.

    Keras twin of :class:`omnibias.torch.blocks.conv.AnalyticGaussianConv2d`. The
    isotropic Gaussian is separable, so the 2-D kernel is the outer product of two
    1-D :func:`analytic_gaussian_taps` profiles. ``derivative_order`` is ``(n_y, n_x)``;
    a Laplacian-of-Gaussian is the sum of a ``(2, 0)`` and a ``(0, 2)`` layer.

    Layout: channels-last ``(B, H, W, channels) -> (B, H_out, W_out, channels)``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int],
        *,
        derivative_order: int | tuple[int, int] = 0,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        strides: int | tuple[int, int] = 1,
        padding: str = "same",
        **kwargs: Any,
    ) -> None:
        super().__init__(channels, sigma_init, learnable_sigma, **kwargs)
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        ny, nx = (
            (derivative_order, derivative_order)
            if isinstance(derivative_order, int)
            else derivative_order
        )
        self.kernel_size = (kh, kw)
        self.derivative_order = (ny, nx)
        self.strides = strides
        self.padding = padding

    def call(self, x: Any) -> Any:
        sigma = self._safe_sigma()
        kh, kw = self.kernel_size
        ny, nx = self.derivative_order
        ty = analytic_gaussian_taps(kh, sigma, ny)  # (C, Kh)
        tx = analytic_gaussian_taps(kw, sigma, nx)  # (C, Kw)
        kernel2d = ops.expand_dims(ty, -1) * ops.expand_dims(tx, -2)  # (C, Kh, Kw)
        kernel = ops.expand_dims(ops.transpose(kernel2d, (1, 2, 0)), -1)  # (Kh, Kw, C, 1)
        return ops.depthwise_conv(x, ops.cast(kernel, x.dtype), strides=self.strides, padding=self.padding)


__all__ = [
    "AnalyticGaussianConv1D",
    "AnalyticGaussianConv2D",
    "analytic_gaussian_taps",
    "cmbConv1D",
    "cmbConv2D",
]

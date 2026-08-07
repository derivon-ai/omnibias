# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fourier Neural Operator (JAX) -- bit-identical twin of the torch FNO.

Derivatives remain FFT-based and periodic-grid-bound; the closed-form claim
does not transfer. See the torch module docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.jax.activations import JaxActivationSpec, get_activation


@dataclass(frozen=True)
class SpectralConv1d:
    """Complex spectral convolution on the leading ``modes`` Fourier modes."""

    weight_real: Array  # (in, out, modes)
    weight_imag: Array
    modes: int

    def __call__(self, x: Array) -> Array:
        if x.ndim != 3:
            raise ValueError(f"SpectralConv1d expects (B, C, N); got {tuple(x.shape)}")
        n = x.shape[-1]
        x_ft = jnp.fft.rfft(x, axis=-1)
        m = min(self.modes, x_ft.shape[-1])
        w = self.weight_real[..., :m] + 1j * self.weight_imag[..., :m]
        out_ft = jnp.zeros(
            (x.shape[0], w.shape[1], x_ft.shape[-1]), dtype=x_ft.dtype
        )
        contracted = jnp.einsum("bim,iom->bom", x_ft[:, :, :m], w)
        out_ft = out_ft.at[:, :, :m].set(contracted)
        return jnp.fft.irfft(out_ft, n=n, axis=-1)


@dataclass(frozen=True)
class FNO1d:
    """1-D Fourier Neural Operator on a periodic grid (JAX)."""

    lift_w: Array
    lift_b: Array
    proj_w: Array
    proj_b: Array
    spectral: tuple[SpectralConv1d, ...]
    pointwise_w: tuple[Array, ...]
    pointwise_b: tuple[Array, ...]
    spec: JaxActivationSpec

    def __call__(self, u0: Array) -> Array:
        if u0.ndim == 2:
            u0 = u0[..., None]
        h = u0 @ self.lift_w.T + self.lift_b
        h = jnp.transpose(h, (0, 2, 1))
        for sc, pw, pb in zip(
            self.spectral, self.pointwise_w, self.pointwise_b, strict=True
        ):
            h1 = sc(h)
            h2 = jnp.transpose(
                jnp.transpose(h, (0, 2, 1)) @ pw.T + pb, (0, 2, 1)
            )
            h = self.spec.forward(h1 + h2)
        out = jnp.transpose(h, (0, 2, 1)) @ self.proj_w.T + self.proj_b
        return out


def _flatten_fno(
    net: FNO1d,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    leaves: list[Any] = [net.lift_w, net.lift_b, net.proj_w, net.proj_b]
    for sc in net.spectral:
        leaves.extend([sc.weight_real, sc.weight_imag])
    leaves.extend(net.pointwise_w)
    leaves.extend(net.pointwise_b)
    aux = (net.spec, len(net.spectral), net.spectral[0].modes)
    return tuple(leaves), aux


def _unflatten_fno(aux: tuple[Any, ...], leaves: tuple[Any, ...]) -> FNO1d:
    spec, n_layers, modes = aux
    lift_w, lift_b, proj_w, proj_b = leaves[:4]
    idx = 4
    spectral = []
    for _ in range(n_layers):
        spectral.append(
            SpectralConv1d(
                weight_real=leaves[idx], weight_imag=leaves[idx + 1], modes=modes
            )
        )
        idx += 2
    pointwise_w = leaves[idx : idx + n_layers]
    pointwise_b = leaves[idx + n_layers : idx + 2 * n_layers]
    return FNO1d(
        lift_w=lift_w,
        lift_b=lift_b,
        proj_w=proj_w,
        proj_b=proj_b,
        spectral=tuple(spectral),
        pointwise_w=tuple(pointwise_w),
        pointwise_b=tuple(pointwise_b),
        spec=spec,
    )


jax.tree_util.register_pytree_node(FNO1d, _flatten_fno, _unflatten_fno)


def make_fno1d(
    *,
    modes: int = 8,
    width: int = 32,
    n_layers: int = 4,
    in_channels: int = 1,
    out_channels: int = 1,
    base: str | JaxActivationSpec = "gelu",
    seed: int = 0,
    dtype: Any = jnp.float64,
) -> FNO1d:
    """Build a randomly-initialised :class:`FNO1d` (JAX)."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    spec = get_activation(base)
    key = jax.random.PRNGKey(seed)

    def _linear(k: Array, din: int, dout: int) -> tuple[Array, tuple[Array, Array]]:
        k, wk = jax.random.split(k)
        scale = 1.0 / math.sqrt(din)
        return k, (
            jax.random.normal(wk, (dout, din), dtype=dtype) * scale,
            jnp.zeros((dout,), dtype=dtype),
        )

    key, (lift_w, lift_b) = _linear(key, in_channels, width)
    key, (proj_w, proj_b) = _linear(key, width, out_channels)
    spectral = []
    pointwise_w = []
    pointwise_b = []
    scale = 1.0 / (width * width)
    for _ in range(n_layers):
        key, rk, ik, pk = jax.random.split(key, 4)
        spectral.append(
            SpectralConv1d(
                weight_real=scale
                * jax.random.normal(rk, (width, width, modes), dtype=dtype),
                weight_imag=scale
                * jax.random.normal(ik, (width, width, modes), dtype=dtype),
                modes=modes,
            )
        )
        key, (pw, pb) = _linear(key, width, width)
        pointwise_w.append(pw)
        pointwise_b.append(pb)
    return FNO1d(
        lift_w=lift_w,
        lift_b=lift_b,
        proj_w=proj_w,
        proj_b=proj_b,
        spectral=tuple(spectral),
        pointwise_w=tuple(pointwise_w),
        pointwise_b=tuple(pointwise_b),
        spec=spec,
    )


__all__ = [
    "FNO1d",
    "SpectralConv1d",
    "make_fno1d",
]

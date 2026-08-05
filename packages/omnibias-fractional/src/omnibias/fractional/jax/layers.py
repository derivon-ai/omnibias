# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Functional / equinox-style (jax) fractional layers.

Each layer is a registered pytree whose single array leaf ``raw_order`` is the
unconstrained order; the constrained order is ``alpha = lo + (hi - lo)
sigmoid(raw_order)`` (mirroring :class:`omnibias.fractional.torch.order.LearnableOrder`),
so ``jax.grad`` / ``optax`` train the order like any other parameter. Static
configuration (``lo``, ``hi``, ``h`` / ``length`` / ``bc`` / ``real``) rides in the
pytree aux data. Mirrors :mod:`omnibias.fractional.torch.layers`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid as jax_sigmoid
from jax.tree_util import register_pytree_node_class
from omnibias.fractional.jax.ops.fractional import grunwald_letnikov, spectral_fractional
from omnibias.fractional.jax.ops.spectral import spectral_fractional_laplacian


def _raw_from_order(order: float, lo: float, hi: float) -> Array:
    if not (lo < order < hi):
        raise ValueError(f"order {order} must lie in the open interval ({lo}, {hi})")
    p = (order - lo) / (hi - lo)
    return jnp.asarray(math.log(p / (1.0 - p)))


@register_pytree_node_class
class GrunwaldLetnikovLayer:
    r"""Grid Grunwald-Letnikov fractional-derivative layer (learnable order leaf)."""

    def __init__(self, raw_order: Array, *, h: float, lo: float = 0.0, hi: float = 2.0) -> None:
        if h <= 0.0:
            raise ValueError(f"grid spacing h must be > 0, got {h}")
        self.raw_order = raw_order
        self.h = float(h)
        self.lo = float(lo)
        self.hi = float(hi)

    @classmethod
    def from_order(
        cls, order: float = 0.5, *, h: float, lo: float = 0.0, hi: float = 2.0
    ) -> GrunwaldLetnikovLayer:
        return cls(_raw_from_order(order, lo, hi), h=h, lo=lo, hi=hi)

    @property
    def alpha(self) -> Array:
        return self.lo + (self.hi - self.lo) * jax_sigmoid(self.raw_order)

    def __call__(self, f: Array) -> Array:
        out: Array = grunwald_letnikov(f, alpha=self.alpha, h=self.h)
        return out

    def tree_flatten(self) -> tuple[tuple[Array], tuple[float, float, float]]:
        return (self.raw_order,), (self.h, self.lo, self.hi)

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[float, float, float], children: tuple[Array]
    ) -> GrunwaldLetnikovLayer:
        (raw_order,) = children
        h, lo, hi = aux
        return cls(raw_order, h=h, lo=lo, hi=hi)


@register_pytree_node_class
class SpectralFractionalLayer:
    r"""Periodic FFT fractional-derivative layer (real part by default)."""

    def __init__(
        self,
        raw_order: Array,
        *,
        length: float,
        lo: float = 0.0,
        hi: float = 2.0,
        real: bool = True,
    ) -> None:
        if length <= 0.0:
            raise ValueError(f"length must be > 0, got {length}")
        self.raw_order = raw_order
        self.length = float(length)
        self.lo = float(lo)
        self.hi = float(hi)
        self.real = bool(real)

    @classmethod
    def from_order(
        cls,
        order: float = 1.0,
        *,
        length: float,
        lo: float = 0.0,
        hi: float = 2.0,
        real: bool = True,
    ) -> SpectralFractionalLayer:
        return cls(_raw_from_order(order, lo, hi), length=length, lo=lo, hi=hi, real=real)

    @property
    def alpha(self) -> Array:
        return self.lo + (self.hi - self.lo) * jax_sigmoid(self.raw_order)

    def __call__(self, f: Array) -> Array:
        out: Array = spectral_fractional(f, alpha=self.alpha, length=self.length)
        return out.real if self.real else out

    def tree_flatten(self) -> tuple[tuple[Array], tuple[float, float, float, bool]]:
        return (self.raw_order,), (self.length, self.lo, self.hi, self.real)

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[float, float, float, bool], children: tuple[Array]
    ) -> SpectralFractionalLayer:
        (raw_order,) = children
        length, lo, hi, real = aux
        return cls(raw_order, length=length, lo=lo, hi=hi, real=real)


@register_pytree_node_class
class SpectralFractionalLaplacianLayer:
    r"""Two-sided spectral fractional Laplacian ``(-Delta)^{alpha/2}`` layer (DST/DCT)."""

    def __init__(
        self,
        raw_order: Array,
        *,
        length: float,
        bc: str = "dirichlet",
        lo: float = 0.0,
        hi: float = 2.0,
    ) -> None:
        if length <= 0.0:
            raise ValueError(f"length must be > 0, got {length}")
        if bc not in ("dirichlet", "neumann"):
            raise ValueError(f"bc must be 'dirichlet' or 'neumann', got {bc!r}")
        self.raw_order = raw_order
        self.length = float(length)
        self.bc = str(bc)
        self.lo = float(lo)
        self.hi = float(hi)

    @classmethod
    def from_order(
        cls,
        order: float = 1.0,
        *,
        length: float,
        bc: str = "dirichlet",
        lo: float = 0.0,
        hi: float = 2.0,
    ) -> SpectralFractionalLaplacianLayer:
        return cls(_raw_from_order(order, lo, hi), length=length, bc=bc, lo=lo, hi=hi)

    @property
    def alpha(self) -> Array:
        return self.lo + (self.hi - self.lo) * jax_sigmoid(self.raw_order)

    def __call__(self, f: Array) -> Array:
        out: Array = spectral_fractional_laplacian(
            f, alpha=self.alpha, length=self.length, bc=self.bc
        )
        return out

    def tree_flatten(self) -> tuple[tuple[Array], tuple[float, str, float, float]]:
        return (self.raw_order,), (self.length, self.bc, self.lo, self.hi)

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[float, str, float, float], children: tuple[Array]
    ) -> SpectralFractionalLaplacianLayer:
        (raw_order,) = children
        length, bc, lo, hi = aux
        return cls(raw_order, length=length, bc=bc, lo=lo, hi=hi)


__all__ = [
    "GrunwaldLetnikovLayer",
    "SpectralFractionalLaplacianLayer",
    "SpectralFractionalLayer",
]

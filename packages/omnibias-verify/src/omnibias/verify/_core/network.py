# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-neutral feed-forward network description consumed by the verifier.

A :class:`Network` is just an ordered list of layers; each layer is a small
immutable record (affine map or activation tag).  The verifier walks the list and
applies the matching rigorous enclosure, so the same description drives interval,
Taylor-model and branch-and-bound propagation.  Weights are plain Python floats:
the torch / jax frontends (:mod:`omnibias.verify.torch`,
:mod:`omnibias.verify.jax`) convert trained modules into this form.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearLayer:
    """Affine map ``y = W x + b`` (``weight`` is ``out x in``, ``bias`` length ``out``)."""

    weight: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    @property
    def in_features(self) -> int:
        return len(self.weight[0]) if self.weight else 0

    @property
    def out_features(self) -> int:
        return len(self.weight)

    def __post_init__(self) -> None:
        if len(self.bias) != len(self.weight):
            raise ValueError("bias length must equal the number of output rows")
        widths = {len(row) for row in self.weight}
        if len(widths) > 1:
            raise ValueError("all weight rows must have equal length")


@dataclass(frozen=True)
class ReLULayer:
    """Elementwise ``max(0, x)``."""


@dataclass(frozen=True)
class TanhLayer:
    """Elementwise ``tanh`` (smooth; enclosed via the closed-form ``tanh_iv``)."""


@dataclass(frozen=True)
class SigmoidLayer:
    """Elementwise logistic sigmoid (smooth; enclosed via ``sigmoid_iv``)."""


@dataclass(frozen=True)
class GELULayer:
    """Elementwise exact GELU ``x * Phi(x)`` (smooth; Phi via the Gaussian tower)."""


@dataclass(frozen=True)
class MaxPoolLayer:
    """Group-wise ``max``: output ``k`` is ``max(x_i for i in groups[k])``."""

    groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.groups or any(len(g) == 0 for g in self.groups):
            raise ValueError("each max-pool group must be non-empty")


Layer = LinearLayer | ReLULayer | TanhLayer | SigmoidLayer | GELULayer | MaxPoolLayer


@dataclass(frozen=True)
class Network:
    """An ordered feed-forward stack of :data:`Layer` records."""

    layers: tuple[Layer, ...]

    def __init__(self, layers: Sequence[Layer]) -> None:
        object.__setattr__(self, "layers", tuple(layers))

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)


def affine_layer(weight: Sequence[Sequence[float]], bias: Sequence[float]) -> LinearLayer:
    """Build a :class:`LinearLayer` from nested float sequences."""
    return LinearLayer(
        weight=tuple(tuple(float(x) for x in row) for row in weight),
        bias=tuple(float(b) for b in bias),
    )


__all__ = [
    "GELULayer",
    "Layer",
    "LinearLayer",
    "MaxPoolLayer",
    "Network",
    "ReLULayer",
    "SigmoidLayer",
    "TanhLayer",
    "affine_layer",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Block / coordinate exact search algebra (theory 08-07).

A structured block (last linear layer, one OMBU bias, one arrangement
normal) is a sparse direction. The step along that direction is spec
03-12: a Taylor model of ``phi(s)``, optional certified radius, and
``verify=True`` never-worse. Last-layer least squares is exactly
quadratic, so order 2 with remainder bound 0 is exact.

This is a coordinate / block sweep, not a global solver and not CCF
stretch. Bias collapse (``delta -> 0``) supplies the tower. Arrangement
``beta`` is caller-owned; this step is at fixed ``beta``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.line_search import JetLineSearchConfig

BlockKind = Literal["mask", "last_linear", "ombu_bias", "arrangement_W"]


@dataclass(frozen=True)
class BlockSpec:
    """Which coordinates of a flat parameter vector may move."""

    kind: BlockKind = "mask"
    width: int | None = None
    index: int = 0
    slot: int = 0
    n_channels: int | None = None
    k_biases: int | None = None
    n_features: int | None = None
    n_hyperplanes: int | None = None
    offset: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("mask", "last_linear", "ombu_bias", "arrangement_W"):
            raise ValueError(f"unknown block kind {self.kind!r}")
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")
        if self.slot < 0:
            raise ValueError(f"slot must be >= 0, got {self.slot}")
        if self.offset < 0:
            raise ValueError(f"offset must be >= 0, got {self.offset}")
        if self.kind == "last_linear":
            if self.width is None or int(self.width) < 1:
                raise ValueError("last_linear requires width >= 1")
        if self.kind == "ombu_bias":
            if self.n_channels is None or int(self.n_channels) < 1:
                raise ValueError("ombu_bias requires n_channels >= 1")
            if self.k_biases is None or int(self.k_biases) < 1:
                raise ValueError("ombu_bias requires k_biases >= 1")
            if self.index >= int(self.n_channels):
                raise ValueError(
                    f"channel index {self.index} >= n_channels {self.n_channels}"
                )
            if self.slot >= int(self.k_biases):
                raise ValueError(f"slot {self.slot} >= k_biases {self.k_biases}")
        if self.kind == "arrangement_W":
            if self.n_features is None or int(self.n_features) < 1:
                raise ValueError("arrangement_W requires n_features >= 1")
            if self.n_hyperplanes is None or int(self.n_hyperplanes) < 1:
                raise ValueError("arrangement_W requires n_hyperplanes >= 1")
            if self.index >= int(self.n_hyperplanes):
                raise ValueError(
                    f"row index {self.index} >= n_hyperplanes {self.n_hyperplanes}"
                )


def last_linear_block(width: int, *, offset: int = 0) -> BlockSpec:
    """Last-linear slice: ``width`` entries starting at ``offset`` from the tail."""
    return BlockSpec(kind="last_linear", width=int(width), offset=int(offset))


def ombu_bias_block(
    *,
    n_channels: int,
    k_biases: int,
    channel: int = 0,
    slot: int = 0,
    offset: int = 0,
) -> BlockSpec:
    """One OMBU bias entry in a row-major ``(n_channels, K)`` bank."""
    return BlockSpec(
        kind="ombu_bias",
        n_channels=int(n_channels),
        k_biases=int(k_biases),
        index=int(channel),
        slot=int(slot),
        offset=int(offset),
    )


def arrangement_w_block(
    *,
    n_hyperplanes: int,
    n_features: int,
    row: int = 0,
    offset: int = 0,
) -> BlockSpec:
    """One arrangement normal (one row of ``W``), row-major."""
    return BlockSpec(
        kind="arrangement_W",
        n_hyperplanes=int(n_hyperplanes),
        n_features=int(n_features),
        index=int(row),
        offset=int(offset),
    )


def resolve_block_mask(
    n_params: int,
    spec: BlockSpec,
    mask: Sequence[bool] | None = None,
) -> tuple[bool, ...]:
    """Boolean mask of length ``n_params`` for ``spec``."""
    n = int(n_params)
    if n < 1:
        raise ValueError(f"n_params must be >= 1, got {n_params}")
    flags = [False] * n
    if spec.kind == "mask":
        if mask is None:
            raise ValueError("kind='mask' requires a boolean mask")
        if len(mask) != n:
            raise ValueError(f"mask length {len(mask)} != n_params {n}")
        return tuple(bool(v) for v in mask)
    if spec.kind == "last_linear":
        width = int(spec.width or 0)
        start = n - width - int(spec.offset)
        stop = n - int(spec.offset)
        if start < 0 or stop > n or start >= stop:
            raise ValueError(
                f"last_linear slice [{start}, {stop}) is outside 0..{n}"
            )
        for i in range(start, stop):
            flags[i] = True
        return tuple(flags)
    if spec.kind == "ombu_bias":
        loc = int(spec.offset) + int(spec.index) * int(spec.k_biases or 0) + int(spec.slot)
        if loc >= n:
            raise ValueError(f"ombu_bias index {loc} >= n_params {n}")
        flags[loc] = True
        return tuple(flags)
    width = int(spec.n_features or 0)
    start = int(spec.offset) + int(spec.index) * width
    stop = start + width
    if stop > n:
        raise ValueError(f"arrangement_W slice [{start}, {stop}) exceeds n_params {n}")
    for i in range(start, stop):
        flags[i] = True
    return tuple(flags)


def unit_direction_from_mask(
    mask: Sequence[bool],
    probe: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Unit vector supported on ``mask``.

    If ``probe`` is given (typically ``-grad``), it is restricted to the
    mask and normalised. A zero restriction falls back to the first
    active standard-basis vector.
    """
    active = [i for i, flag in enumerate(mask) if flag]
    if not active:
        raise ValueError("block mask is empty")
    direction = [0.0] * len(mask)
    if probe is not None:
        if len(probe) != len(mask):
            raise ValueError(f"probe length {len(probe)} != mask length {len(mask)}")
        restricted = [float(probe[i]) if mask[i] else 0.0 for i in range(len(mask))]
        norm = math.sqrt(sum(v * v for v in restricted))
        if norm > 0.0 and math.isfinite(norm):
            return tuple(v / norm for v in restricted)
    direction[active[0]] = 1.0
    return tuple(direction)


def apply_block_step(
    params: Sequence[float],
    direction: Sequence[float],
    step: float,
) -> tuple[float, ...]:
    """``params + step * direction`` as a flat tuple."""
    if len(params) != len(direction):
        raise ValueError("params and direction must have the same length")
    if not math.isfinite(float(step)):
        raise ValueError(f"step must be finite, got {step!r}")
    return tuple(float(p) + float(step) * float(d) for p, d in zip(params, direction, strict=True))


def default_block_config(*, exact_quadratic: bool = False) -> JetLineSearchConfig:
    """03-12 config. Last-layer LS uses order 2 and a larger explicit radius."""
    if exact_quadratic:
        return JetLineSearchConfig(
            order=2,
            trust_radius=8.0,
            verify=True,
            max_step=8.0,
        )
    return JetLineSearchConfig(verify=True)


__all__ = [
    "BlockKind",
    "BlockSpec",
    "apply_block_step",
    "arrangement_w_block",
    "default_block_config",
    "last_linear_block",
    "ombu_bias_block",
    "resolve_block_mask",
    "unit_direction_from_mask",
]

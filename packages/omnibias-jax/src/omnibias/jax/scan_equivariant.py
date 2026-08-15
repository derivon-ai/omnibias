# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Discrete-orbit equivariant scan (JAX twin; theory 02-08).

Exact steering is gaussian-family only. ``C_L`` is a discrete orbit, not SO(2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.scan import BankSpec
from omnibias.jax.scan import init_bias_scan, scan_response

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class OrientationBank:
    angles: tuple[float, ...]
    steerable_order: int | None = None


def steerable_basis(order: int, dim: int, *, base: str = "gaussian") -> None | tuple[int, int]:
    if str(base).lower() != "gaussian" or dim != 2:
        return None
    return (int(order), 2)


def equivariant_scan_apply(
    x: Array,
    angles: Sequence[float],
    offsets: BankSpec,
    *,
    base: str = "gaussian",
    template: str = "grad",
) -> Array:
    act, off, scales, tmpl, _taps = init_bias_scan(1, offsets, template=template, base=base)
    outs = []
    for angle in angles:
        c, s = math.cos(float(angle)), math.sin(float(angle))
        w = jnp.zeros((x.shape[-1],), dtype=x.dtype)
        w = w.at[0].set(c).at[1].set(s)
        z = (x * w).sum(axis=-1, keepdims=True)
        outs.append(scan_response(z, off, scales, tmpl, act))
    return jnp.stack(outs, axis=-1)


__all__ = [
    "OrientationBank",
    "equivariant_scan_apply",
    "steerable_basis",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Geometry encoding for operator conditioning (JAX twin)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.pinn.domain._core.sdf import SDF, evaluate_sdf


def probe_grid(
    bounds: Sequence[tuple[float, float]],
    *,
    n_per_axis: int = 4,
) -> np.ndarray:
    """Tensor-product probe grid over a bounding box."""
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis must be >= 2, got {n_per_axis}")
    axes = [np.linspace(lo, hi, n_per_axis) for (lo, hi) in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def encode_geometry(
    sdf: SDF,
    probes: np.ndarray | Array,
    *,
    dtype: Any = jnp.float64,
) -> Array:
    probes_np = np.asarray(probes, dtype=float)
    vals = evaluate_sdf(sdf, probes_np)
    return jnp.asarray(vals, dtype=dtype)


def encode_geometry_batch(
    sdfs: Sequence[SDF],
    probes: np.ndarray | Array,
    *,
    dtype: Any = jnp.float64,
) -> Array:
    rows = [encode_geometry(s, probes, dtype=dtype) for s in sdfs]
    return jnp.stack(rows, axis=0)


__all__ = [
    "encode_geometry",
    "encode_geometry_batch",
    "probe_grid",
]

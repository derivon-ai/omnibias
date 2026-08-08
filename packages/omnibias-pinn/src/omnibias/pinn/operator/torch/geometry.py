# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Geometry encoding for operator conditioning (torch).

The standard shape-encoding trick: evaluate an SDF at a fixed set of probe
points and feed the resulting vector to the DeepONet branch as the geometry
head. Reuses :mod:`omnibias.pinn.domain`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from omnibias.pinn.domain._core.sdf import SDF, evaluate_sdf
from torch import Tensor


def probe_grid(
    bounds: Sequence[tuple[float, float]],
    *,
    n_per_axis: int = 4,
) -> np.ndarray:
    """Tensor-product probe grid over a bounding box, shape ``(n_per_axis**d, d)``."""
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis must be >= 2, got {n_per_axis}")
    axes = [
        np.linspace(lo, hi, n_per_axis) for (lo, hi) in bounds
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def encode_geometry(
    sdf: SDF,
    probes: np.ndarray | Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Evaluate ``sdf`` at ``probes`` and return a 1-D torch geometry code."""
    if isinstance(probes, Tensor):
        probes_np = probes.detach().cpu().numpy()
    else:
        probes_np = np.asarray(probes, dtype=float)
    vals = evaluate_sdf(sdf, probes_np)
    return torch.as_tensor(vals, dtype=dtype)


def encode_geometry_batch(
    sdfs: Sequence[SDF],
    probes: np.ndarray | Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Stack geometry codes for a batch of SDFs, shape ``(F, n_probes)``."""
    rows = [encode_geometry(s, probes, dtype=dtype) for s in sdfs]
    return torch.stack(rows, dim=0)


__all__ = [
    "encode_geometry",
    "encode_geometry_batch",
    "probe_grid",
]

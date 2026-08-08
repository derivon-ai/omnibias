# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""FBPINN-style multi-level window field (torch).

Finite-basis PINNs (Moseley et al., 2023) replace a single global MLP with a
*fixed* overlapping window hierarchy. Each subdomain network sees
locally-normalized coordinates and an optional per-level frequency scale, which
is what actually reaches high-frequency content -- distinct from
:class:`~omnibias.pinn.partition.torch.PartitionedField`, whose oblique splits
are learned.

The blend ``u = sum_l w_l u_l`` reuses the partition-of-unity combine pattern;
window weights are raised-cosine bumps that sum to one on the interior of the
domain. Derivatives go through autodiff (dispatch tag ``"partitioned"``).

Honesty: this is a *numerical* mitigation of spectral bias, not a removal of it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.fbpinn import window_centers_1d
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from torch import Tensor


def _raised_cosine_1d(x: Tensor, center: float, half_width: float) -> Tensor:
    """Compact raised-cosine window supported on ``[c-h, c+h]``."""
    z = (x - float(center)) / float(half_width)
    inside = z.abs() < 1.0
    w = 0.5 * (1.0 + torch.cos(torch.pi * z))
    return torch.where(inside, w, torch.zeros_like(w))


class FBPINNField(FieldBase):
    """Multi-window PINN with fixed overlapping windows and local normalization.

    Parameters
    ----------
    coordinate_spec, components
        Shared metadata. The *first spatial axis* is windowed in v1; other
        axes are passed through unchanged (enough for 1-D and space-time).
    n_windows
        Number of overlapping windows on the windowed axis.
    overlap
        Fractional overlap in ``(0, 1)``; default ``0.5``.
    frequency_scales
        Optional per-window frequency multiplier applied to the locally
        normalized coordinate before the sub-network. Defaults to all ``1``.
    hidden, base
        Sub-network width / activation (one-layer fields).
    """

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        n_windows: int = 4,
        overlap: float = 0.5,
        frequency_scales: Sequence[float] | None = None,
        hidden: int = 16,
        base: str = "tanh",
        window_axis: int | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(coordinate_spec=coordinate_spec, components=components)
        if coordinate_spec.domain is None:
            raise ValueError("FBPINNField requires coordinate_spec.domain bounds")
        if window_axis is None:
            # Prefer the first spatial axis.
            ax_name = coordinate_spec.spatial_axes[0]
            ax = coordinate_spec.axis_index(ax_name)
        elif isinstance(window_axis, str):
            ax = coordinate_spec.axis_index(window_axis)
        else:
            ax = int(window_axis)
        self.window_axis = ax
        lo, hi = coordinate_spec.domain[ax]
        centers, half_width = window_centers_1d(lo, hi, n_windows, overlap=overlap)
        self.centers = centers
        self.half_width = half_width
        if frequency_scales is None:
            frequency_scales = tuple(1.0 for _ in range(n_windows))
        if len(frequency_scales) != n_windows:
            raise ValueError(
                f"frequency_scales length {len(frequency_scales)} != n_windows={n_windows}"
            )
        self.frequency_scales = tuple(float(s) for s in frequency_scales)
        # Local-coordinate fields: same component count, same ambient dim.
        self.subfields = nn.ModuleList(
            [
                OneLayerVectorField(
                    coordinate_spec=coordinate_spec,
                    components=components,
                    hidden=hidden,
                    base=base,
                    dtype=dtype,
                )
                for _ in range(n_windows)
            ]
        )
        self._dtype = dtype

    @property
    def n_windows(self) -> int:
        return len(self.centers)

    def window_weights(self, coords: Tensor) -> Tensor:
        """Raised-cosine weights of shape ``(B, n_windows)``, row-normalized."""
        x = coords[:, self.window_axis]
        cols = [
            _raised_cosine_1d(x, c, self.half_width) for c in self.centers
        ]
        w = torch.stack(cols, dim=-1)
        denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return w / denom

    def _local_coords(self, coords: Tensor, window_index: int) -> Tensor:
        """Map the windowed axis to ``[-1, 1]`` about the window centre, then scale."""
        local = coords.clone()
        c = self.centers[window_index]
        local[:, self.window_axis] = (
            (coords[:, self.window_axis] - c) / self.half_width
        ) * self.frequency_scales[window_index]
        return local

    def forward_values(self, coords: Tensor) -> Tensor:
        w = self.window_weights(coords)  # (B, L)
        outs = []
        for i, sub in enumerate(self.subfields):
            local = self._local_coords(coords, i)
            outs.append(sub.forward_values(local))  # (B, C)
        stacked = torch.stack(outs, dim=1)  # (B, L, C)
        return torch.einsum("bl,blc->bc", w, stacked)

    def value_component(self, state, name: str) -> Tensor:  # type: ignore[no-untyped-def]
        ci = self.components.index(name)
        return self.forward_values(state.coords)[:, ci]

    def derivative(self, state, name: str, *, axis: int, order: int = 1) -> Tensor:  # type: ignore[no-untyped-def]
        coords = state.coords.detach().requires_grad_(True)
        u = self.forward_values(coords)[:, self.components.index(name)]
        cur = u
        for _ in range(order):
            (g,) = torch.autograd.grad(
                cur.sum(), coords, create_graph=True, retain_graph=True
            )
            cur = g[:, axis]
        return cur


def build_fbpinn_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    n_windows: int = 4,
    overlap: float = 0.5,
    frequency_scales: Sequence[float] | None = None,
    hidden: int = 16,
    base: str = "tanh",
    dtype: torch.dtype = torch.float64,
) -> FBPINNField:
    """Build an :class:`FBPINNField`."""
    return FBPINNField(
        coordinate_spec=coordinate_spec,
        components=components,
        n_windows=n_windows,
        overlap=overlap,
        frequency_scales=frequency_scales,
        hidden=hidden,
        base=base,
        dtype=dtype,
    )


FBPINNField._omnibias_dispatch = "partitioned"  # type: ignore[attr-defined]
FBPINNField._omnibias_readout_independent = False  # type: ignore[attr-defined]

__all__ = [
    "FBPINNField",
    "build_fbpinn_field",
    "window_centers_1d",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""FBPINN-style multi-level window field (torch).

Finite-basis PINNs (Moseley et al., 2023) replace a single global MLP with a
*fixed* overlapping window hierarchy. Each level tiles the domain with
raised-cosine windows; sub-networks see locally normalised coordinates and an
optional per-window frequency scale. Levels are summed:

.. math:: u(x) = \sum_l \sum_w w_{l,w}(x)\, u_{l,w}(\tilde x_{l,w})

The blend reuses :func:`omnibias.partition.torch.weights.combine` when the
optional ``partition`` extra is installed; otherwise a softmax-normalised
raised-cosine fallback is used (documented in :func:`_blend_outputs`).

Honesty: this is a *numerical* mitigation of spectral bias, not a removal of it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.fbpinn import (
    FBPINNLevelSpec,
    default_multilevel_specs,
    resolve_level_specs,
    window_centers_1d,
)
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from torch import Tensor

if False:  # pragma: no cover
    pass


def _partition_combine(weights: Tensor, region_outputs: Tensor) -> Tensor:
    """Blend window outputs; prefer omnibias.partition when installed."""
    try:
        from omnibias.partition.torch.weights import combine

        return combine(weights, region_outputs)
    except ImportError:
        return torch.einsum("bl,blc->bc", weights, region_outputs)


def _raised_cosine_1d(x: Tensor, center: float, half_width: float) -> Tensor:
    """Compact raised-cosine window supported on ``[c-h, c+h]``."""
    z = (x - float(center)) / float(half_width)
    inside = z.abs() < 1.0
    w = 0.5 * (1.0 + torch.cos(torch.pi * z))
    return torch.where(inside, w, torch.zeros_like(w))


class _FBPINNLevel(nn.Module):
    """One fixed window level inside :class:`FBPINNField`."""

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        spec: FBPINNLevelSpec,
        lo: float,
        hi: float,
        window_axis: int,
        hidden: int,
        base: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        centers, half_width = window_centers_1d(
            lo, hi, spec.n_windows, overlap=spec.overlap
        )
        self.centers = centers
        self.half_width = half_width
        if spec.frequency_scales is None:
            frequency_scales = tuple(1.0 for _ in range(spec.n_windows))
        else:
            frequency_scales = spec.frequency_scales
        self.frequency_scales = frequency_scales
        self.window_axis = int(window_axis)
        self.subfields = nn.ModuleList(
            [
                OneLayerVectorField(
                    coordinate_spec=coordinate_spec,
                    components=components,
                    hidden=hidden,
                    base=base,
                    dtype=dtype,
                )
                for _ in range(spec.n_windows)
            ]
        )

    @property
    def n_windows(self) -> int:
        return len(self.centers)

    def window_weights(self, coords: Tensor) -> Tensor:
        x = coords[:, self.window_axis]
        cols = [_raised_cosine_1d(x, c, self.half_width) for c in self.centers]
        w = torch.stack(cols, dim=-1)
        denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return w / denom

    def _local_coords(self, coords: Tensor, window_index: int) -> Tensor:
        local = coords.clone()
        c = self.centers[window_index]
        local[:, self.window_axis] = (
            (coords[:, self.window_axis] - c) / self.half_width
        ) * self.frequency_scales[window_index]
        return local

    def forward_level(self, coords: Tensor) -> Tensor:
        w = self.window_weights(coords)
        outs = []
        for i, sub in enumerate(self.subfields):
            local = self._local_coords(coords, i)
            outs.append(sub.forward_values(local))
        stacked = torch.stack(outs, dim=1)
        return _partition_combine(w, stacked)


class FBPINNField(FieldBase):
    """Fixed multilevel FBPINN field with overlapping windows per level.

    Parameters
    ----------
    coordinate_spec, components
        Shared metadata. The windowed axis is set by ``window_axis``.
    level_specs
        Explicit per-level geometry. Mutually exclusive with ``n_windows`` /
        ``n_levels`` shorthands.
    n_windows
        Single-level shorthand: one level with this many windows.
    n_levels
        Multilevel shorthand via :func:`default_multilevel_specs`.
    overlap, frequency_scales
        Single-level frequency scales (one per window).
    hidden, base
        Sub-network width / activation.
    window_axis
        Axis index or name to window; defaults to the first spatial axis.
    dtype
        Parameter dtype; defaults to ``torch.get_default_dtype()``.
    """

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        level_specs: Sequence[FBPINNLevelSpec] | None = None,
        n_windows: int | None = None,
        n_levels: int | None = None,
        overlap: float = 0.5,
        frequency_scales: Sequence[float] | None = None,
        hidden: int = 16,
        base: str = "tanh",
        window_axis: int | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(coordinate_spec=coordinate_spec, components=components)
        if coordinate_spec.domain is None:
            raise ValueError("FBPINNField requires coordinate_spec.domain bounds")
        if dtype is None:
            dtype = torch.get_default_dtype()
        if window_axis is None:
            ax_name = coordinate_spec.spatial_axes[0]
            ax = coordinate_spec.axis_index(ax_name)
        elif isinstance(window_axis, str):
            ax = coordinate_spec.axis_index(window_axis)
        else:
            ax = int(window_axis)
        self.window_axis = ax
        lo, hi = coordinate_spec.domain[ax]
        specs = resolve_level_specs(
            n_windows=n_windows,
            overlap=overlap,
            frequency_scales=frequency_scales,
            level_specs=level_specs,
            n_levels=n_levels,
        )
        self.levels = nn.ModuleList(
            [
                _FBPINNLevel(
                    coordinate_spec=coordinate_spec,
                    components=components,
                    spec=spec,
                    lo=lo,
                    hi=hi,
                    window_axis=ax,
                    hidden=hidden,
                    base=base,
                    dtype=dtype,
                )
                for spec in specs
            ]
        )
        self._dtype = dtype

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def n_windows(self) -> int:
        return sum(level.n_windows for level in self.levels)

    def window_weights(self, coords: Tensor, *, level: int = 0) -> Tensor:
        """Raised-cosine weights for one level, row-normalised to one."""
        return self.levels[level].window_weights(coords)

    def _pre_activations(self, coords: Tensor) -> Tensor | None:
        return None

    def forward_values(self, coords: Tensor) -> Tensor:
        coords = coords.to(dtype=self._dtype)
        total = self.levels[0].forward_level(coords)
        for level in self.levels[1:]:
            total = total + level.forward_level(coords)
        return total

    def value_component(self, state, name: str) -> Tensor:  # type: ignore[no-untyped-def]
        ci = self.components.index(name)
        return self.forward_values(state.coords)[:, ci]

    def _grad_along(self, u: Tensor, x: Tensor, axis: int) -> Tensor:
        g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        return g[:, axis]

    def derivative(self, state, name: str, *, axis: int, order: int = 1) -> Tensor:  # type: ignore[no-untyped-def]
        ci = self.components.index(name)
        x = state.coords.to(dtype=self._dtype)
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        u = self.forward_values(x)[:, ci]
        for _ in range(order):
            u = self._grad_along(u, x, axis)
        return u

    def mixed_partial(
        self, state, name: str, axes: tuple[int, ...], orders: tuple[int, ...]
    ) -> Tensor:  # type: ignore[no-untyped-def]
        ci = self.components.index(name)
        x = state.coords.to(dtype=self._dtype)
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        u = self.forward_values(x)[:, ci]
        for a, o in zip(axes, orders, strict=False):
            for _ in range(int(o)):
                u = self._grad_along(u, x, a)
        return u


def build_fbpinn_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    level_specs: Sequence[FBPINNLevelSpec] | None = None,
    n_windows: int | None = None,
    n_levels: int | None = None,
    overlap: float = 0.5,
    frequency_scales: Sequence[float] | None = None,
    hidden: int = 16,
    base: str = "tanh",
    window_axis: int | str | None = None,
    dtype: torch.dtype | None = None,
) -> FBPINNField:
    """Build an :class:`FBPINNField`."""
    return FBPINNField(
        coordinate_spec=coordinate_spec,
        components=components,
        level_specs=level_specs,
        n_windows=n_windows,
        n_levels=n_levels,
        overlap=overlap,
        frequency_scales=frequency_scales,
        hidden=hidden,
        base=base,
        window_axis=window_axis,
        dtype=dtype,
    )


FBPINNField._omnibias_dispatch = "partitioned"  # type: ignore[attr-defined]
FBPINNField._omnibias_readout_independent = False  # type: ignore[attr-defined]

__all__ = [
    "FBPINNField",
    "FBPINNLevelSpec",
    "build_fbpinn_field",
    "default_multilevel_specs",
    "window_centers_1d",
]

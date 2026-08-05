# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rectangular problem domain (a thin wrapper over ``CoordinateSpec``).

A :class:`Domain` pins the axes, per-axis ``(lo, hi)`` bounds, periodicity, and
which axis (if any) is time. It reuses the pure-Python
:class:`omnibias.fields._core.coords.CoordinateSpec` so the metadata is shared
with the field ansaetze rather than re-invented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from omnibias.fields._core.coords import CoordinateSpec


@dataclass(frozen=True)
class Domain:
    """A rectangular (box) domain over named axes.

    Parameters
    ----------
    axes
        Axis names in evaluation order, spatial first then time last, e.g.
        ``("x", "y", "t")``.
    bounds
        One ``(lo, hi)`` pair per axis (required -- sampling needs them).
    periodic
        Per-axis periodicity (or a single bool). Default: no axis periodic.
    time_axis
        Name of the time axis. Defaults to ``"t"`` if present, else steady.
    """

    coordinate_spec: CoordinateSpec = field(compare=True)

    def __init__(
        self,
        axes: Sequence[str],
        bounds: Sequence[tuple[float, float]],
        *,
        periodic: Sequence[bool] | bool | None = None,
        time_axis: str | None | object = ...,
    ) -> None:
        cs = CoordinateSpec(
            axes,
            periodicity=periodic,
            domain=bounds,
            time_axis=time_axis,
        )
        if cs.domain is None:
            raise ValueError("Domain requires explicit (lo, hi) bounds per axis")
        object.__setattr__(self, "coordinate_spec", cs)

    # -- convenience accessors delegating to the coordinate spec ------

    @property
    def axes(self) -> tuple[str, ...]:
        return self.coordinate_spec.axes

    @property
    def ndim(self) -> int:
        return self.coordinate_spec.ndim

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        assert self.coordinate_spec.domain is not None  # enforced in __init__
        return self.coordinate_spec.domain

    @property
    def spatial_axes(self) -> tuple[str, ...]:
        return self.coordinate_spec.spatial_axes

    @property
    def n_spatial(self) -> int:
        return self.coordinate_spec.n_spatial

    @property
    def time_axis(self) -> str | None:
        return self.coordinate_spec.time_axis

    @property
    def is_time_dependent(self) -> bool:
        return self.coordinate_spec.time_axis is not None

    def bound(self, axis: int | str) -> tuple[float, float]:
        return self.bounds[self.coordinate_spec.axis_index(axis)]

    def time_bounds(self) -> tuple[float, float]:
        if self.time_axis is None:
            raise ValueError("domain has no time axis")
        return self.bound(self.time_axis)

    def __repr__(self) -> str:
        return f"Domain(axes={self.axes!r}, bounds={self.bounds!r}, time_axis={self.time_axis!r})"


__all__ = ["Domain"]

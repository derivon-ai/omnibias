# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Problem domain: rectangular box, optionally carved by an SDF.

A :class:`Domain` pins the axes, per-axis ``(lo, hi)`` bounds, periodicity, and
which axis (if any) is time. It reuses the pure-Python
:class:`omnibias.fields._core.coords.CoordinateSpec` so the metadata is shared
with the field ansaetze rather than re-invented.

An optional ``sdf`` (from :mod:`omnibias.pinn.domain`) carves a non-box
interior out of the bounding box. Sampling then uses the SDF-aware helpers in
:mod:`omnibias.pinn.domain._core.sampling`; the bounding box remains the
rejection envelope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from omnibias.fields._core.coords import CoordinateSpec


@dataclass(frozen=True)
class Domain:
    """A problem domain over named axes, optionally restricted by an SDF.

    Parameters
    ----------
    axes
        Axis names in evaluation order, spatial first then time last, e.g.
        ``("x", "y", "t")``.
    bounds
        One ``(lo, hi)`` pair per axis (required -- sampling needs them).
        When ``sdf`` is set these are the bounding-box envelope.
    periodic
        Per-axis periodicity (or a single bool). Default: no axis periodic.
    time_axis
        Name of the time axis. Defaults to ``"t"`` if present, else steady.
    sdf
        Optional signed-distance object (negative inside). Must expose
        ``ndim`` matching the number of *spatial* axes and be callable on
        ``(n, n_spatial)`` arrays. Typically a primitive from
        :mod:`omnibias.pinn.domain`.
    """

    coordinate_spec: CoordinateSpec = field(compare=True)
    sdf: Any = field(default=None, compare=False)

    def __init__(
        self,
        axes: Sequence[str],
        bounds: Sequence[tuple[float, float]],
        *,
        periodic: Sequence[bool] | bool | None = None,
        time_axis: str | None | object = ...,
        sdf: Any = None,
    ) -> None:
        cs = CoordinateSpec(
            axes,
            periodicity=periodic,
            domain=bounds,
            time_axis=time_axis,
        )
        if cs.domain is None:
            raise ValueError("Domain requires explicit (lo, hi) bounds per axis")
        if sdf is not None:
            n_spatial = cs.n_spatial
            sdf_ndim = getattr(sdf, "ndim", None)
            if sdf_ndim is None:
                raise TypeError("sdf must expose an ndim attribute")
            if int(sdf_ndim) != int(n_spatial):
                raise ValueError(
                    f"sdf.ndim={sdf_ndim} must equal n_spatial={n_spatial}"
                )
        object.__setattr__(self, "coordinate_spec", cs)
        object.__setattr__(self, "sdf", sdf)

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

    @property
    def is_sdf(self) -> bool:
        """Whether this domain is carved by a non-box SDF."""
        return self.sdf is not None

    def spatial_bounds(self) -> tuple[tuple[float, float], ...]:
        """``(lo, hi)`` pairs of the spatial axes only."""
        cs = self.coordinate_spec
        return tuple(
            self.bounds[cs.axis_index(ax)] for ax in self.spatial_axes
        )

    def bound(self, axis: int | str) -> tuple[float, float]:
        return self.bounds[self.coordinate_spec.axis_index(axis)]

    def time_bounds(self) -> tuple[float, float]:
        if self.time_axis is None:
            raise ValueError("domain has no time axis")
        return self.bound(self.time_axis)

    def __repr__(self) -> str:
        sdf_tag = f", sdf={type(self.sdf).__name__}" if self.sdf is not None else ""
        return (
            f"Domain(axes={self.axes!r}, bounds={self.bounds!r}, "
            f"time_axis={self.time_axis!r}{sdf_tag})"
        )


__all__ = ["Domain"]

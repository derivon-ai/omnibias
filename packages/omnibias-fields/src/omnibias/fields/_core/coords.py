# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Coordinate specification for typed PINN fields.

A :class:`CoordinateSpec` is a backend-agnostic, pure-Python descriptor of
the input domain of a typed field: which spatial / temporal axes it carries,
which of those are periodic, and the length of each axis. The spec lives in
``omnibias.fields._core`` so torch and jax fields share the same metadata
shape; this is the same pattern as :class:`omnibias.core.spec.ActivationSpec`
driving both backends today.

Two conventions worth remembering:

- Axis names are strings, not integers. Inside the kernel an axis is
  always referred to by ``int`` (its position), but the user-facing API
  passes strings (``"x"``, ``"y"``, ``"z"``, ``"t"``) and the spec is
  responsible for the lookup.
- Time, if present, is the *last* axis. Spatial axes come first. The
  convention is shared with :mod:`omnibias.torch.architectures.pinn` and
  with the existing 2D NS solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


def _normalise_axes(axes: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for a in axes:
        if not isinstance(a, str):
            raise TypeError(f"axis name must be str, got {type(a).__name__}: {a!r}")
        if not a:
            raise ValueError("axis names must be non-empty strings")
        if a in seen:
            raise ValueError(f"duplicate axis name: {a!r}")
        seen.add(a)
        out.append(a)
    return tuple(out)


def _coerce_periodicity(
    periodicity: Sequence[bool] | bool | None,
    n_axes: int,
) -> tuple[bool, ...]:
    if periodicity is None:
        return tuple(False for _ in range(n_axes))
    if isinstance(periodicity, bool):
        return tuple(periodicity for _ in range(n_axes))
    out = tuple(bool(b) for b in periodicity)
    if len(out) != n_axes:
        raise ValueError(
            f"periodicity has length {len(out)} but spec declares {n_axes} axes"
        )
    return out


def _coerce_domain(
    domain: Sequence[tuple[float, float]] | None,
    n_axes: int,
) -> tuple[tuple[float, float], ...] | None:
    if domain is None:
        return None
    out: list[tuple[float, float]] = []
    for i, ab in enumerate(domain):
        if len(ab) != 2:
            raise ValueError(
                f"domain[{i}] must be a (lo, hi) pair, got {ab!r}"
            )
        a, b = float(ab[0]), float(ab[1])
        if not (a < b):
            raise ValueError(
                f"domain[{i}] requires lo < hi, got ({a}, {b})"
            )
        out.append((a, b))
    if len(out) != n_axes:
        raise ValueError(
            f"domain has length {len(out)} but spec declares {n_axes} axes"
        )
    return tuple(out)


@dataclass(frozen=True)
class CoordinateSpec:
    """Frozen descriptor of a PINN field's input axes.

    Parameters
    ----------
    axes
        Names of the input axes in evaluation order. The convention is
        spatial axes first, then time last (e.g. ``("x", "y", "z", "t")``).
    periodicity
        Per-axis booleans indicating whether the axis is periodic. ``None``
        is the default and means *no* axis is periodic. A single bool is
        broadcast over all axes.
    domain
        Optional ``(lo, hi)`` pair per axis. ``None`` means the domain is
        not pinned by the spec (caller manages it). When given,
        ``hi > lo`` is required per axis.
    time_axis
        The name of the time axis. ``None`` means the field is steady (no
        explicit time axis). Defaults to ``"t"`` if present in ``axes``,
        otherwise ``None``.
    """

    axes: tuple[str, ...]
    periodicity: tuple[bool, ...] = field(default=())
    domain: tuple[tuple[float, float], ...] | None = field(default=None)
    time_axis: str | None = None

    def __init__(
        self,
        axes: Sequence[str],
        *,
        periodicity: Sequence[bool] | bool | None = None,
        domain: Sequence[tuple[float, float]] | None = None,
        time_axis: str | None | object = ...,
    ) -> None:
        norm_axes = _normalise_axes(axes)
        norm_period = _coerce_periodicity(periodicity, len(norm_axes))
        norm_domain = _coerce_domain(domain, len(norm_axes))
        if time_axis is ...:
            ta: str | None = "t" if "t" in norm_axes else None
        else:
            ta = time_axis  # type: ignore[assignment]
            if ta is not None and ta not in norm_axes:
                raise ValueError(
                    f"time_axis {ta!r} not in axes {norm_axes!r}"
                )
        object.__setattr__(self, "axes", norm_axes)
        object.__setattr__(self, "periodicity", norm_period)
        object.__setattr__(self, "domain", norm_domain)
        object.__setattr__(self, "time_axis", ta)

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.axes)

    @property
    def spatial_axes(self) -> tuple[str, ...]:
        """All axes except the time axis."""
        if self.time_axis is None:
            return self.axes
        return tuple(a for a in self.axes if a != self.time_axis)

    @property
    def n_spatial(self) -> int:
        """Number of spatial axes."""
        return len(self.spatial_axes)

    def axis_index(self, axis: int | str) -> int:
        """Resolve an axis name or integer to its integer position."""
        if isinstance(axis, int):
            if not (-self.ndim <= axis < self.ndim):
                raise IndexError(
                    f"axis index {axis} out of range for {self.ndim}-dim spec"
                )
            return axis % self.ndim
        if isinstance(axis, str):
            if axis not in self.axes:
                raise KeyError(
                    f"axis {axis!r} not in spec axes {self.axes!r}"
                )
            return self.axes.index(axis)
        raise TypeError(
            f"axis must be int or str, got {type(axis).__name__}: {axis!r}"
        )

    def is_periodic(self, axis: int | str) -> bool:
        return self.periodicity[self.axis_index(axis)]

    def is_spatial(self, axis: int | str) -> bool:
        return self.axes[self.axis_index(axis)] != self.time_axis

    def is_time(self, axis: int | str) -> bool:
        return self.axes[self.axis_index(axis)] == self.time_axis

    def __repr__(self) -> str:
        bits = [f"axes={self.axes!r}"]
        if any(self.periodicity):
            bits.append(f"periodicity={self.periodicity!r}")
        if self.domain is not None:
            bits.append(f"domain={self.domain!r}")
        if self.time_axis is not None:
            bits.append(f"time_axis={self.time_axis!r}")
        return f"CoordinateSpec({', '.join(bits)})"

    def asdict(self) -> dict[str, Any]:
        """Return a plain-dict view (handy for JSON serialisation)."""
        return {
            "axes": list(self.axes),
            "periodicity": list(self.periodicity),
            "domain": [list(d) for d in self.domain] if self.domain is not None else None,
            "time_axis": self.time_axis,
        }


__all__ = ["CoordinateSpec"]

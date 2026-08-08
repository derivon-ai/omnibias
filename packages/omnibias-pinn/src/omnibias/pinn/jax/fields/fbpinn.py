# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""FBPINN-style multi-level window field (JAX twin).

Window geometry is shared with the torch twin via
:mod:`omnibias.pinn._core.fbpinn`; sub-networks are one-layer JAX fields
blended per level by raised-cosine weights and
:func:`omnibias.partition.jax.weights.combine` when available.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.fbpinn import (
    FBPINNLevelSpec,
    default_multilevel_specs,
    resolve_level_specs,
    window_centers_1d,
)
from omnibias.pinn.jax.fields.base import FieldBase
from omnibias.pinn.jax.fields.one_layer import (
    OneLayerVectorField,
    make_one_layer_vector_field,
)

if False:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def _partition_combine(weights: Array, region_outputs: Array) -> Array:
    try:
        from omnibias.partition.jax.weights import combine

        return combine(weights, region_outputs)
    except ImportError:
        return jnp.einsum("bl,blc->bc", weights, region_outputs)


def _raised_cosine_1d(x: Array, center: float, half_width: float) -> Array:
    z = (x - float(center)) / float(half_width)
    inside = jnp.abs(z) < 1.0
    w = 0.5 * (1.0 + jnp.cos(jnp.pi * z))
    return jnp.where(inside, w, jnp.zeros_like(w))


def _deriv_along(fn, axis: int):
    def wrapped(x_row: Array) -> Array:
        g = jax.grad(fn)(x_row)
        return g[axis]

    return wrapped


@dataclass(frozen=True)
class _FBPINNLevel:
    subfields: tuple[OneLayerVectorField, ...]
    centers: tuple[float, ...]
    half_width: float
    frequency_scales: tuple[float, ...]
    window_axis: int

    @property
    def n_windows(self) -> int:
        return len(self.centers)

    def window_weights(self, coords: Array) -> Array:
        x = coords[:, self.window_axis]
        cols = [_raised_cosine_1d(x, c, self.half_width) for c in self.centers]
        w = jnp.stack(cols, axis=-1)
        denom = jnp.maximum(w.sum(axis=-1, keepdims=True), 1e-12)
        return w / denom

    def _local_coords(self, coords: Array, window_index: int) -> Array:
        c = self.centers[window_index]
        local_axis = (
            (coords[:, self.window_axis] - c) / self.half_width
        ) * self.frequency_scales[window_index]
        return coords.at[:, self.window_axis].set(local_axis)

    def forward_level(self, coords: Array) -> Array:
        w = self.window_weights(coords)
        outs = []
        for i, sub in enumerate(self.subfields):
            local = self._local_coords(coords, i)
            outs.append(sub.forward_values(local))
        stacked = jnp.stack(outs, axis=1)
        return _partition_combine(w, stacked)


@dataclass(frozen=True)
class FBPINNField(FieldBase):
    """Fixed multilevel FBPINN field (JAX)."""

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    levels: tuple[_FBPINNLevel, ...]
    window_axis: int = 0

    def _pre_activations(self, coords: Array) -> Array | None:
        return None

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def n_windows(self) -> int:
        return sum(level.n_windows for level in self.levels)

    def window_weights(self, coords: Array, *, level: int = 0) -> Array:
        return self.levels[level].window_weights(coords)

    def forward_values(self, coords: Array) -> Array:
        total = self.levels[0].forward_level(coords)
        for lev in self.levels[1:]:
            total = total + lev.forward_level(coords)
        return total

    def value_component(self, state: Any, name: str) -> Array:
        ci = self.components.index(name)
        return self.forward_values(state.coords)[:, ci]

    def _point_component(self, x_row: Array, ci: int) -> Array:
        return self.forward_values(x_row[None, :])[0, ci]

    def derivative(self, state: FieldState, name: str, *, axis: int, order: int = 1) -> Array:
        ci = self.components.index(name)

        def base_fn(x_row: Array) -> Array:
            return self._point_component(x_row, ci)

        fn = base_fn
        for _ in range(order):
            fn = _deriv_along(fn, axis)
        return jax.vmap(fn)(state.coords)

    def mixed_partial(
        self, state: FieldState, name: str, axes: tuple[int, ...], orders: tuple[int, ...]
    ) -> Array:
        ci = self.components.index(name)

        def base_fn(x_row: Array) -> Array:
            return self._point_component(x_row, ci)

        fn = base_fn
        for a, o in zip(axes, orders, strict=False):
            for _ in range(int(o)):
                fn = _deriv_along(fn, int(a))
        return jax.vmap(fn)(state.coords)


def _level_flatten(level: _FBPINNLevel):
    return level.subfields, (
        level.centers,
        level.half_width,
        level.frequency_scales,
        level.window_axis,
    )


def _level_unflatten(aux, leaves):
    centers, half_width, frequency_scales, window_axis = aux
    return _FBPINNLevel(
        subfields=leaves,
        centers=centers,
        half_width=half_width,
        frequency_scales=frequency_scales,
        window_axis=window_axis,
    )


jax.tree_util.register_pytree_node(_FBPINNLevel, _level_flatten, _level_unflatten)


def _fb_flatten(f: FBPINNField):
    return (f.levels, (f.coordinate_spec, f.components, f.window_axis))


def _fb_unflatten(aux, leaves):
    coordinate_spec, components, window_axis = aux
    obj = FBPINNField.__new__(FBPINNField)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "levels", leaves)
    object.__setattr__(obj, "window_axis", window_axis)
    return obj


jax.tree_util.register_pytree_node(FBPINNField, _fb_flatten, _fb_unflatten)

FBPINNField._omnibias_dispatch = "partitioned"  # type: ignore[attr-defined]
FBPINNField._omnibias_readout_independent = False  # type: ignore[attr-defined]


def make_fbpinn_field(
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
    seed: int = 0,
    dtype: Any = None,
) -> FBPINNField:
    if coordinate_spec.domain is None:
        raise ValueError("FBPINNField requires coordinate_spec.domain bounds")
    if dtype is None:
        dtype = jnp.result_type(jnp.array(1.0))
    if window_axis is None:
        ax_name = coordinate_spec.spatial_axes[0]
        ax = coordinate_spec.axis_index(ax_name)
    elif isinstance(window_axis, str):
        ax = coordinate_spec.axis_index(window_axis)
    else:
        ax = int(window_axis)
    lo, hi = coordinate_spec.domain[ax]
    specs = resolve_level_specs(
        n_windows=n_windows,
        overlap=overlap,
        frequency_scales=frequency_scales,
        level_specs=level_specs,
        n_levels=n_levels,
    )
    levels: list[_FBPINNLevel] = []
    sub_seed = seed
    for spec in specs:
        centers, half_width = window_centers_1d(
            lo, hi, spec.n_windows, overlap=spec.overlap
        )
        if spec.frequency_scales is None:
            scales = tuple(1.0 for _ in range(spec.n_windows))
        else:
            scales = spec.frequency_scales
        subs = tuple(
            make_one_layer_vector_field(
                coordinate_spec=coordinate_spec,
                components=components,
                hidden=hidden,
                base=base,
                seed=sub_seed + i,
                dtype=dtype,
            )
            for i in range(spec.n_windows)
        )
        sub_seed += spec.n_windows
        levels.append(
            _FBPINNLevel(
                subfields=subs,
                centers=centers,
                half_width=half_width,
                frequency_scales=scales,
                window_axis=ax,
            )
        )
    return FBPINNField(
        coordinate_spec=coordinate_spec,
        components=components,
        levels=tuple(levels),
        window_axis=ax,
    )


__all__ = [
    "FBPINNField",
    "FBPINNLevelSpec",
    "default_multilevel_specs",
    "make_fbpinn_field",
    "window_centers_1d",
]

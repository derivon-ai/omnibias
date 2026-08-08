# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""FBPINN-style multi-level window field (JAX twin).

Window geometry is shared with the torch twin via
:func:`~omnibias.pinn.torch.fields.fbpinn.window_centers_1d` (pure arithmetic);
sub-networks are one-layer JAX fields blended by raised-cosine weights.
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
from omnibias.pinn.jax.fields.base import FieldBase
from omnibias.pinn.jax.fields.one_layer import (
    OneLayerVectorField,
    make_one_layer_vector_field,
)
from omnibias.pinn._core.fbpinn import window_centers_1d


def _raised_cosine_1d(x: Array, center: float, half_width: float) -> Array:
    z = (x - float(center)) / float(half_width)
    inside = jnp.abs(z) < 1.0
    w = 0.5 * (1.0 + jnp.cos(jnp.pi * z))
    return jnp.where(inside, w, jnp.zeros_like(w))


@dataclass(frozen=True)
class FBPINNField(FieldBase):
    """Multi-window PINN with fixed overlapping windows (JAX)."""

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    subfields: tuple[OneLayerVectorField, ...]
    centers: tuple[float, ...]
    half_width: float
    frequency_scales: tuple[float, ...]
    window_axis: int = 0

    def _pre_activations(self, coords: Array) -> Array | None:
        return None

    @property
    def n_windows(self) -> int:
        return len(self.centers)

    def window_weights(self, coords: Array) -> Array:
        x = coords[:, self.window_axis]
        cols = [_raised_cosine_1d(x, c, self.half_width) for c in self.centers]
        w = jnp.stack(cols, axis=-1)
        denom = jnp.clip(w.sum(axis=-1, keepdims=True), a_min=1e-12)
        return w / denom

    def _local_coords(self, coords: Array, window_index: int) -> Array:
        c = self.centers[window_index]
        local_axis = (
            (coords[:, self.window_axis] - c) / self.half_width
        ) * self.frequency_scales[window_index]
        return coords.at[:, self.window_axis].set(local_axis)

    def forward_values(self, coords: Array) -> Array:
        w = self.window_weights(coords)
        outs = []
        for i, sub in enumerate(self.subfields):
            local = self._local_coords(coords, i)
            outs.append(sub.forward_values(local))
        stacked = jnp.stack(outs, axis=1)
        return jnp.einsum("bl,blc->bc", w, stacked)

    def value_component(self, state: Any, name: str) -> Array:
        ci = self.components.index(name)
        return self.forward_values(state.coords)[:, ci]


def _fb_flatten(f: FBPINNField):
    return f.subfields, (
        f.coordinate_spec,
        f.components,
        f.centers,
        f.half_width,
        f.frequency_scales,
        f.window_axis,
    )


def _fb_unflatten(aux, leaves):
    coordinate_spec, components, centers, half_width, frequency_scales, window_axis = aux
    obj = FBPINNField.__new__(FBPINNField)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "subfields", leaves)
    object.__setattr__(obj, "centers", centers)
    object.__setattr__(obj, "half_width", half_width)
    object.__setattr__(obj, "frequency_scales", frequency_scales)
    object.__setattr__(obj, "window_axis", window_axis)
    return obj


jax.tree_util.register_pytree_node(FBPINNField, _fb_flatten, _fb_unflatten)

FBPINNField._omnibias_dispatch = "partitioned"  # type: ignore[attr-defined]
FBPINNField._omnibias_readout_independent = False  # type: ignore[attr-defined]


def make_fbpinn_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    n_windows: int = 4,
    overlap: float = 0.5,
    frequency_scales: Sequence[float] | None = None,
    hidden: int = 16,
    base: str = "tanh",
    seed: int = 0,
    dtype: Any = jnp.float64,
) -> FBPINNField:
    if coordinate_spec.domain is None:
        raise ValueError("FBPINNField requires coordinate_spec.domain bounds")
    ax_name = coordinate_spec.spatial_axes[0]
    ax = coordinate_spec.axis_index(ax_name)
    lo, hi = coordinate_spec.domain[ax]
    centers, half_width = window_centers_1d(lo, hi, n_windows, overlap=overlap)
    if frequency_scales is None:
        frequency_scales = tuple(1.0 for _ in range(n_windows))
    subs = tuple(
        make_one_layer_vector_field(
            coordinate_spec=coordinate_spec,
            components=components,
            hidden=hidden,
            base=base,
            seed=seed + i,
            dtype=dtype,
        )
        for i in range(n_windows)
    )
    return FBPINNField(
        coordinate_spec=coordinate_spec,
        components=components,
        subfields=subs,
        centers=centers,
        half_width=half_width,
        frequency_scales=tuple(float(s) for s in frequency_scales),
        window_axis=ax,
    )


__all__ = [
    "FBPINNField",
    "make_fbpinn_field",
    "window_centers_1d",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bloch-periodic cage (JAX twin)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.cage.incompressible import _CageFieldBase
from omnibias.pinn.jax.fields.base import FieldBase, _import_jax_ops


@dataclass(frozen=True)
class BlochPeriodicField(_CageFieldBase):
    r"""JAX twin of :class:`omnibias.qpinn.torch.cage.BlochPeriodicField`."""

    base: FieldBase
    k: Array
    base_group: str
    output_group: str
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    spatial_idx: tuple[int, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec

    def _phase(self, coords: Array) -> Array:
        phase = jnp.zeros(coords.shape[0], dtype=coords.dtype)
        for i, ax in enumerate(self.spatial_idx):
            phase = phase + self.k[i] * coords[..., ax]
        return phase

    def evaluate(self, coords: Array) -> FieldState[Array]:
        coords = jnp.asarray(coords)
        if coords.ndim != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != "
                f"coordinate_spec.ndim {self.coordinate_spec.ndim}"
            )
        phase = self._phase(coords)
        cos_p = jnp.cos(phase)
        sin_p = jnp.sin(phase)
        inner_state = self.base.evaluate(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_jax_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={
                "_cage_inner_state": inner_state,
                "_cos_phase": cos_p,
                "_sin_phase": sin_p,
            },
        )

    def _u_value(self, inner: FieldState) -> tuple[Array, Array]:
        members = self.base.components.group_members(self.base_group)
        u_re = inner.ops.value(inner, members[0])
        u_im = inner.ops.value(inner, members[1])
        return u_re, u_im

    def _u_derivative(
        self, inner: FieldState, *, axis: int, order: int,
    ) -> tuple[Array, Array]:
        members = self.base.components.group_members(self.base_group)
        d_re = inner.ops.derivative(inner, members[0], axis=axis, order=order)
        d_im = inner.ops.derivative(inner, members[1], axis=axis, order=order)
        return d_re, d_im

    def value_component(self, state: FieldState, name: str) -> Array:
        inner = state.extra["_cage_inner_state"]
        cos_p = state.extra["_cos_phase"]
        sin_p = state.extra["_sin_phase"]
        if name in self.passthrough_names:
            return inner.ops.value(inner, name)
        u_re, u_im = self._u_value(inner)
        if name == self.velocity_names[0]:
            return cos_p * u_re - sin_p * u_im
        if name == self.velocity_names[1]:
            return sin_p * u_re + cos_p * u_im
        raise KeyError(name)

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Array:
        if order == 0:
            return self.value_component(state, name)
        if order > 2:
            raise NotImplementedError(
                f"BlochPeriodicField supports derivative orders up to 2 in "
                f"v0.0.1 (got order={order})."
            )
        inner = state.extra["_cage_inner_state"]
        cos_p = state.extra["_cos_phase"]
        sin_p = state.extra["_sin_phase"]
        if name in self.passthrough_names:
            return inner.ops.derivative(inner, name, axis=axis, order=order)
        if name not in self.velocity_names:
            raise KeyError(name)
        try:
            slot = self.spatial_idx.index(axis)
            k_a = float(self.k[slot])
        except ValueError:
            k_a = 0.0

        u_re, u_im = self._u_value(inner)
        du_re, du_im = self._u_derivative(inner, axis=axis, order=1)
        if order == 1:
            if name == self.velocity_names[0]:
                return cos_p * du_re - sin_p * du_im - k_a * (sin_p * u_re + cos_p * u_im)
            return sin_p * du_re + cos_p * du_im + k_a * (cos_p * u_re - sin_p * u_im)

        d2u_re, d2u_im = self._u_derivative(inner, axis=axis, order=2)
        if name == self.velocity_names[0]:
            return (
                cos_p * d2u_re - sin_p * d2u_im
                - 2.0 * k_a * (sin_p * du_re + cos_p * du_im)
                - k_a * k_a * (cos_p * u_re - sin_p * u_im)
            )
        return (
            sin_p * d2u_re + cos_p * d2u_im
            + 2.0 * k_a * (cos_p * du_re - sin_p * du_im)
            - k_a * k_a * (sin_p * u_re + cos_p * u_im)
        )

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        if name in self.passthrough_names:
            inner = state.extra["_cage_inner_state"]
            return inner.ops.mixed_partial(inner, name, axes, orders)
        folded: dict[int, int] = {}
        for a, o in zip(axes, orders, strict=False):
            if o < 1:
                continue
            folded[a] = folded.get(a, 0) + int(o)
        if len(folded) == 1:
            ((ax, total_order),) = folded.items()
            return self.derivative(state, name, axis=ax, order=total_order)
        raise NotImplementedError(
            "BlochPeriodicField only supports single-axis derivatives in v0.0.1; "
            "true mixed partials need the multi-axis Leibniz expansion (planned)."
        )


def make_bloch_periodic_field(
    *,
    base: FieldBase,
    k: Sequence[float] | Array,
    base_group: str = "u",
    output_group: str = "psi",
) -> BlochPeriodicField:
    r"""JAX twin of
    :func:`omnibias.qpinn.torch.cage.make_bloch_periodic_field`."""
    if not base.components.is_group(base_group):
        raise ValueError(
            f"base does not have a wavefunction group {base_group!r}; "
            "build it with omnibias.qpinn.make_psi_components"
        )
    members = base.components.group_members(base_group)
    if len(members) != 2:
        raise ValueError(
            f"wavefunction group {base_group!r} must have exactly 2 "
            f"components (re, im); got {members!r}"
        )
    n_spatial = base.coordinate_spec.n_spatial
    k_arr = jnp.asarray(k, dtype=jnp.float64)
    if k_arr.shape != (n_spatial,):
        raise ValueError(
            f"k must have shape ({n_spatial},) matching coordinate_spec "
            f"spatial axes; got shape {tuple(k_arr.shape)}"
        )
    spatial_idx = tuple(
        base.coordinate_spec.axis_index(a)
        for a in base.coordinate_spec.spatial_axes
    )
    velocity_names = (f"{output_group}_re", f"{output_group}_im")
    cage_components = ComponentSpec(
        velocity_names,
        groups={output_group: velocity_names},
    )
    passthrough = tuple(
        n for n in base.components.names if n not in members
    )
    return BlochPeriodicField(
        base=base,
        k=k_arr,
        base_group=base_group,
        output_group=output_group,
        velocity_names=velocity_names,
        passthrough_names=passthrough,
        spatial_idx=spatial_idx,
        coordinate_spec=base.coordinate_spec,
        components=cage_components,
    )


def _bloch_flatten(f: BlochPeriodicField):
    leaves = (f.base, f.k)
    aux = (
        f.base_group, f.output_group,
        f.velocity_names, f.passthrough_names,
        f.spatial_idx,
        f.coordinate_spec, f.components,
    )
    return leaves, aux


def _bloch_unflatten(aux, leaves):
    base, k = leaves
    (base_group, output_group, velocity_names, passthrough_names,
     spatial_idx, coordinate_spec, components) = aux
    obj = BlochPeriodicField.__new__(BlochPeriodicField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "k", k)
    object.__setattr__(obj, "base_group", base_group)
    object.__setattr__(obj, "output_group", output_group)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "spatial_idx", spatial_idx)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    return obj


jax.tree_util.register_pytree_node(
    BlochPeriodicField, _bloch_flatten, _bloch_unflatten,
)


__all__ = ["BlochPeriodicField", "make_bloch_periodic_field"]

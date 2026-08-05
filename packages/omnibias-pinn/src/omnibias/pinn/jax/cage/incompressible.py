# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Incompressible-flow cage fields for the JAX backend.

JAX twin of :mod:`omnibias.pinn.torch.cage.incompressible`. The fields
here wrap an underlying base :class:`FieldBase`, expose a different
:class:`ComponentSpec` to the outside world (caged velocity components
plus pass-throughs), and route value / derivative / mixed-partial calls
through a projection that uses only the base field's closed-form ops.

All cage fields are pytree-registered so they survive ``jax.grad`` /
``jax.jit``: the base field's leaves carry the trainable parameters,
and the cage's static metadata (component / coordinate spec, names)
travels in the auxiliary tree-def.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.fields.base import FieldBase, _import_jax_ops

if TYPE_CHECKING:  # pragma: no cover
    pass


# ----------------- shared base -------------------------------------


class _CageFieldBase(FieldBase):
    """Common machinery for JAX cage fields."""

    base: FieldBase
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec

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
        inner_state = self.base.evaluate(coords)
        cage_state = FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_jax_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={"_cage_inner_state": inner_state},
        )
        return cage_state

    def __call__(self, coords: Array) -> FieldState[Array]:
        return self.evaluate(coords)

    def value_component(self, state: FieldState, name: str) -> Array:
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return inner.ops.value(inner, name)
        if name in self.velocity_names:
            return self._velocity_value(inner, name)
        raise KeyError(name)

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Array:
        if order == 0:
            return self.value_component(state, name)
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return inner.ops.derivative(inner, name, axis=axis, order=order)
        if name in self.velocity_names:
            return self._velocity_derivative(
                inner, name, axis=axis, order=order,
            )
        raise KeyError(name)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return inner.ops.mixed_partial(inner, name, axes, orders)
        if name in self.velocity_names:
            return self._velocity_mixed(inner, name, axes, orders)
        raise KeyError(name)

    def _velocity_value(self, inner: FieldState, name: str) -> Array:
        raise NotImplementedError

    def _velocity_derivative(
        self, inner: FieldState, name: str, *, axis: int, order: int,
    ) -> Array:
        raise NotImplementedError

    def _velocity_mixed(
        self,
        inner: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        raise NotImplementedError


# ----------------- 2D streamfunction --------------------------------


@dataclass(frozen=True)
class StreamfunctionField(_CageFieldBase):
    """JAX 2D streamfunction cage."""

    base: FieldBase
    psi: str
    velocity_names: tuple[str, str]
    passthrough_names: tuple[str, ...]
    spatial_axes: tuple[str, str]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    x_idx: int
    y_idx: int

    def _vsign_and_axis(self, name: str) -> tuple[float, int]:
        if name == self.velocity_names[0]:
            return 1.0, self.y_idx
        if name == self.velocity_names[1]:
            return -1.0, self.x_idx
        raise KeyError(f"{name!r} not in {self.velocity_names!r}")

    def _velocity_value(self, inner: FieldState, name: str) -> Array:
        sign, partial_axis = self._vsign_and_axis(name)
        d = inner.ops.derivative(
            inner, self.psi, axis=partial_axis, order=1,
        )
        return sign * d

    def _velocity_derivative(
        self, inner: FieldState, name: str, *, axis: int, order: int,
    ) -> Array:
        sign, partial_axis = self._vsign_and_axis(name)
        if axis == partial_axis:
            d = inner.ops.derivative(
                inner, self.psi, axis=axis, order=order + 1,
            )
        else:
            d = inner.ops.mixed_partial(
                inner, self.psi, (partial_axis, axis), (1, order),
            )
        return sign * d

    def _velocity_mixed(
        self,
        inner: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        sign, partial_axis = self._vsign_and_axis(name)
        all_axes = list(axes) + [partial_axis]
        all_orders = list(orders) + [1]
        folded: dict[int, int] = {}
        for a, o in zip(all_axes, all_orders, strict=False):
            folded[a] = folded.get(a, 0) + int(o)
        if not folded:
            return sign * inner.ops.value(inner, self.psi)
        ax = tuple(folded.keys())
        og = tuple(folded[a] for a in ax)
        d = inner.ops.mixed_partial(inner, self.psi, ax, og)
        return sign * d


def make_streamfunction_field(
    *,
    base: FieldBase,
    psi: str = "psi",
    velocity_names: tuple[str, str] = ("u", "v"),
    passthrough_names: tuple[str, ...] = (),
    spatial_axes: tuple[str, str] = ("x", "y"),
) -> StreamfunctionField:
    if base.coordinate_spec.n_spatial != 2:
        raise ValueError(
            f"StreamfunctionField requires 2D spatial domain; got "
            f"{base.coordinate_spec.n_spatial}"
        )
    if not base.components.is_component(psi):
        raise ValueError(f"{psi!r} not in base components")
    for n in passthrough_names:
        if not base.components.is_component(n):
            raise ValueError(f"passthrough {n!r} not in base")
    if len(velocity_names) != 2:
        raise ValueError("velocity_names length 2 required")
    component_names = tuple(velocity_names) + tuple(passthrough_names)
    components = ComponentSpec(
        component_names,
        groups={"velocity": tuple(velocity_names)},
    )
    return StreamfunctionField(
        base=base,
        psi=psi,
        velocity_names=tuple(velocity_names),
        passthrough_names=tuple(passthrough_names),
        spatial_axes=tuple(spatial_axes),
        coordinate_spec=base.coordinate_spec,
        components=components,
        x_idx=base.coordinate_spec.axis_index(spatial_axes[0]),
        y_idx=base.coordinate_spec.axis_index(spatial_axes[1]),
    )


def _sf_flatten(f: StreamfunctionField):
    return (f.base,), (
        f.psi,
        f.velocity_names,
        f.passthrough_names,
        f.spatial_axes,
        f.coordinate_spec,
        f.components,
        f.x_idx,
        f.y_idx,
    )


def _sf_unflatten(aux, leaves):
    base, = leaves
    (
        psi, velocity_names, passthrough_names, spatial_axes,
        coordinate_spec, components, x_idx, y_idx,
    ) = aux
    obj = StreamfunctionField.__new__(StreamfunctionField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "psi", psi)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "spatial_axes", spatial_axes)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "x_idx", x_idx)
    object.__setattr__(obj, "y_idx", y_idx)
    return obj


jax.tree_util.register_pytree_node(
    StreamfunctionField, _sf_flatten, _sf_unflatten,
)


# ----------------- 3D vector potential ------------------------------


@dataclass(frozen=True)
class VectorPotentialField(_CageFieldBase):
    """JAX 3D vector-potential cage."""

    base: FieldBase
    A_components: tuple[str, str, str]
    velocity_names: tuple[str, str, str]
    passthrough_names: tuple[str, ...]
    spatial_axes: tuple[str, str, str]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    x_idx: int
    y_idx: int
    z_idx: int

    def _curl_terms(self, name: str) -> tuple[
        tuple[str, int, float], tuple[str, int, float],
    ]:
        A1, A2, A3 = self.A_components
        x_, y_, z_ = self.x_idx, self.y_idx, self.z_idx
        u, v, w = self.velocity_names
        if name == u:
            return (A3, y_, 1.0), (A2, z_, -1.0)
        if name == v:
            return (A1, z_, 1.0), (A3, x_, -1.0)
        if name == w:
            return (A2, x_, 1.0), (A1, y_, -1.0)
        raise KeyError(name)

    def _velocity_value(self, inner: FieldState, name: str) -> Array:
        (a, ax_a, sa), (b, ax_b, sb) = self._curl_terms(name)
        d_a = inner.ops.derivative(inner, a, axis=ax_a, order=1)
        d_b = inner.ops.derivative(inner, b, axis=ax_b, order=1)
        return sa * d_a + sb * d_b

    def _velocity_derivative(
        self, inner: FieldState, name: str, *, axis: int, order: int,
    ) -> Array:
        (a, ax_a, sa), (b, ax_b, sb) = self._curl_terms(name)
        if axis == ax_a:
            d_a = inner.ops.derivative(inner, a, axis=axis, order=order + 1)
        else:
            d_a = inner.ops.mixed_partial(
                inner, a, (ax_a, axis), (1, order),
            )
        if axis == ax_b:
            d_b = inner.ops.derivative(inner, b, axis=axis, order=order + 1)
        else:
            d_b = inner.ops.mixed_partial(
                inner, b, (ax_b, axis), (1, order),
            )
        return sa * d_a + sb * d_b

    def _velocity_mixed(
        self, inner, name, axes, orders,
    ) -> Array:
        (a, ax_a, sa), (b, ax_b, sb) = self._curl_terms(name)

        def _composed(comp: str, partial_axis: int) -> Array:
            full_axes = list(axes) + [partial_axis]
            full_orders = list(orders) + [1]
            folded: dict[int, int] = {}
            for ax, o in zip(full_axes, full_orders, strict=False):
                folded[ax] = folded.get(ax, 0) + int(o)
            ax_t = tuple(folded.keys())
            og_t = tuple(folded[ax] for ax in ax_t)
            return inner.ops.mixed_partial(inner, comp, ax_t, og_t)

        return sa * _composed(a, ax_a) + sb * _composed(b, ax_b)


def make_vector_potential_field(
    *,
    base: FieldBase,
    A_components: tuple[str, str, str] = ("A1", "A2", "A3"),
    velocity_names: tuple[str, str, str] = ("u", "v", "w"),
    passthrough_names: tuple[str, ...] = (),
    spatial_axes: tuple[str, str, str] = ("x", "y", "z"),
) -> VectorPotentialField:
    if base.coordinate_spec.n_spatial != 3:
        raise ValueError(
            f"VectorPotentialField requires 3D spatial domain; got "
            f"{base.coordinate_spec.n_spatial}"
        )
    for n in A_components:
        if not base.components.is_component(n):
            raise ValueError(f"{n!r} not in base components")
    for n in passthrough_names:
        if not base.components.is_component(n):
            raise ValueError(f"passthrough {n!r} not in base")
    if len(velocity_names) != 3 or len(A_components) != 3:
        raise ValueError("velocity_names and A_components must have length 3")
    component_names = tuple(velocity_names) + tuple(passthrough_names)
    components = ComponentSpec(
        component_names,
        groups={"velocity": tuple(velocity_names)},
    )
    return VectorPotentialField(
        base=base,
        A_components=tuple(A_components),
        velocity_names=tuple(velocity_names),
        passthrough_names=tuple(passthrough_names),
        spatial_axes=tuple(spatial_axes),
        coordinate_spec=base.coordinate_spec,
        components=components,
        x_idx=base.coordinate_spec.axis_index(spatial_axes[0]),
        y_idx=base.coordinate_spec.axis_index(spatial_axes[1]),
        z_idx=base.coordinate_spec.axis_index(spatial_axes[2]),
    )


def _vp_flatten(f: VectorPotentialField):
    return (f.base,), (
        f.A_components,
        f.velocity_names,
        f.passthrough_names,
        f.spatial_axes,
        f.coordinate_spec,
        f.components,
        f.x_idx,
        f.y_idx,
        f.z_idx,
    )


def _vp_unflatten(aux, leaves):
    base, = leaves
    (
        A_components, velocity_names, passthrough_names, spatial_axes,
        coordinate_spec, components, x_idx, y_idx, z_idx,
    ) = aux
    obj = VectorPotentialField.__new__(VectorPotentialField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "A_components", A_components)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "spatial_axes", spatial_axes)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "x_idx", x_idx)
    object.__setattr__(obj, "y_idx", y_idx)
    object.__setattr__(obj, "z_idx", z_idx)
    return obj


jax.tree_util.register_pytree_node(
    VectorPotentialField, _vp_flatten, _vp_unflatten,
)


# ----------------- Helmholtz projection -----------------------------


@dataclass(frozen=True)
class HelmholtzProjectionField(_CageFieldBase):
    """JAX Helmholtz projection cage: ``u = u_pred - grad(phi)``."""

    base: FieldBase
    u_pred_components: tuple[str, ...]
    phi: str
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    spatial_axis_indices: tuple[int, ...]

    def _vel_index(self, name: str) -> int:
        return self.velocity_names.index(name)

    def _velocity_value(self, inner: FieldState, name: str) -> Array:
        i = self._vel_index(name)
        u_i = inner.ops.value(inner, self.u_pred_components[i])
        ax_i = self.spatial_axis_indices[i]
        d_phi = inner.ops.derivative(inner, self.phi, axis=ax_i, order=1)
        return u_i - d_phi

    def _velocity_derivative(
        self, inner: FieldState, name: str, *, axis: int, order: int,
    ) -> Array:
        i = self._vel_index(name)
        d_u = inner.ops.derivative(
            inner, self.u_pred_components[i], axis=axis, order=order,
        )
        ax_i = self.spatial_axis_indices[i]
        if axis == ax_i:
            d_phi = inner.ops.derivative(
                inner, self.phi, axis=axis, order=order + 1,
            )
        else:
            d_phi = inner.ops.mixed_partial(
                inner, self.phi, (ax_i, axis), (1, order),
            )
        return d_u - d_phi

    def _velocity_mixed(
        self, inner, name, axes, orders,
    ) -> Array:
        i = self._vel_index(name)
        d_u = inner.ops.mixed_partial(
            inner, self.u_pred_components[i], axes, orders,
        )
        ax_i = self.spatial_axis_indices[i]
        full_axes = list(axes) + [ax_i]
        full_orders = list(orders) + [1]
        folded: dict[int, int] = {}
        for ax, o in zip(full_axes, full_orders, strict=False):
            folded[ax] = folded.get(ax, 0) + int(o)
        ax_t = tuple(folded.keys())
        og_t = tuple(folded[ax] for ax in ax_t)
        d_phi = inner.ops.mixed_partial(inner, self.phi, ax_t, og_t)
        return d_u - d_phi


def make_helmholtz_projection_field(
    *,
    base: FieldBase,
    u_pred_components: tuple[str, ...],
    phi: str = "phi",
    velocity_names: tuple[str, ...] | None = None,
    passthrough_names: tuple[str, ...] = (),
) -> HelmholtzProjectionField:
    n_spatial = base.coordinate_spec.n_spatial
    if len(u_pred_components) != n_spatial:
        raise ValueError(
            f"u_pred has {len(u_pred_components)} components but "
            f"coordinate spec has {n_spatial} spatial axes"
        )
    for n in u_pred_components:
        if not base.components.is_component(n):
            raise ValueError(f"{n!r} not in base components")
    if not base.components.is_component(phi):
        raise ValueError(f"{phi!r} not in base components")
    for n in passthrough_names:
        if not base.components.is_component(n):
            raise ValueError(f"passthrough {n!r} not in base")
    if velocity_names is None:
        velocity_names = tuple(f"u{i+1}" for i in range(n_spatial))
    if len(velocity_names) != n_spatial:
        raise ValueError(
            f"velocity_names length {len(velocity_names)} != "
            f"n_spatial {n_spatial}"
        )
    component_names = tuple(velocity_names) + tuple(passthrough_names)
    components = ComponentSpec(
        component_names,
        groups={"velocity": tuple(velocity_names)},
    )
    return HelmholtzProjectionField(
        base=base,
        u_pred_components=tuple(u_pred_components),
        phi=phi,
        velocity_names=tuple(velocity_names),
        passthrough_names=tuple(passthrough_names),
        coordinate_spec=base.coordinate_spec,
        components=components,
        spatial_axis_indices=tuple(
            base.coordinate_spec.axis_index(a)
            for a in base.coordinate_spec.spatial_axes
        ),
    )


def _hp_flatten(f: HelmholtzProjectionField):
    return (f.base,), (
        f.u_pred_components,
        f.phi,
        f.velocity_names,
        f.passthrough_names,
        f.coordinate_spec,
        f.components,
        f.spatial_axis_indices,
    )


def _hp_unflatten(aux, leaves):
    base, = leaves
    (
        u_pred_components, phi, velocity_names, passthrough_names,
        coordinate_spec, components, spatial_axis_indices,
    ) = aux
    obj = HelmholtzProjectionField.__new__(HelmholtzProjectionField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "u_pred_components", u_pred_components)
    object.__setattr__(obj, "phi", phi)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "spatial_axis_indices", spatial_axis_indices)
    return obj


jax.tree_util.register_pytree_node(
    HelmholtzProjectionField, _hp_flatten, _hp_unflatten,
)


# ----------------- gauge constraints --------------------------------


def coulomb_gauge_loss(
    field: VectorPotentialField,
    coords: Array,
    *,
    inner_state: FieldState | None = None,
) -> Array:
    if inner_state is None:
        inner_state = field.base.evaluate(coords)
    div_A = inner_state.ops.divergence(inner_state, field.A_components)
    return jnp.mean(div_A ** 2)


def helmholtz_gauge_loss(
    field: HelmholtzProjectionField,
    coords: Array,
    *,
    inner_state: FieldState | None = None,
) -> Array:
    if inner_state is None:
        inner_state = field.base.evaluate(coords)
    lap_phi = inner_state.ops.laplacian(inner_state, field.phi)
    div_u_pred = inner_state.ops.divergence(
        inner_state, field.u_pred_components,
    )
    return jnp.mean((lap_phi - div_u_pred) ** 2)


def is_cage_field(state: FieldState) -> bool:
    return isinstance(state.field, _CageFieldBase)


__all__ = [
    "HelmholtzProjectionField",
    "StreamfunctionField",
    "VectorPotentialField",
    "coulomb_gauge_loss",
    "helmholtz_gauge_loss",
    "is_cage_field",
    "make_helmholtz_projection_field",
    "make_streamfunction_field",
    "make_vector_potential_field",
]

# Marker read by the omnibias-fields backend ops to select the dispatch path
# (inherited by every concrete cage field).
_CageFieldBase._omnibias_dispatch = "cage"

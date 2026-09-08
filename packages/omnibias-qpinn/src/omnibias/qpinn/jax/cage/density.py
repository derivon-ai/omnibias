# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard density-matrix cage: ``rho = G G^dag / Tr(G G^dag)`` (JAX).

JAX twin of :mod:`omnibias.qpinn.torch.cage.density`. The base field's
split-real channels are a generator ``G``; the cage exposes the unique
trace-1 PSD ``rho``. First-order derivatives use the product / quotient
rule; higher orders are refused.

Do not conflate this with founding bias collapse (``delta -> 0``) or
temperature collapse (``beta -> inf``, feasibility sense).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.cage.incompressible import _CageFieldBase
from omnibias.pinn.jax.fields.base import FieldBase, _import_jax_ops
from omnibias.qpinn._core.density import parse_rho_entry_name, rho_entry_names


def _assemble_G(inner: FieldState, group: str, dim: int, *, axis=None, order: int = 0) -> Array:
    rows: list[Array] = []
    for i in range(dim):
        row: list[Array] = []
        for j in range(dim):
            re_name, im_name = rho_entry_names(group, i, j)
            if axis is None or order == 0:
                re_v = inner.ops.value(inner, re_name)
                im_v = inner.ops.value(inner, im_name)
            else:
                re_v = inner.ops.derivative(inner, re_name, axis=axis, order=order)
                im_v = inner.ops.derivative(inner, im_name, axis=axis, order=order)
            row.append(re_v + 1j * im_v)
        rows.append(jnp.stack(row, axis=-1))
    return jnp.stack(rows, axis=-2)


def _rho_from_G(gen: Array) -> Array:
    dag = jnp.conj(jnp.swapaxes(gen, -2, -1))
    gram = gen @ dag
    trace = jnp.real(jnp.einsum("...ii->...", gram))
    eps = jnp.finfo(trace.dtype).tiny
    return gram / jnp.maximum(trace, eps)[..., None, None]


def _drho_from_G(gen: Array, dgen: Array) -> Array:
    dag = jnp.conj(jnp.swapaxes(gen, -2, -1))
    ddag = jnp.conj(jnp.swapaxes(dgen, -2, -1))
    gram = gen @ dag
    dgram = dgen @ dag + gen @ ddag
    trace = jnp.real(jnp.einsum("...ii->...", gram))
    dtrace = jnp.real(jnp.einsum("...ii->...", dgram))
    eps = jnp.finfo(trace.dtype).tiny
    nrm = jnp.maximum(trace, eps)[..., None, None]
    dn = dtrace[..., None, None]
    return (dgram * nrm - gram * dn) / (nrm * nrm)


@dataclass(frozen=True)
class DensityMatrixField(_CageFieldBase):
    """Hard PSD / trace-1 cage for a split-real density matrix (JAX)."""

    base: FieldBase
    rho_group_name: str
    dim: int
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec

    @property
    def _omnibias_readout_independent(self) -> bool:
        return False

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
        gen = _assemble_G(inner_state, self.rho_group_name, self.dim)
        rho = _rho_from_G(gen)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_jax_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={"_cage_inner_state": inner_state, "_rho": rho, "_G": gen},
        )

    def value_component(self, state: FieldState, name: str) -> Array:
        if name in self.passthrough_names:
            inner = state.extra["_cage_inner_state"]
            return inner.ops.value(inner, name)
        part, i, j = parse_rho_entry_name(name, group=self.rho_group_name)
        entry = state.extra["_rho"][..., i, j]
        return jnp.real(entry) if part == "re" else jnp.imag(entry)

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1
    ) -> Array:
        if order == 0:
            return self.value_component(state, name)
        if name in self.passthrough_names:
            inner = state.extra["_cage_inner_state"]
            return inner.ops.derivative(inner, name, axis=axis, order=order)
        if order != 1:
            raise NotImplementedError("DensityMatrixField derivatives are first-order only")
        inner = state.extra["_cage_inner_state"]
        gen = state.extra["_G"]
        dgen = _assemble_G(inner, self.rho_group_name, self.dim, axis=axis, order=1)
        drho = _drho_from_G(gen, dgen)
        part, i, j = parse_rho_entry_name(name, group=self.rho_group_name)
        entry = drho[..., i, j]
        return jnp.real(entry) if part == "re" else jnp.imag(entry)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        if not orders or max(orders) == 0:
            return self.value_component(state, name)
        if len(orders) == 1 and orders[0] == 1:
            return self.derivative(state, name, axis=axes[0], order=1)
        raise NotImplementedError("DensityMatrixField mixed partials are first-order only")


def make_density_matrix_field(
    *, base: FieldBase, group: str = "rho", dim: int = 2
) -> DensityMatrixField:
    """Build a :class:`DensityMatrixField` around a generator field (JAX)."""
    if not base.components.is_group(group):
        raise ValueError(
            f"base does not have a density-matrix group {group!r}; "
            "build it with omnibias.qpinn.make_rho_components"
        )
    members = base.components.group_members(group)
    expected = 2 * dim * dim
    if len(members) != expected:
        raise ValueError(
            f"group {group!r} must have {expected} components; got {len(members)}"
        )
    passthrough = tuple(n for n in base.components.names if n not in members)
    return DensityMatrixField(
        base=base,
        rho_group_name=group,
        dim=dim,
        velocity_names=tuple(members),
        passthrough_names=passthrough,
        coordinate_spec=base.coordinate_spec,
        components=base.components,
    )


def _density_cage_flatten(field: DensityMatrixField):
    return (field.base,), (
        field.rho_group_name,
        field.dim,
        field.velocity_names,
        field.passthrough_names,
        field.coordinate_spec,
        field.components,
    )


def _density_cage_unflatten(aux, leaves):
    (base,) = leaves
    rho_group_name, dim, velocity_names, passthrough_names, coordinate_spec, components = aux
    obj = DensityMatrixField.__new__(DensityMatrixField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "rho_group_name", rho_group_name)
    object.__setattr__(obj, "dim", dim)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    return obj


jax.tree_util.register_pytree_node(
    DensityMatrixField, _density_cage_flatten, _density_cage_unflatten
)


__all__ = ["DensityMatrixField", "make_density_matrix_field"]

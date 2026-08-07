# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""L^2-norm-conservation cage for the JAX backend.

JAX twin of :mod:`omnibias.qpinn.torch.cage.norm`. The cage wraps a
base :class:`FieldBase`, computes the wavefunction's :math:`L^2` norm on
a user-supplied fixed quadrature grid, and divides every value /
derivative by it. The result is a wavefunction with unit :math:`L^2`
norm by construction (to quadrature accuracy).

Pytree registration
~~~~~~~~~~~~~~~~~~~

The cage is a frozen dataclass with three pytree leaves --
``(base, quadrature_coords, quadrature_weights)`` -- so the trainable
parameters of the base field and the (fixed but device-resident)
quadrature data travel together through ``jax.grad`` / ``jax.jit``.
All static metadata lives in the auxiliary tree-def.
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


@dataclass(frozen=True)
class NormConservationField(_CageFieldBase):
    r"""Hard :math:`L^2`-norm conservation for a complex wavefunction (JAX).

    Nonlinear in the readout (``psi / ||psi||``), so this cage declines the
    frozen-feature linear solver.
    """

    base: FieldBase
    quadrature_coords: Array
    quadrature_weights: Array
    psi_group_name: str
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec

    @property
    def _omnibias_readout_independent(self) -> bool:
        return False

    def _compute_norm(self) -> Array:
        re_name, im_name = self.velocity_names
        quad_state = self.base.evaluate(self.quadrature_coords)
        psi_re_q = quad_state.ops.value(quad_state, re_name)
        psi_im_q = quad_state.ops.value(quad_state, im_name)
        density_q = psi_re_q * psi_re_q + psi_im_q * psi_im_q
        norm_sq = jnp.sum(self.quadrature_weights * density_q)
        eps = jnp.finfo(norm_sq.dtype).tiny
        return jnp.sqrt(norm_sq + eps)

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
        norm = self._compute_norm()
        inner_state = self.base.evaluate(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_jax_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={"_cage_inner_state": inner_state, "_norm": norm},
        )

    def value_component(self, state: FieldState, name: str) -> Array:
        inner = state.extra["_cage_inner_state"]
        norm = state.extra["_norm"]
        v = inner.ops.value(inner, name)
        if name in self.velocity_names:
            return v / norm
        if name in self.passthrough_names:
            return v
        raise KeyError(
            f"{name!r} is neither a wavefunction component "
            f"{self.velocity_names!r} nor a passthrough "
            f"{self.passthrough_names!r}"
        )

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Array:
        if order == 0:
            return self.value_component(state, name)
        inner = state.extra["_cage_inner_state"]
        norm = state.extra["_norm"]
        d = inner.ops.derivative(inner, name, axis=axis, order=order)
        if name in self.velocity_names:
            return d / norm
        if name in self.passthrough_names:
            return d
        raise KeyError(name)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        inner = state.extra["_cage_inner_state"]
        norm = state.extra["_norm"]
        d = inner.ops.mixed_partial(inner, name, axes, orders)
        if name in self.velocity_names:
            return d / norm
        if name in self.passthrough_names:
            return d
        raise KeyError(name)


def make_norm_conservation_field(
    *,
    base: FieldBase,
    quadrature_coords: Array,
    quadrature_weights: Array,
    psi_group: str = "psi",
) -> NormConservationField:
    r"""Build a :class:`NormConservationField` (JAX backend).

    Mirrors :func:`omnibias.qpinn.torch.cage.make_norm_conservation_field`.

    Parameters
    ----------
    base
        Base :class:`FieldBase` whose component spec carries a
        wavefunction group with exactly two real components.
    quadrature_coords
        Fixed quadrature grid, shape ``(B_q, D)``.
    quadrature_weights
        Quadrature weights, shape ``(B_q,)``.
    psi_group
        Wavefunction group name. Default ``"psi"``.

    Returns
    -------
    NormConservationField
        Pytree-registered cage field.
    """
    if not base.components.is_group(psi_group):
        raise ValueError(
            f"base does not have a wavefunction group {psi_group!r}; "
            "build it with omnibias.qpinn.make_psi_components"
        )
    members = base.components.group_members(psi_group)
    if len(members) != 2:
        raise ValueError(
            f"wavefunction group {psi_group!r} must have exactly 2 "
            f"components (re, im); got {len(members)}: {members!r}"
        )
    qc = jnp.asarray(quadrature_coords)
    qw = jnp.asarray(quadrature_weights)
    if qc.ndim != 2:
        raise ValueError(
            f"quadrature_coords must be 2D (B_q, D), got shape {tuple(qc.shape)}"
        )
    if qw.ndim != 1:
        raise ValueError(
            f"quadrature_weights must be 1D (B_q,), got shape {tuple(qw.shape)}"
        )
    if qc.shape[0] != qw.shape[0]:
        raise ValueError(
            f"quadrature batch mismatch: coords {qc.shape[0]} vs weights {qw.shape[0]}"
        )
    if qc.shape[-1] != base.coordinate_spec.ndim:
        raise ValueError(
            f"quadrature_coords last dim {qc.shape[-1]} != "
            f"coordinate_spec.ndim {base.coordinate_spec.ndim}"
        )
    if jnp.any(qw < 0):
        raise ValueError("quadrature_weights must be non-negative")
    passthrough = tuple(
        n for n in base.components.names if n not in members
    )
    return NormConservationField(
        base=base,
        quadrature_coords=qc,
        quadrature_weights=qw,
        psi_group_name=psi_group,
        velocity_names=tuple(members),
        passthrough_names=passthrough,
        coordinate_spec=base.coordinate_spec,
        components=base.components,
    )


def norm_loss(
    state: FieldState,
    *,
    group: str = "psi",
    quadrature_weights: Array | None = None,
    target_norm: float = 1.0,
) -> Array:
    r"""Soft norm-conservation loss
    :math:`\big(\int |\psi|^2\,dx - n\big)^2` (JAX twin)."""
    re_name = f"{group}_re"
    im_name = f"{group}_im"
    psi_re = state.ops.value(state, re_name)
    psi_im = state.ops.value(state, im_name)
    density = psi_re * psi_re + psi_im * psi_im
    if quadrature_weights is None:
        norm_sq = jnp.mean(density)
    else:
        if quadrature_weights.shape != density.shape:
            raise ValueError(
                f"quadrature_weights shape {tuple(quadrature_weights.shape)} "
                f"!= density shape {tuple(density.shape)}"
            )
        norm_sq = jnp.sum(quadrature_weights * density)
    return (norm_sq - target_norm) ** 2


# ----- pytree registration ----------------------------------------------

def _norm_cage_flatten(f: NormConservationField):
    leaves = (f.base, f.quadrature_coords, f.quadrature_weights)
    aux = (
        f.psi_group_name,
        f.velocity_names,
        f.passthrough_names,
        f.coordinate_spec,
        f.components,
    )
    return leaves, aux


def _norm_cage_unflatten(aux, leaves):
    base, quadrature_coords, quadrature_weights = leaves
    psi_group_name, velocity_names, passthrough_names, coordinate_spec, components = aux
    obj = NormConservationField.__new__(NormConservationField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "quadrature_coords", quadrature_coords)
    object.__setattr__(obj, "quadrature_weights", quadrature_weights)
    object.__setattr__(obj, "psi_group_name", psi_group_name)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    return obj


jax.tree_util.register_pytree_node(
    NormConservationField, _norm_cage_flatten, _norm_cage_unflatten,
)


__all__ = [
    "NormConservationField",
    "make_norm_conservation_field",
    "norm_loss",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""``PartitionedField`` -- JAX twin of the discontinuity-capturing PINN field.

Bit-parity twin of :mod:`omnibias.pinn.partition.torch.field`. A single smooth
activation network cannot represent a kink / shock; a **partition of unity** of
smooth sub-solutions can. :class:`PartitionedField` wraps

* a soft partition (:func:`omnibias.partition.jax.weights.partition_weights_arrays`
  split gates over the coordinates), and
* one :class:`~omnibias.pinn.jax.fields.OneLayerVectorField` sub-solution per region,

and evaluates the blend

.. math:: u(x) = \sum_l w_l(x)\, u_l(x),

a genuine field that plugs into the existing JAX PINN ops.

Honesty
-------
Derivatives of ``u`` go through the **autodiff product rule** (``jax.grad``), *not*
the closed-form ``sigma``-tower -- the tower does not cover products of sigmoids (a
Faa-di-Bruno-on-products tower is future work). The field therefore sets
``_omnibias_dispatch = "partitioned"`` so the fields ops route it to the autodiff
state-method path (exactly like the spectral / cage fields), never the closed-form
path -- identical routing to the torch twin.

Terminology: the gate's ``beta -> inf`` hardening is the feasibility / temperature
sense of "collapse", distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see
``docs/theory.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def _deriv_along(fn: Callable[[Array], Array], axis: int) -> Callable[[Array], Array]:
    """Return ``x -> d fn / d x_axis`` for a scalar-valued ``fn`` of one point."""

    def d(x_row: Array) -> Array:
        return jax.grad(fn)(x_row)[axis]

    return d


@dataclass(frozen=True)
class PartitionedField(FieldBase):
    r"""Discontinuity-capturing field ``u(x) = sum_l w_l(x) u_l(x)`` (JAX twin).

    Parameters
    ----------
    coordinate_spec, components:
        Shared input-axis / output-channel metadata (all sub-solutions use these).
    subfields:
        The ``2**depth`` region sub-solutions, one per region.
    split_W:
        ``(depth, D)`` oblique split directions of the partition gates.
    split_t:
        ``(depth,)`` split thresholds; gate ``l`` is ``sigmoid(beta (W_l . x - t_l))``.
    depth:
        Partition depth (``2**depth`` regions).
    beta:
        Gate sharpness. Larger -> sharper interface (the ``beta -> inf`` limit).
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    subfields: tuple[OneLayerVectorField, ...]
    split_W: Array  # (depth, D)
    split_t: Array  # (depth,)
    depth: int
    beta: float = 8.0

    def __post_init__(self) -> None:
        if self.split_W.ndim != 2:
            raise ValueError(f"split_W must be (depth, D), got shape {tuple(self.split_W.shape)}")
        depth, D = self.split_W.shape
        if D != self.coordinate_spec.ndim:
            raise ValueError(f"split_W D={D} != coordinate ndim {self.coordinate_spec.ndim}")
        if self.split_t.shape != (depth,):
            raise ValueError(f"split_t must be (depth,)={depth}, got {tuple(self.split_t.shape)}")
        if self.depth != int(depth):
            raise ValueError(f"depth={self.depth} inconsistent with split_W depth {depth}")
        n_regions = 1 << int(depth)
        if len(self.subfields) != n_regions:
            raise ValueError(
                f"expected {n_regions} subfields (2**depth for depth={depth}), "
                f"got {len(self.subfields)}"
            )

    @property
    def n_regions(self) -> int:
        return 1 << self.depth

    # -- FieldBase plumbing: no single pre-activation tower (autodiff path) --------- #
    def _pre_activations(self, coords: Array) -> Array | None:
        return None

    # -- the blended forward u = sum_l w_l(x) u_l(x) -------------------------------- #
    def partition_weights(self, coords: Array, beta: float | None = None) -> Array:
        r"""Soft partition-of-unity weights ``(B, n_regions)`` over the coordinates."""
        from omnibias.partition.jax.weights import partition_weights_arrays

        b = self.beta if beta is None else float(beta)
        return partition_weights_arrays(self.split_W, self.split_t, coords, b, self.depth)

    def _subfield_values(self, coords: Array) -> Array:
        r"""Each region sub-solution's raw values, stacked ``(B, n_regions, C)``."""
        outs = [sub.value_all(sub._sigma(sub._pre_activations(coords))) for sub in self.subfields]
        return jnp.stack(outs, axis=1)

    def forward_values(self, coords: Array, beta: float | None = None) -> Array:
        r"""Blended field values ``u(x) = sum_l w_l(x) u_l(x)`` of shape ``(B, C)``."""
        w = self.partition_weights(coords, beta)  # (B, L)
        o = self._subfield_values(coords)  # (B, L, C)
        return jnp.einsum("bl,blc->bc", w, o)

    # -- state-method path consumed by the fields ops dispatch ("partitioned") ------ #
    def value_component(self, state: FieldState, name: str) -> Array:
        ci = self.components.index(name)
        return self.forward_values(state.coords)[:, ci]

    def _point_component(self, x_row: Array, ci: int) -> Array:
        r"""Scalar ``u_ci`` at a single point ``x_row`` of shape ``(D,)``."""
        return self.forward_values(x_row[None, :])[0, ci]

    def derivative(self, state: FieldState, name: str, *, axis: int, order: int = 1) -> Array:
        r"""``d^order u_name / dx_axis^order`` via the autodiff product rule."""
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

    def __repr__(self) -> str:
        return (
            f"PartitionedField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, n_regions={self.n_regions}, "
            f"beta={self.beta})"
        )


# Marker read by the omnibias-fields backend ops to select the autodiff state-method
# path (mirrors the torch twin; avoids a fields -> pinn import cycle).
PartitionedField._omnibias_dispatch = "partitioned"


# Pytree registration keeps the field through jit/grad/vmap transformations.
def _pf_flatten(field: PartitionedField) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    children = (field.subfields, field.split_W, field.split_t)
    aux = (field.coordinate_spec, field.components, field.depth, field.beta)
    return children, aux


def _pf_unflatten(aux: tuple[Any, ...], children: tuple[Any, ...]) -> PartitionedField:
    coordinate_spec, components, depth, beta = aux
    subfields, split_W, split_t = children
    obj = PartitionedField.__new__(PartitionedField)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "subfields", subfields)
    object.__setattr__(obj, "split_W", split_W)
    object.__setattr__(obj, "split_t", split_t)
    object.__setattr__(obj, "depth", depth)
    object.__setattr__(obj, "beta", beta)
    return obj


jax.tree_util.register_pytree_node(PartitionedField, _pf_flatten, _pf_unflatten)


def build_partitioned_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    split_dirs: Sequence[Sequence[float]] | Array,
    split_thresh: Sequence[float] | Array,
    hidden: int = 16,
    base: str = "tanh",
    beta: float = 8.0,
    seed: int = 0,
    dtype: Any = jnp.float64,
) -> PartitionedField:
    r"""Convenience builder: one :class:`OneLayerVectorField` per region + the split.

    ``split_dirs`` is ``(depth, D)`` and ``split_thresh`` is ``(depth,)``; the field has
    ``2**depth`` regions. Each region sub-solution is an independent small omnibias field
    (distinct PRNG seed per region).
    """
    W = jnp.asarray(split_dirs, dtype=dtype)
    t = jnp.asarray(split_thresh, dtype=dtype)
    if W.ndim != 2:
        raise ValueError(f"split_dirs must be (depth, D), got shape {tuple(W.shape)}")
    depth = int(W.shape[0])
    n_regions = 1 << depth
    subfields = tuple(
        make_one_layer_vector_field(
            coordinate_spec=coordinate_spec,
            components=components,
            hidden=hidden,
            base=base,
            seed=seed + i,
            dtype=dtype,
        )
        for i in range(n_regions)
    )
    return PartitionedField(
        coordinate_spec=coordinate_spec,
        components=components,
        subfields=subfields,
        split_W=W,
        split_t=t,
        depth=depth,
        beta=float(beta),
    )


__all__ = ["PartitionedField", "build_partitioned_field"]

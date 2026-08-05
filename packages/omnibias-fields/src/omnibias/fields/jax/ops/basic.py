# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX basic ops (mirrors :mod:`omnibias.fields.torch.ops.basic`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _resolve_axis(state: FieldState, axis: int | str) -> int:
    return state.coordinate_spec.axis_index(axis)


def _is_one_layer(state: FieldState) -> bool:
    return getattr(state.field, "_omnibias_dispatch", None) == "one_layer"


def _is_spectral(state: FieldState) -> bool:
    return getattr(state.field, "_omnibias_dispatch", None) == "spectral"


def _is_chebyshev(state: FieldState) -> bool:
    return getattr(state.field, "_omnibias_dispatch", None) == "chebyshev"


def _is_cage(state: FieldState) -> bool:
    return getattr(state.field, "_omnibias_dispatch", None) == "cage"


def _is_partitioned(state: FieldState) -> bool:
    # A partition-of-unity field ``u = sum_l w_l(x) u_l(x)`` (omnibias.pinn.partition);
    # its derivatives are the autodiff product-rule state-methods, so it routes like cage.
    return getattr(state.field, "_omnibias_dispatch", None) == "partitioned"


def _sigma_of_order(state: FieldState, order: int) -> Array:
    if not _is_one_layer(state):
        raise NotImplementedError(
            "sigma cache is only meaningful for OneLayerVectorField"
        )
    return state.sigma_cache.get_or_compute(
        order, lambda n: state.field._sigma_n(state.sigma_cache.z, n),
    )


def value(state: FieldState, name: str) -> Array:
    if _is_one_layer(state):
        sigma_z = _sigma_of_order(state, 0)
        return state.field.value(sigma_z, name)
    if _is_spectral(state):
        return state.field.value_component(state, name)  # type: ignore[attr-defined]
    if _is_chebyshev(state):
        return state.field.value_component(state, name)  # type: ignore[attr-defined]
    if _is_cage(state) or _is_partitioned(state):
        return state.field.value_component(state, name)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"value op not implemented for field type {type(state.field).__name__}"
    )


def stack_components(state: FieldState, names: tuple[str, ...]) -> Array:
    cols = [value(state, n) for n in names]
    return jnp.stack(cols, axis=-1)


def derivative(
    state: FieldState, name: str, *, axis: int | str, order: int = 1,
) -> Array:
    if order < 1:
        if order == 0:
            return value(state, name)
        raise ValueError(f"derivative order must be >= 0, got {order}")
    a = _resolve_axis(state, axis)
    if _is_one_layer(state):
        sigma_n = _sigma_of_order(state, order)
        return state.field.nth_partial(sigma_n, name, a, order)
    if _is_spectral(state):
        return state.field.derivative(state, name, axis=a, order=order)  # type: ignore[attr-defined]
    if _is_chebyshev(state):
        return state.field.derivative(state, name, axis=a, order=order)  # type: ignore[attr-defined]
    if _is_cage(state) or _is_partitioned(state):
        return state.field.derivative(state, name, axis=a, order=order)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"derivative op not implemented for field type {type(state.field).__name__}"
    )


def vector_derivative(
    state: FieldState,
    names: tuple[str, ...],
    *,
    axis: int | str,
    order: int = 1,
) -> Array:
    cols = [derivative(state, n, axis=axis, order=order) for n in names]
    return jnp.stack(cols, axis=-1)


def mixed_partial(
    state: FieldState,
    name: str,
    axes: tuple[int | str, ...],
    orders: tuple[int, ...],
) -> Array:
    if len(axes) != len(orders):
        raise ValueError(
            f"axes and orders must have the same length; got {len(axes)} and {len(orders)}"
        )
    if not axes:
        return value(state, name)
    folded: dict[int, int] = {}
    for a, o in zip(axes, orders, strict=False):
        if o < 1:
            continue
        ai = _resolve_axis(state, a)
        folded[ai] = folded.get(ai, 0) + int(o)
    if not folded:
        return value(state, name)
    int_axes = tuple(folded)
    int_orders = tuple(folded[a] for a in int_axes)
    total_order = sum(int_orders)
    if _is_one_layer(state):
        sigma_n = _sigma_of_order(state, total_order)
        return state.field.mixed_partial(sigma_n, name, int_axes, int_orders)
    if _is_spectral(state):
        return state.field.mixed_partial(state, name, int_axes, int_orders)  # type: ignore[attr-defined]
    if _is_chebyshev(state):
        return state.field.mixed_partial(state, name, int_axes, int_orders)  # type: ignore[attr-defined]
    if _is_cage(state):
        return state.field.mixed_partial(state, name, int_axes, int_orders)  # type: ignore[attr-defined]
    if _is_partitioned(state):
        return state.field.mixed_partial(state, name, int_axes, int_orders)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"mixed_partial not implemented for field type {type(state.field).__name__}"
    )


def gradient(
    state: FieldState,
    name: str,
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Array:
    if axes is None:
        axis_idx = tuple(
            state.coordinate_spec.axis_index(a)
            for a in state.coordinate_spec.spatial_axes
        )
    else:
        axis_idx = tuple(_resolve_axis(state, a) for a in axes)
    if _is_one_layer(state):
        sigma_p = _sigma_of_order(state, 1)
        full = state.field.gradient_full(sigma_p, name)
        if axis_idx == tuple(range(full.shape[-1])):
            return full
        return full[..., list(axis_idx)]
    if _is_spectral(state) or _is_chebyshev(state) or _is_cage(state) or _is_partitioned(state):
        cols = [
            derivative(state, name, axis=a, order=1) for a in axis_idx
        ]
        return jnp.stack(cols, axis=-1)
    raise NotImplementedError(
        f"gradient op not implemented for field type {type(state.field).__name__}"
    )


def divergence(state: FieldState, names: tuple[str, ...]) -> Array:
    sa = state.coordinate_spec.spatial_axes
    if len(names) != len(sa):
        raise ValueError(
            f"divergence: number of components {len(names)} must equal "
            f"number of spatial axes {len(sa)} ({sa!r})"
        )
    out = None
    for n, a in zip(names, sa, strict=False):
        d = derivative(state, n, axis=a, order=1)
        out = d if out is None else out + d
    assert out is not None
    return out


def laplacian(
    state: FieldState,
    name: str,
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Array:
    if axes is None and _is_one_layer(state):
        sigma_pp = _sigma_of_order(state, 2)
        return state.field.laplacian(sigma_pp, name)
    if axes is None:
        axes = tuple(state.coordinate_spec.spatial_axes)
    out = None
    for a in axes:
        d2 = derivative(state, name, axis=a, order=2)
        out = d2 if out is None else out + d2
    assert out is not None
    return out


__all__ = [
    "derivative",
    "divergence",
    "gradient",
    "laplacian",
    "mixed_partial",
    "stack_components",
    "value",
    "vector_derivative",
]

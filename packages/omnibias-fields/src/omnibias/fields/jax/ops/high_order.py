# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX high-order ops (mirrors :mod:`omnibias.fields.torch.ops.high_order`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import (
    _is_chebyshev,
    _is_one_layer,
    _is_spectral,
    _resolve_axis,
    _sigma_of_order,
    derivative,
    gradient,
)

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def biharmonic(state: FieldState, name: str) -> Array:
    if _is_one_layer(state):
        sigma_4 = _sigma_of_order(state, 4)
        return state.field.polylaplacian(sigma_4, name, k=2)
    if _is_spectral(state):
        return state.field.biharmonic(state, name)  # type: ignore[attr-defined]
    if _is_chebyshev(state):
        return state.field.biharmonic(state, name)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"biharmonic not implemented for field type {type(state.field).__name__}"
    )


def polylaplacian(state: FieldState, name: str, *, k: int) -> Array:
    if k < 1:
        raise ValueError(f"polylaplacian k must be >= 1, got {k}")
    if k == 1:
        from omnibias.fields.jax.ops.basic import laplacian
        return laplacian(state, name)
    if _is_one_layer(state):
        sigma_2k = _sigma_of_order(state, 2 * k)
        return state.field.polylaplacian(sigma_2k, name, k=k)
    if _is_spectral(state):
        return state.field.polylaplacian(state, name, k=k)  # type: ignore[attr-defined]
    if _is_chebyshev(state):
        return state.field.polylaplacian(state, name, k=k)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"polylaplacian not implemented for field type {type(state.field).__name__}"
    )


def hessian(
    state: FieldState,
    name: str,
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Array:
    axis_idx = (
        tuple(range(state.coordinate_spec.ndim))
        if axes is None
        else tuple(_resolve_axis(state, a) for a in axes)
    )
    if _is_one_layer(state):
        sigma_pp = _sigma_of_order(state, 2)
        full = state.field.hessian_full(sigma_pp, name)
        if axis_idx == tuple(range(full.shape[-1])):
            return full
        idx = jnp.array(list(axis_idx))
        return full[..., idx, :][..., :, idx]
    rows = []
    idx = jnp.array(list(axis_idx))
    for i in axis_idx:
        gi = gradient_of_derivative(state, name, axis=i)
        rows.append(gi[..., idx])
    return jnp.stack(rows, axis=-2)


def spatial_hessian(state: FieldState, name: str) -> Array:
    return hessian(state, name, axes=tuple(state.coordinate_spec.spatial_axes))


def gradient_of_derivative(
    state: FieldState, name: str, *, axis: int | str,
) -> Array:
    a = _resolve_axis(state, axis)
    D = state.coordinate_spec.ndim
    cols = []
    for j in range(D):
        if a == j:
            cols.append(derivative(state, name, axis=a, order=2))
        else:
            from omnibias.fields.jax.ops.basic import mixed_partial
            cols.append(mixed_partial(state, name, (a, j), (1, 1)))
    return jnp.stack(cols, axis=-1)


def jacobian(state: FieldState, names: tuple[str, ...]) -> Array:
    if _is_one_layer(state):
        sigma_p = _sigma_of_order(state, 1)
        rows = [
            state.field.gradient_full(sigma_p, n) for n in names
        ]
        return jnp.stack(rows, axis=-2)
    rows = []
    for n in names:
        rows.append(gradient(state, n, axes=tuple(range(state.coordinate_spec.ndim))))
    return jnp.stack(rows, axis=-2)


def vector_hessian(
    state: FieldState,
    names: tuple[str, ...],
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Array:
    rows = [hessian(state, n, axes=axes) for n in names]
    return jnp.stack(rows, axis=-3)


def vector_laplacian(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.basic import laplacian
    cols = [laplacian(state, n) for n in names]
    return jnp.stack(cols, axis=-1)


def vector_biharmonic(state: FieldState, names: tuple[str, ...]) -> Array:
    cols = [biharmonic(state, n) for n in names]
    return jnp.stack(cols, axis=-1)


def vector_polylaplacian(state: FieldState, names: tuple[str, ...], *, k: int) -> Array:
    cols = [polylaplacian(state, n, k=k) for n in names]
    return jnp.stack(cols, axis=-1)


__all__ = [
    "biharmonic",
    "gradient_of_derivative",
    "hessian",
    "jacobian",
    "polylaplacian",
    "spatial_hessian",
    "vector_biharmonic",
    "vector_hessian",
    "vector_laplacian",
    "vector_polylaplacian",
]

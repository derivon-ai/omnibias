# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""High-order ops: ``biharmonic``, ``polylaplacian``, ``hessian``, ``jacobian``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import (
    _is_chebyshev,
    _is_jet_mlp,
    _is_one_layer,
    _is_spectral,
    _resolve_axis,
    _sigma_of_order,
    derivative,
    gradient,
)
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def biharmonic(state: FieldState, name: str) -> Tensor:
    """``Delta^2 f_name`` of shape ``(B,)``.

    Closed-form on one-layer omnibias fields (one extra fast-path call
    at order 4); spectral fields get a diagonal-multiplier shortcut.
    """
    if _is_one_layer(state):
        sigma_4 = _sigma_of_order(state, 4)
        return state.field.polylaplacian(sigma_4, name, k=2)
    if _is_spectral(state):
        return state.field.biharmonic(state, name)  # type: ignore[attr-defined]
    if _is_chebyshev(state):
        return state.field.biharmonic(state, name)  # type: ignore[attr-defined]
    if _is_jet_mlp(state):
        return state.field.biharmonic(state, name)  # type: ignore[attr-defined]
    # Generic fallback: chain through laplacian-of-laplacian.
    raise NotImplementedError(
        f"biharmonic not implemented for field type {type(state.field).__name__}"
    )


def polylaplacian(state: FieldState, name: str, *, k: int) -> Tensor:
    """``Delta^k f_name`` of shape ``(B,)``.

    Memory and time both ``O(B * H)``, *independent of* ``k``, because
    the omnibias closed-form expresses ``Delta^k`` as one ``sigma^{(2k)}``
    fast-path call plus a per-mode reduction.
    """
    if k < 1:
        raise ValueError(f"polylaplacian k must be >= 1, got {k}")
    if k == 1:
        from omnibias.fields.torch.ops.basic import laplacian
        return laplacian(state, name)
    if _is_one_layer(state):
        sigma_2k = _sigma_of_order(state, 2 * k)
        return state.field.polylaplacian(sigma_2k, name, k=k)
    if _is_spectral(state):
        return state.field.polylaplacian(state, name, k=k)  # type: ignore[attr-defined]
    if _is_chebyshev(state):
        # Composition fallback: Delta^k = Delta(Delta^{k-1}).
        return state.field.polylaplacian(state, name, k=k)  # type: ignore[attr-defined]
    if _is_jet_mlp(state):
        # Multinomial expansion of (sum_i d_i^2)^k over one order-2k jet.
        return state.field.polylaplacian(state, name, k=k)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"polylaplacian not implemented for field type {type(state.field).__name__}"
    )


def hessian(
    state: FieldState,
    name: str,
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Tensor:
    """Hessian over selected axes, shape ``(B, len(axes), len(axes))``.

    ``axes=None`` keeps the historical behaviour: all coordinate axes,
    including time. Use :func:`spatial_hessian` for the common spatial-only
    Hessian.
    """
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
        idx = list(axis_idx)
        return full[..., idx, :][..., :, idx]
    if _is_jet_mlp(state):
        # The order-2 block of a single jet is the whole Hessian.
        full = state.field.hessian_full(state, name)  # type: ignore[attr-defined]
        if axis_idx == tuple(range(full.shape[-1])):
            return full
        idx = list(axis_idx)
        return full[..., idx, :][..., :, idx]
    # Generic fallback: build by taking first-derivative-of-gradient.
    rows = []
    for i in axis_idx:
        # d/dx_i grad f -> the i-th column is grad of (d f / dx_i).
        gi = gradient_of_derivative(state, name, axis=i)
        rows.append(gi[..., list(axis_idx)])
    return torch.stack(rows, dim=-2)  # (B, D, D); rows index outer axis


def spatial_hessian(state: FieldState, name: str) -> Tensor:
    """Spatial-only Hessian, excluding the time axis."""
    return hessian(state, name, axes=tuple(state.coordinate_spec.spatial_axes))


def gradient_of_derivative(
    state: FieldState, name: str, *, axis: int | str,
) -> Tensor:
    """Helper: ``grad (d f / dx_axis)``. Used to assemble the Hessian for
    field types without a closed-form full Hessian."""
    a = _resolve_axis(state, axis)
    D = state.coordinate_spec.ndim
    cols = []
    for j in range(D):
        if a == j:
            cols.append(derivative(state, name, axis=a, order=2))
        else:
            from omnibias.fields.torch.ops.basic import mixed_partial
            cols.append(mixed_partial(state, name, (a, j), (1, 1)))
    return torch.stack(cols, dim=-1)


def jacobian(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """Jacobian of the vector ``(f_{names[0]}, ...)``, shape ``(B, C, D)``.

    ``J[b, c, d] = d f_{names[c]} / dx_d`` over *all* axes (including
    the time axis), since the user can subset afterwards.
    """
    if _is_one_layer(state):
        sigma_p = _sigma_of_order(state, 1)
        rows = [
            state.field.gradient_full(sigma_p, n) for n in names
        ]
        return torch.stack(rows, dim=-2)  # (B, C, D)
    if _is_jet_mlp(state):
        rows = [state.field.gradient_full(state, n) for n in names]  # type: ignore[attr-defined]
        return torch.stack(rows, dim=-2)  # (B, C, D)
    rows = []
    for n in names:
        rows.append(gradient(state, n, axes=tuple(range(state.coordinate_spec.ndim))))
    return torch.stack(rows, dim=-2)


def vector_hessian(
    state: FieldState,
    names: tuple[str, ...],
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Tensor:
    """Stack component Hessians, shape ``(B, C, D, D)`` or ``(B, C, A, A)``."""
    rows = [hessian(state, n, axes=axes) for n in names]
    return torch.stack(rows, dim=-3)


def vector_laplacian(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """Laplacian applied per component, shape ``(B, len(names))``."""
    from omnibias.fields.torch.ops.basic import laplacian
    cols = [laplacian(state, n) for n in names]
    return torch.stack(cols, dim=-1)


def vector_biharmonic(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """Biharmonic per component, shape ``(B, len(names))``."""
    cols = [biharmonic(state, n) for n in names]
    return torch.stack(cols, dim=-1)


def vector_polylaplacian(state: FieldState, names: tuple[str, ...], *, k: int) -> Tensor:
    """Polylaplacian applied per component, shape ``(B, len(names))``."""
    cols = [polylaplacian(state, n, k=k) for n in names]
    return torch.stack(cols, dim=-1)


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

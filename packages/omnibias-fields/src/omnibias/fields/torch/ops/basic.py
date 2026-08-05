# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Basic ops: ``value``, ``derivative``, ``gradient``, ``divergence``, ``laplacian``.

Each op consumes a :class:`FieldState`, lazily fills the state's
:class:`SigmaCache` with the required ``sigma^(n)(z)`` order, and asks
the field for the corresponding closed-form chain-rule reduction.

Field-type dispatch lives here: each op picks the right kernel based on
``type(state.field)``. The dispatch is intentionally explicit (no
multi-dispatch decorator) -- the field types in v0.1 are small and the
explicit branches are easy to read in the parity tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


# ---------------- helpers --------------------------------------------


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
    # A partition-of-unity field ``u = sum_l w_l(x) u_l(x)`` (omnibias.pinn.partition).
    # Its derivatives are the autodiff product-rule state-methods (products of sigmoids are
    # NOT the closed-form sigma-tower path), so it routes exactly like spectral / cage.
    return getattr(state.field, "_omnibias_dispatch", None) == "partitioned"


def _is_jet_mlp(state: FieldState) -> bool:
    # A deep / Fourier-feature field (omnibias.pinn.<backend>.fields.jet_mlp) whose
    # derivatives come from the exact multivariate jet ``mlp_jet_mv``. Still closed
    # form -- just multi-layer Faa di Bruno rather than the single-layer sigma-tower
    # reduction -- so it uses the state-method path with its own jet cache.
    return getattr(state.field, "_omnibias_dispatch", None) == "jet_mlp"


def _sigma_of_order(state: FieldState, order: int) -> Tensor:
    """Return ``sigma^(order)(z)`` from the cache, computing it once."""
    if not _is_one_layer(state):
        raise NotImplementedError(
            "sigma cache is only meaningful for OneLayerVectorField; "
            "spectral / chebyshev fields use a different code path"
        )
    return state.sigma_cache.get_or_compute(
        order, lambda n: state.field._sigma_n(state.sigma_cache.z, n),
    )


# ---------------- value ----------------------------------------------


def value(state: FieldState, name: str) -> Tensor:
    """Return ``f_name(coords)`` of shape ``(B,)``."""
    if _is_one_layer(state):
        sigma_z = _sigma_of_order(state, 0)
        return state.field.value(sigma_z, name)
    if _is_spectral(state):
        return state.field.value_component(state, name)  # type: ignore[attr-defined]
    if _is_chebyshev(state):
        return state.field.value_component(state, name)  # type: ignore[attr-defined]
    if _is_cage(state) or _is_partitioned(state) or _is_jet_mlp(state):
        return state.field.value_component(state, name)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"value op not implemented for field type {type(state.field).__name__}"
    )


def stack_components(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """Stack component values into a ``(B, len(names))`` tensor."""
    cols = [value(state, n) for n in names]
    return torch.stack(cols, dim=-1)


# ---------------- derivative ------------------------------------------


def derivative(
    state: FieldState, name: str, *, axis: int | str, order: int = 1,
) -> Tensor:
    """``d^order f_name / dx_axis^order`` of shape ``(B,)``."""
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
    if _is_cage(state) or _is_partitioned(state) or _is_jet_mlp(state):
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
) -> Tensor:
    """Stack of per-component derivatives, shape ``(B, len(names))``."""
    cols = [derivative(state, n, axis=axis, order=order) for n in names]
    return torch.stack(cols, dim=-1)


def mixed_partial(
    state: FieldState,
    name: str,
    axes: tuple[int | str, ...],
    orders: tuple[int, ...],
) -> Tensor:
    """``d^|orders| f_name / prod_a dx_a^{orders_a}`` of shape ``(B,)``.

    Equivalent to repeated single-axis differentiation. Distinct axes
    produce a true mixed partial; repeated axes are folded by summing
    orders.
    """
    if len(axes) != len(orders):
        raise ValueError(
            f"axes and orders must have the same length; got {len(axes)} and {len(orders)}"
        )
    if not axes:
        return value(state, name)
    # Fold repeated axes into a single (axis, total_order) pair.
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
    if _is_cage(state) or _is_partitioned(state) or _is_jet_mlp(state):
        return state.field.mixed_partial(state, name, int_axes, int_orders)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"mixed_partial op not implemented for field type {type(state.field).__name__}"
    )


# ---------------- gradient -------------------------------------------


def gradient(
    state: FieldState,
    name: str,
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Tensor:
    """``nabla f_name`` of shape ``(B, len(axes))``.

    Default ``axes=None`` means *all spatial* axes (the time axis is
    excluded). Pass ``axes=tuple(range(state.coordinate_spec.ndim))``
    to include time.
    """
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
    if _is_jet_mlp(state):
        # One jet already holds every first partial; read them all in one go.
        full = state.field.gradient_full(state, name)  # type: ignore[attr-defined]
        if axis_idx == tuple(range(full.shape[-1])):
            return full
        return full[..., list(axis_idx)]
    if _is_spectral(state) or _is_chebyshev(state) or _is_cage(state) or _is_partitioned(state):
        cols = [
            derivative(state, name, axis=a, order=1) for a in axis_idx
        ]
        return torch.stack(cols, dim=-1)
    raise NotImplementedError(
        f"gradient op not implemented for field type {type(state.field).__name__}"
    )


# ---------------- divergence -----------------------------------------


def divergence(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """``nabla . u`` for a vector field of components ``names``.

    Shape ``(B,)``. The ``i``-th component is differentiated along the
    ``i``-th *spatial* axis, in the order given by
    :attr:`CoordinateSpec.spatial_axes`. ``len(names)`` must equal the
    number of spatial axes.
    """
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


# ---------------- Laplacian -------------------------------------------


def laplacian(
    state: FieldState,
    name: str,
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Tensor:
    """``Delta f_name = sum_a d^2 f / dx_a^2`` over the spatial axes.

    Default ``axes=None`` reduces over the spatial axes. Pass an
    explicit axes tuple to over-ride (e.g. include the time axis to get
    the Wave-equation operator).
    """
    if axes is None and _is_one_layer(state):
        # Fast-path: closed-form via row-norm-sq.
        sigma_pp = _sigma_of_order(state, 2)
        return state.field.laplacian(sigma_pp, name)
    if axes is None and _is_jet_mlp(state):
        # Fast-path: the spatial trace of the Hessian block of a single jet.
        return state.field.laplacian(state, name)  # type: ignore[attr-defined]
    # Fallback: sum of pure 2nd partials along the requested axes.
    if axes is None:
        axes = tuple(state.coordinate_spec.spatial_axes)
    out = None
    for a in axes:
        d2 = derivative(state, name, axis=a, order=2)
        out = d2 if out is None else out + d2
    assert out is not None, "axes must be non-empty"
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

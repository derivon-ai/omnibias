# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX vector ops (mirrors :mod:`omnibias.fields.torch.ops.vector`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import derivative, gradient, mixed_partial

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def curl(state: FieldState, names: tuple[str, ...]) -> Array:
    sa = state.coordinate_spec.spatial_axes
    if len(names) != len(sa):
        raise ValueError(
            f"curl: vector length {len(names)} must equal n_spatial {len(sa)} "
            f"({sa!r})"
        )
    if len(sa) == 1:
        raise ValueError(
            "curl is not defined for 1D vector fields; use derivative instead"
        )
    if len(sa) == 2:
        u, v = names
        x_axis, y_axis = sa[0], sa[1]
        cz = derivative(state, v, axis=x_axis) - derivative(state, u, axis=y_axis)
        return cz[..., None]
    if len(sa) == 3:
        u, v, w = names
        x_axis, y_axis, z_axis = sa[0], sa[1], sa[2]
        cx = derivative(state, w, axis=y_axis) - derivative(state, v, axis=z_axis)
        cy = derivative(state, u, axis=z_axis) - derivative(state, w, axis=x_axis)
        cz = derivative(state, v, axis=x_axis) - derivative(state, u, axis=y_axis)
        return jnp.stack([cx, cy, cz], axis=-1)
    raise ValueError(
        f"curl supported for 2D / 3D vector fields, got {len(sa)} spatial axes"
    )


def vorticity(state: FieldState, names: tuple[str, ...]) -> Array:
    return curl(state, names)


def spatial_jacobian(state: FieldState, names: tuple[str, ...]) -> Array:
    rows = [gradient(state, n) for n in names]
    return jnp.stack(rows, axis=-2)


def deformation_gradient(state: FieldState, names: tuple[str, ...]) -> Array:
    sa = state.coordinate_spec.spatial_axes
    if len(names) != len(sa):
        raise ValueError(
            f"deformation_gradient: vector length {len(names)} must equal "
            f"n_spatial {len(sa)} ({sa!r})"
        )
    return spatial_jacobian(state, names)


def strain_rate(state: FieldState, names: tuple[str, ...]) -> Array:
    J = deformation_gradient(state, names)
    if J.shape[-2] != J.shape[-1]:
        raise ValueError(
            f"strain_rate requires square Jacobian; got shape {tuple(J.shape)}"
        )
    return 0.5 * (J + jnp.swapaxes(J, -1, -2))


def rate_of_rotation_tensor(state: FieldState, names: tuple[str, ...]) -> Array:
    r"""Antisymmetric velocity-gradient (spin) tensor :math:`W = \tfrac12(J - J^T)`.

    Complement of :func:`strain_rate`; together ``J = strain_rate + rate_of_rotation_tensor``.
    """
    J = deformation_gradient(state, names)
    if J.shape[-2] != J.shape[-1]:
        raise ValueError(
            f"rate_of_rotation_tensor requires square Jacobian; got shape {tuple(J.shape)}"
        )
    return 0.5 * (J - jnp.swapaxes(J, -1, -2))


def gradient_of_divergence(state: FieldState, names: tuple[str, ...]) -> Array:
    r""":math:`\nabla(\nabla\cdot u)`, shape ``(B, d)``.

    Component ``i`` is :math:`\sum_j \partial^2_{ij} u_j`.
    """
    sa = state.coordinate_spec.spatial_axes
    if len(names) != len(sa):
        raise ValueError(
            f"gradient_of_divergence: vector length {len(names)} must equal "
            f"n_spatial {len(sa)} ({sa!r})"
        )
    cols = []
    for i in sa:
        acc = None
        for n, j in zip(names, sa, strict=False):
            term = mixed_partial(state, n, (i, j), (1, 1))
            acc = term if acc is None else acc + term
        cols.append(acc)
    return jnp.stack(cols, axis=-1)


def curl_of_curl(state: FieldState, names: tuple[str, ...]) -> Array:
    r""":math:`\nabla\times(\nabla\times u)` for 2D / 3D vector fields.

    Expanded directly from second partials (not via the identity), so a test can
    independently assert ``curl_of_curl == gradient_of_divergence - vector_laplacian``.
    """
    sa = state.coordinate_spec.spatial_axes
    if len(names) != len(sa):
        raise ValueError(
            f"curl_of_curl: vector length {len(names)} must equal n_spatial "
            f"{len(sa)} ({sa!r})"
        )
    if len(sa) == 2:
        u, v = names
        x, y = sa
        s_y = mixed_partial(state, v, (x, y), (1, 1)) - derivative(state, u, axis=y, order=2)
        s_x = derivative(state, v, axis=x, order=2) - mixed_partial(state, u, (x, y), (1, 1))
        return jnp.stack([s_y, -s_x], axis=-1)
    if len(sa) == 3:
        u, v, w = names
        x, y, z = sa
        ccx = (
            mixed_partial(state, v, (x, y), (1, 1)) + mixed_partial(state, w, (x, z), (1, 1))
            - derivative(state, u, axis=y, order=2) - derivative(state, u, axis=z, order=2)
        )
        ccy = (
            mixed_partial(state, w, (y, z), (1, 1)) + mixed_partial(state, u, (x, y), (1, 1))
            - derivative(state, v, axis=z, order=2) - derivative(state, v, axis=x, order=2)
        )
        ccz = (
            mixed_partial(state, u, (x, z), (1, 1)) + mixed_partial(state, v, (y, z), (1, 1))
            - derivative(state, w, axis=x, order=2) - derivative(state, w, axis=y, order=2)
        )
        return jnp.stack([ccx, ccy, ccz], axis=-1)
    raise ValueError(
        f"curl_of_curl supported for 2D / 3D vector fields, got {len(sa)} spatial axes"
    )


__all__ = [
    "curl",
    "curl_of_curl",
    "deformation_gradient",
    "gradient_of_divergence",
    "rate_of_rotation_tensor",
    "spatial_jacobian",
    "strain_rate",
    "vorticity",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Vector ops: ``curl``, ``vorticity``, ``strain_rate``, ``deformation_gradient``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import derivative, gradient, mixed_partial
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def curl(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """``nabla x u`` of a 3D vector field, shape ``(B, 3)``.

    For 2D this returns the *scalar* curl ``omega = du2/dx1 - du1/dx2``
    of shape ``(B, 1)``. For 1D it raises (curl is not defined).

    The vector ``names`` length must equal the number of spatial axes.
    """
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
        return cz.unsqueeze(-1)                            # (B, 1)
    if len(sa) == 3:
        u, v, w = names
        x_axis, y_axis, z_axis = sa[0], sa[1], sa[2]
        cx = derivative(state, w, axis=y_axis) - derivative(state, v, axis=z_axis)
        cy = derivative(state, u, axis=z_axis) - derivative(state, w, axis=x_axis)
        cz = derivative(state, v, axis=x_axis) - derivative(state, u, axis=y_axis)
        return torch.stack([cx, cy, cz], dim=-1)            # (B, 3)
    raise ValueError(
        f"curl supported for 2D / 3D vector fields, got {len(sa)} spatial axes"
    )


def vorticity(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """Alias for :func:`curl` -- vorticity ``omega = nabla x velocity``."""
    return curl(state, names)


def spatial_jacobian(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """Spatial Jacobian ``J[b,i,j] = d f_i / dx_j`` of shape ``(B, C, n_spatial)``.

    Unlike :func:`deformation_gradient`, this accepts any number of
    components; it is useful for multi-output scalar stacks as well as
    physical vector fields.
    """
    rows = [gradient(state, n) for n in names]
    return torch.stack(rows, dim=-2)


def deformation_gradient(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """Spatial gradient tensor ``J[b,i,j] = du_i / dx_j`` of shape ``(B, C, n_spatial)``.

    This is the standard "deformation gradient" used in continuum
    mechanics. For an incompressible flow the trace equals
    :math:`\\nabla \\cdot u = 0`.
    """
    sa = state.coordinate_spec.spatial_axes
    if len(names) != len(sa):
        raise ValueError(
            f"deformation_gradient: vector length {len(names)} must equal "
            f"n_spatial {len(sa)} ({sa!r})"
        )
    return spatial_jacobian(state, names)


def strain_rate(state: FieldState, names: tuple[str, ...]) -> Tensor:
    """Symmetric strain-rate ``S = 0.5 (J + J^T)`` of shape ``(B, C, C)``.

    Requires ``len(names) == n_spatial``. This is the symmetric part of
    the deformation gradient.
    """
    J = deformation_gradient(state, names)                 # (B, C, n_spatial)
    if J.shape[-2] != J.shape[-1]:
        raise ValueError(
            f"strain_rate requires square Jacobian; got shape {tuple(J.shape)}"
        )
    return 0.5 * (J + J.transpose(-1, -2))


def rate_of_rotation_tensor(state: FieldState, names: tuple[str, ...]) -> Tensor:
    r"""Antisymmetric velocity-gradient (spin) tensor :math:`W = \tfrac12(J - J^T)`.

    Complement of :func:`strain_rate`; together ``J = strain_rate + rate_of_rotation_tensor``.
    """
    J = deformation_gradient(state, names)
    if J.shape[-2] != J.shape[-1]:
        raise ValueError(
            f"rate_of_rotation_tensor requires square Jacobian; got shape {tuple(J.shape)}"
        )
    return 0.5 * (J - J.transpose(-1, -2))


def gradient_of_divergence(state: FieldState, names: tuple[str, ...]) -> Tensor:
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
    return torch.stack(cols, dim=-1)


def curl_of_curl(state: FieldState, names: tuple[str, ...]) -> Tensor:
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
        return torch.stack([s_y, -s_x], dim=-1)
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
        return torch.stack([ccx, ccy, ccz], dim=-1)
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

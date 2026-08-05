# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Field-coupled geometric operators (jax): Laplace-Beltrami, covariant deriv.

Bit-identical twin of :mod:`omnibias.geometry.torch.ops.field_ops`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import gradient, value
from omnibias.fields.jax.ops.high_order import hessian
from omnibias.geometry.jax.ops.connection import christoffel, inverse_metric

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.geometry._core.manifold import ManifoldSpec


def _axes(manifold: ManifoldSpec, state: FieldState) -> tuple[str, ...]:
    axes = tuple(state.coordinate_spec.axes)
    if len(axes) != manifold.dim:
        raise ValueError(
            f"manifold dim {manifold.dim} != field coordinate axes {axes!r}"
        )
    return axes


def laplace_beltrami(
    state: FieldState, name: str, manifold: ManifoldSpec,
) -> Array:
    r"""Laplace-Beltrami operator :math:`\Delta_g f` of shape ``(B,)``.

    :math:`\Delta_g f = g^{ij}(\partial_i\partial_j f - \Gamma^k_{ij}\partial_k f)`.
    """
    axes = _axes(manifold, state)
    coords = state.coords
    grad = gradient(state, name, axes=axes)
    hess = hessian(state, name, axes=axes)
    ginv = inverse_metric(coords, manifold)
    gamma = christoffel(coords, manifold)
    corrected = hess - jnp.einsum("bkij,bk->bij", gamma, grad)
    return jnp.einsum("bij,bij->b", ginv, corrected)


def covariant_derivative(
    state: FieldState,
    name: str | Sequence[str],
    manifold: ManifoldSpec,
    *,
    kind: Literal["scalar", "vector", "one_form", "tensor"] = "scalar",
) -> Array:
    r"""Covariant derivative :math:`\nabla` (see the torch twin for the full doc)."""
    axes = _axes(manifold, state)
    coords = state.coords
    if kind == "scalar":
        if not isinstance(name, str):
            raise TypeError("scalar covariant derivative needs a single component name")
        return gradient(state, name, axes=axes)

    names = (name,) if isinstance(name, str) else tuple(name)
    d = manifold.dim
    if len(names) != d:
        raise ValueError(f"{kind} field needs {d} component names, got {len(names)}")
    partial = jnp.stack([gradient(state, n, axes=axes) for n in names], axis=-1)
    comp = jnp.stack([value(state, n) for n in names], axis=-1)
    gamma = christoffel(coords, manifold)

    if kind == "vector":
        corr = jnp.einsum("bkil,bl->bik", gamma, comp)
        return partial + corr
    if kind == "one_form":
        corr = jnp.einsum("blik,bl->bik", gamma, comp)
        return partial - corr
    raise NotImplementedError(
        f"covariant_derivative kind={kind!r} is not implemented yet; "
        "supported kinds are 'scalar', 'vector', 'one_form'."
    )


__all__ = ["covariant_derivative", "laplace_beltrami"]

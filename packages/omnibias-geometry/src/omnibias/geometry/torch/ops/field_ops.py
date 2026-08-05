# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Field-coupled geometric operators (torch): Laplace-Beltrami, covariant deriv.

These consume a :class:`omnibias.fields._core.state.FieldState` (for the
*closed-form* field derivatives via the sigma tower) together with a
:class:`ManifoldSpec` (for the metric / connection). The field-function
derivatives are exact closed form; the metric derivatives inside the Christoffel
symbols come from autodiff of the analytic metric (see ``connection``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
from omnibias.fields.torch.ops.basic import gradient, value
from omnibias.fields.torch.ops.high_order import hessian
from omnibias.geometry.torch.ops.connection import christoffel, inverse_metric
from torch import Tensor

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
) -> Tensor:
    r"""Laplace-Beltrami operator :math:`\Delta_g f` of shape ``(B,)``.

    Uses the identity
    :math:`\Delta_g f = g^{ij}\,(\partial_i\partial_j f - \Gamma^k_{ij}\,\partial_k f)`,
    with the field Hessian and gradient supplied in closed form and the
    Christoffel symbols from the metric. For the flat Euclidean metric this
    reduces to the ordinary Laplacian.
    """
    axes = _axes(manifold, state)
    coords = state.coords
    grad = gradient(state, name, axes=axes)        # (B, d)
    hess = hessian(state, name, axes=axes)         # (B, d, d)
    ginv = inverse_metric(coords, manifold)        # (B, d, d)
    gamma = christoffel(coords, manifold)          # (B, k, i, j)
    corrected = hess - torch.einsum("bkij,bk->bij", gamma, grad)
    return torch.einsum("bij,bij->b", ginv, corrected)


def covariant_derivative(
    state: FieldState,
    name: str | Sequence[str],
    manifold: ManifoldSpec,
    *,
    kind: Literal["scalar", "vector", "one_form", "tensor"] = "scalar",
) -> Tensor:
    r"""Covariant derivative :math:`\nabla`.

    - ``kind="scalar"``: ``name`` is one component; returns
      :math:`\nabla_i f = \partial_i f` of shape ``(B, d)``.
    - ``kind="vector"``: ``name`` is ``d`` component names ``V^k``; returns
      :math:`\nabla_i V^k = \partial_i V^k + \Gamma^k_{il} V^l`, shape
      ``(B, d, d)`` indexed ``[i, k]``.
    - ``kind="one_form"``: ``name`` is ``d`` component names ``\omega_k``;
      returns :math:`\nabla_i \omega_k = \partial_i \omega_k - \Gamma^l_{ik}
      \omega_l`, shape ``(B, d, d)`` indexed ``[i, k]``.
    """
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
    # partial[b, i, k] = d_i (field_k)
    partial = torch.stack([gradient(state, n, axes=axes) for n in names], dim=-1)
    comp = torch.stack([value(state, n) for n in names], dim=-1)  # (B, d)
    gamma = christoffel(coords, manifold)  # (B, k, i, j) = Gamma^k_{ij}

    if kind == "vector":
        corr = torch.einsum("bkil,bl->bik", gamma, comp)
        return partial + corr
    if kind == "one_form":
        corr = torch.einsum("blik,bl->bik", gamma, comp)
        return partial - corr
    raise NotImplementedError(
        f"covariant_derivative kind={kind!r} is not implemented yet; "
        "supported kinds are 'scalar', 'vector', 'one_form'."
    )


__all__ = ["covariant_derivative", "laplace_beltrami"]

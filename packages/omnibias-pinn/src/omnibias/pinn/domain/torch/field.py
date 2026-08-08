# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Distance-constrained PINN field for curved boundaries (torch).

Implements the multiplicative hard-BC ansatz

.. math::

    u = g + \phi \cdot \mathrm{NN}

where ``phi`` vanishes on the boundary. This generalises the box-only TFC
cage to curved domains described by an SDF / ADF.

The field is a thin wrapper around
:class:`~omnibias.pinn.torch.cage.HardBoundaryField` (dispatch tag ``cage``),
so the whole existing op surface works unchanged. Derivatives of ``phi`` and
``g`` use autodiff (they are simple algebraic expressions); the base network
stays on its closed-form path. When the base exposes a multivariate jet, the
product ``phi * NN`` can additionally be formed with
:func:`~omnibias.torch.jet_mv.jet_multiply` via :meth:`product_jet_at` -- that
is the exact-tower path for callers who need it.

Honesty labels: BC satisfaction on ``phi = 0`` is exact by construction;
``phi`` derivatives are autodiff-exact; residual accuracy is optimised.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from omnibias.core.multi_index import num_multi_indices
from omnibias.pinn.domain._core.sdf import Box, Halfspace, Sphere
from omnibias.pinn.domain.torch.sdf_torch import (
    DistanceFn,
    from_primitive,
    normalize_distance,
)
from omnibias.pinn.torch.cage.conservation import HardBoundaryField
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.torch.jet_mv import jet_multiply
from torch import Tensor


class DistanceConstrainedField(HardBoundaryField):
    """Hard Dirichlet cage ``u = g + phi * NN`` driven by an SDF / ADF.

    Parameters
    ----------
    base
        Free network field.
    distance_fn
        Torch-differentiable ``phi(coords) -> (n,)``. Prefer a normalized ADF
        (:func:`~omnibias.pinn.domain.torch.sdf_torch.normalize_distance`).
    boundary_value_fn
        Optional ``g(coords) -> {component: values}``. Defaults to zero.
    normalize
        If True and ``sdf`` is given, wrap the primitive distance with ADF
        normalization.
    sdf
        Optional numpy primitive (:class:`Sphere` / :class:`Box` /
        :class:`Halfspace`) used to build ``distance_fn`` when the latter is
        omitted.
    """

    def __init__(
        self,
        *,
        base: FieldBase,
        distance_fn: DistanceFn | None = None,
        boundary_value_fn: Callable[[Tensor], dict[str, Tensor]] | None = None,
        bounded_names: Sequence[str] | None = None,
        passthrough_names: tuple[str, ...] = (),
        groups: dict[str, tuple[str, ...]] | None = None,
        max_derivative_order: int = 4,
        sdf: Sphere | Box | Halfspace | None = None,
        normalize: bool = True,
    ) -> None:
        if distance_fn is None:
            if sdf is None:
                raise ValueError("provide distance_fn or sdf")
            distance_fn = from_primitive(sdf)
            if normalize:
                distance_fn = normalize_distance(distance_fn)
        elif normalize and sdf is not None:
            # Caller gave both; still honor normalize on the provided fn.
            distance_fn = normalize_distance(distance_fn)
        super().__init__(
            base=base,
            distance_fn=distance_fn,
            boundary_value_fn=boundary_value_fn,
            bounded_names=bounded_names,
            passthrough_names=passthrough_names,
            groups=groups,
            max_derivative_order=max_derivative_order,
        )
        self.sdf = sdf
        self._normalize = bool(normalize)

    def phi(self, coords: Tensor) -> Tensor:
        """Evaluate the distance / ADF at ``coords``."""
        return self.distance_fn(coords)

    def product_jet_at(
        self,
        x0: Tensor,
        nn_jet: Tensor,
        *,
        order: int,
        phi_jet: Tensor | None = None,
    ) -> Tensor:
        """Exact product jet ``jet(phi) * jet(NN)`` at a single point ``x0``.

        ``nn_jet`` has shape ``(M, C)`` (or ``(M,)``); ``phi_jet`` defaults to
        a first-order identity-composed finite-difference jet of ``phi`` when
        omitted (callers with an analytic ``phi`` jet should pass it).
        """
        if x0.ndim != 1:
            raise ValueError(f"x0 must be 1-D, got shape {tuple(x0.shape)}")
        dim = int(x0.shape[0])
        m = num_multi_indices(dim, order)
        if nn_jet.shape[0] != m:
            raise ValueError(
                f"nn_jet leading dim {nn_jet.shape[0]} != M={m} for "
                f"dim={dim}, order={order}"
            )
        if phi_jet is None:
            # Build a first-order jet of phi via autograd at x0; higher rows 0.
            # Exact for linear SDFs (halfspaces); approximate beyond order 1
            # for curved primitives -- pass an analytic phi_jet for those.
            phi_jet = _phi_jet_autograd(self.distance_fn, x0, order)
        return jet_multiply(phi_jet, nn_jet, dim, order)


def _phi_jet_autograd(
    distance_fn: DistanceFn, x0: Tensor, order: int
) -> Tensor:
    """Multivariate jet of ``phi`` at ``x0`` via autograd (rows above 1 zeroed
    unless order requires them -- we fill order-1 exactly)."""
    from omnibias.core.multi_index import index_position

    dim = int(x0.shape[0])
    m = num_multi_indices(dim, order)
    pos = index_position(dim, order)
    out = torch.zeros(m, dtype=x0.dtype, device=x0.device)
    x = x0.detach().requires_grad_(True)
    val = distance_fn(x.unsqueeze(0))[0]
    out[pos[(0,) * dim]] = val
    if order >= 1:
        (g,) = torch.autograd.grad(val, x, create_graph=True)
        for i in range(dim):
            alpha = tuple(1 if j == i else 0 for j in range(dim))
            # Jet stores D^alpha / alpha!; alpha! = 1 for |alpha|=1.
            out[pos[alpha]] = g[i]
    return out


def build_distance_constrained_field(
    base: FieldBase,
    sdf: Sphere | Box | Halfspace,
    *,
    boundary_value_fn: Callable[[Tensor], dict[str, Tensor]] | None = None,
    normalize: bool = True,
    **kwargs: Any,
) -> DistanceConstrainedField:
    """Convenience builder from a numpy SDF primitive."""
    return DistanceConstrainedField(
        base=base,
        sdf=sdf,
        boundary_value_fn=boundary_value_fn,
        normalize=normalize,
        **kwargs,
    )


__all__ = [
    "DistanceConstrainedField",
    "build_distance_constrained_field",
]

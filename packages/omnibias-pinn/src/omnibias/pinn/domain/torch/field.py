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
from omnibias.pinn.domain._core.boundary import (
    BCMode,
    NonSmoothBoundaryError,
    assert_smooth_for_normal_bc,
)
from omnibias.pinn.domain._core.sdf import SDF, Box, Halfspace, RCompose, Sphere
from omnibias.pinn.domain.torch.boundary_torch import boundary_factor_jet_at
from omnibias.pinn.domain.torch.sdf_torch import (
    DistanceFn,
    from_primitive,
    from_sdf,
    normalize_distance,
)
from omnibias.pinn.torch.cage.conservation import HardBoundaryField
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.torch.jet_mv import jet_multiply
from torch import Tensor


def _wrap_bc_factor(
    phi_fn: DistanceFn,
    *,
    mode: BCMode,
    robin_alpha: float,
    robin_beta: float,
) -> DistanceFn:
    if mode == "dirichlet":
        return phi_fn

    def _fn(coords: Tensor) -> Tensor:
        phi = phi_fn(coords)
        if mode == "neumann":
            return phi * phi
        return robin_alpha * phi + robin_beta * phi * phi

    return _fn


class DistanceConstrainedField(HardBoundaryField):
    """Hard BC cage ``u = g + phi * NN`` driven by an SDF / ADF.

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
        :class:`Halfspace` / :class:`RCompose`) used to build ``distance_fn``
        when the latter is omitted.
    bc_mode
        ``"dirichlet"`` (default), ``"neumann"`` (``phi^2`` factor), or
        ``"robin"`` (``alpha*phi + beta*phi^2``). Neumann / Robin require a
        smooth boundary (no CSG junctions).
    robin_alpha, robin_beta
        Coefficients for ``bc_mode="robin"``.
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
        sdf: SDF | None = None,
        normalize: bool = True,
        bc_mode: BCMode = "dirichlet",
        robin_alpha: float = 1.0,
        robin_beta: float = 0.0,
    ) -> None:
        if bc_mode in ("neumann", "robin") and isinstance(sdf, RCompose):
            raise NonSmoothBoundaryError(
                "Neumann / Robin BCs on RCompose domains require smooth "
                "primitives without junctions; use Dirichlet or a single primitive"
            )
        if distance_fn is None:
            if sdf is None:
                raise ValueError("provide distance_fn or sdf")
            phi_fn = from_sdf(sdf) if not isinstance(sdf, (Sphere, Box, Halfspace)) else from_primitive(sdf)
            if normalize:
                phi_fn = normalize_distance(phi_fn)
            distance_fn = _wrap_bc_factor(
                phi_fn,
                mode=bc_mode,
                robin_alpha=robin_alpha,
                robin_beta=robin_beta,
            )
        elif normalize and sdf is not None and isinstance(sdf, (Sphere, Box, Halfspace)):
            distance_fn = _wrap_bc_factor(
                normalize_distance(distance_fn),
                mode=bc_mode,
                robin_alpha=robin_alpha,
                robin_beta=robin_beta,
            )
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
        self.bc_mode: BCMode = bc_mode
        self.robin_alpha = float(robin_alpha)
        self.robin_beta = float(robin_beta)

    def phi(self, coords: Tensor) -> Tensor:
        """Evaluate the distance / ADF at ``coords``."""
        if self.bc_mode in ("neumann", "robin") and self.sdf is not None:
            assert_smooth_for_normal_bc(
                self.sdf, coords.detach().cpu().numpy()
            )
        return self.distance_fn(coords)

    def product_jet_at(
        self,
        x0: Tensor,
        nn_jet: Tensor,
        *,
        order: int,
        phi_jet: Tensor | None = None,
    ) -> Tensor:
        """Exact product jet ``jet(phi) * jet(NN)`` at a single point ``x0``."""
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
            if self.sdf is None:
                phi_jet = _phi_jet_autograd(self.distance_fn, x0, order)
            else:
                if self.bc_mode in ("neumann", "robin"):
                    assert_smooth_for_normal_bc(
                        self.sdf, x0.detach().cpu().numpy().reshape(1, -1)
                    )
                phi_jet = boundary_factor_jet_at(
                    self.sdf,
                    x0,
                    order=order,
                    mode=self.bc_mode,
                    normalize=self._normalize,
                    robin_alpha=self.robin_alpha,
                    robin_beta=self.robin_beta,
                )
        return jet_multiply(phi_jet, nn_jet, dim, order)


def _phi_jet_autograd(
    distance_fn: DistanceFn, x0: Tensor, order: int
) -> Tensor:
    """Multivariate jet of ``phi`` at ``x0`` via autograd (full requested order)."""
    from omnibias.core.multi_index import index_position, multi_indices

    dim = int(x0.shape[0])
    m = num_multi_indices(dim, order)
    pos = index_position(dim, order)
    out = torch.zeros(m, dtype=x0.dtype, device=x0.device)
    x = x0.detach().requires_grad_(True)
    val = distance_fn(x.unsqueeze(0))[0]
    out[pos[(0,) * dim]] = val

    def _partial(current: Tensor, alpha: tuple[int, ...]) -> Tensor:
        cur = current
        coords = x
        for ax, power in zip(range(dim), alpha, strict=False):
            for _ in range(power):
                (grad,) = torch.autograd.grad(
                    cur, coords, create_graph=True, retain_graph=True
                )
                cur = grad[ax]
        return cur

    for alpha in multi_indices(dim, order):
        if sum(alpha) == 0:
            continue
        out[pos[alpha]] = _partial(val, alpha)
    return out


def build_distance_constrained_field(
    base: FieldBase,
    sdf: SDF,
    *,
    boundary_value_fn: Callable[[Tensor], dict[str, Tensor]] | None = None,
    normalize: bool = True,
    bc_mode: BCMode = "dirichlet",
    robin_alpha: float = 1.0,
    robin_beta: float = 0.0,
    **kwargs: Any,
) -> DistanceConstrainedField:
    """Convenience builder from a numpy SDF primitive."""
    return DistanceConstrainedField(
        base=base,
        sdf=sdf,
        boundary_value_fn=boundary_value_fn,
        normalize=normalize,
        bc_mode=bc_mode,
        robin_alpha=robin_alpha,
        robin_beta=robin_beta,
        **kwargs,
    )


__all__ = [
    "DistanceConstrainedField",
    "build_distance_constrained_field",
]

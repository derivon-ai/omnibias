# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""``PartitionedField`` -- a discontinuity-capturing PINN field on omnibias-partition.

A single smooth activation network cannot represent a kink / shock; a **partition of unity**
of smooth sub-solutions can. :class:`PartitionedField` wraps

* a soft partition (``omnibias.partition`` split gates over the coordinates), and
* one :class:`~omnibias.pinn.torch.fields.OneLayerVectorField` sub-solution per region,

and evaluates the blend

.. math:: u(x) = \sum_l w_l(x)\, u_l(x),

a genuine field that plugs into the existing PINN ops. As the gate sharpness
``beta -> inf`` the partition hardens and ``u`` develops a genuine interface between regions.

Honesty
-------
Derivatives of ``u`` go through the **autodiff product rule** (``torch.autograd``), *not* the
closed-form ``sigma``-tower -- the tower does not cover products of sigmoids (a
Faa-di-Bruno-on-products tower is future work). The field therefore sets
``_omnibias_dispatch = "partitioned"`` so the fields ops route it to the autodiff
state-method path (exactly like the spectral / cage fields), never the closed-form path.

Terminology: the gate's ``beta -> inf`` hardening is the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit
to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


class PartitionedField(FieldBase):
    r"""Discontinuity-capturing field ``u(x) = sum_l w_l(x) u_l(x)`` over a soft partition.

    Parameters
    ----------
    coordinate_spec, components:
        The shared input-axis / output-channel metadata (all sub-solutions use these).
    subfields:
        The ``2**depth`` region sub-solutions (:class:`OneLayerVectorField`), one per region.
    split_dirs:
        ``(depth, D)`` oblique split directions of the partition gates.
    split_thresh:
        ``(depth,)`` split thresholds; the gate ``l`` is ``sigmoid(beta (W_l . x - t_l))``.
    beta:
        Gate sharpness. Larger -> sharper interface (the ``beta -> inf`` hard-partition limit).
    trainable_partition:
        If ``True`` (default) the split is a learnable parameter (the interface can move);
        otherwise it is a fixed buffer (a prescribed / a-priori interface).
    """

    _omnibias_dispatch = "partitioned"

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        subfields: Sequence[OneLayerVectorField],
        split_dirs: Tensor,
        split_thresh: Tensor,
        beta: float = 8.0,
        trainable_partition: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(coordinate_spec=coordinate_spec, components=components)
        W = torch.as_tensor(split_dirs, dtype=dtype)
        t = torch.as_tensor(split_thresh, dtype=dtype)
        if W.dim() != 2:
            raise ValueError(f"split_dirs must be (depth, D), got shape {tuple(W.shape)}")
        depth, D = W.shape
        if D != coordinate_spec.ndim:
            raise ValueError(f"split_dirs D={D} != coordinate ndim {coordinate_spec.ndim}")
        if t.shape != (depth,):
            raise ValueError(f"split_thresh must be (depth,)={depth}, got {tuple(t.shape)}")
        n_regions = 1 << depth
        if len(subfields) != n_regions:
            raise ValueError(
                f"expected {n_regions} subfields (2**depth for depth={depth}), "
                f"got {len(subfields)}"
            )
        self.depth = int(depth)
        self.n_regions = int(n_regions)
        self.beta = float(beta)
        self.subfields = nn.ModuleList(subfields)
        if trainable_partition:
            self.split_W = nn.Parameter(W)
            self.split_t = nn.Parameter(t)
        else:
            self.register_buffer("split_W", W)
            self.register_buffer("split_t", t)

    # -- FieldBase plumbing: no single pre-activation tower (autodiff path) --------- #
    def _pre_activations(self, coords: Tensor) -> Tensor | None:
        return None

    # -- the blended forward u = sum_l w_l(x) u_l(x) -------------------------------- #
    def partition_weights(self, coords: Tensor, beta: float | None = None) -> Tensor:
        r"""Soft partition-of-unity weights ``(B, n_regions)`` over the coordinates."""
        from omnibias.partition.torch.weights import partition_weights_arrays

        b = self.beta if beta is None else float(beta)
        w: Tensor = partition_weights_arrays(self.split_W, self.split_t, coords, b, self.depth)
        return w

    def _subfield_values(self, coords: Tensor) -> Tensor:
        r"""Each region sub-solution's raw values, stacked ``(B, n_regions, C)``."""
        outs = []
        for module in self.subfields:
            sub = cast(OneLayerVectorField, module)
            z = sub._pre_activations(coords)  # (B, H)
            outs.append(sub.value_all(sub._sigma(z)))  # (B, C)
        return torch.stack(outs, dim=1)

    def forward_values(self, coords: Tensor, beta: float | None = None) -> Tensor:
        r"""Blended field values ``u(x) = sum_l w_l(x) u_l(x)`` of shape ``(B, C)``."""
        w = self.partition_weights(coords, beta)  # (B, L)
        o = self._subfield_values(coords)  # (B, L, C)
        return torch.einsum("bl,blc->bc", w, o)

    # -- state-method path consumed by the fields ops dispatch ("partitioned") ------ #
    def value_component(self, state: FieldState, name: str) -> Tensor:
        ci = self.components.index(name)
        return self.forward_values(state.coords)[:, ci]

    def _grad_along(self, u: Tensor, x: Tensor, axis: int) -> Tensor:
        g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]  # (B, D)
        return g[:, axis]

    def derivative(self, state: FieldState, name: str, *, axis: int, order: int = 1) -> Tensor:
        r"""``d^order u_name / dx_axis^order`` via the autodiff product rule."""
        ci = self.components.index(name)
        x = state.coords
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        u = self.forward_values(x)[:, ci]  # (B,)
        for _ in range(order):
            u = self._grad_along(u, x, axis)
        return u

    def mixed_partial(
        self, state: FieldState, name: str, axes: tuple[int, ...], orders: tuple[int, ...]
    ) -> Tensor:
        ci = self.components.index(name)
        x = state.coords
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        u = self.forward_values(x)[:, ci]  # (B,)
        for a, o in zip(axes, orders, strict=False):
            for _ in range(int(o)):
                u = self._grad_along(u, x, a)
        return u

    def __repr__(self) -> str:
        return (
            f"PartitionedField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, n_regions={self.n_regions}, "
            f"beta={self.beta})"
        )


def build_partitioned_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    split_dirs: Tensor,
    split_thresh: Tensor,
    hidden: int = 16,
    base: str = "tanh",
    beta: float = 8.0,
    trainable_partition: bool = True,
    seed: int | None = 0,
    dtype: torch.dtype = torch.float64,
) -> PartitionedField:
    r"""Convenience builder: one :class:`OneLayerVectorField` per region + the split.

    ``split_dirs`` is ``(depth, D)`` and ``split_thresh`` is ``(depth,)``; the field has
    ``2**depth`` regions. Each region sub-solution is an independent small omnibias field.
    """
    W = torch.as_tensor(split_dirs, dtype=dtype)
    depth = W.shape[0]
    n_regions = 1 << depth
    if seed is not None:
        torch.manual_seed(seed)
    subfields = [
        OneLayerVectorField(
            coordinate_spec=coordinate_spec,
            components=components,
            hidden=hidden,
            base=base,
            dtype=dtype,
        )
        for _ in range(n_regions)
    ]
    return PartitionedField(
        coordinate_spec=coordinate_spec,
        components=components,
        subfields=subfields,
        split_dirs=split_dirs,
        split_thresh=split_thresh,
        beta=beta,
        trainable_partition=trainable_partition,
        dtype=dtype,
    )


__all__ = ["PartitionedField", "build_partitioned_field"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard density-matrix cage: ``rho = G G^dag / Tr(G G^dag)`` (torch).

The base field's split-real channels are interpreted as a generator
``G``. The cage exposes the same names as the unique trace-1 PSD
``rho``, so Hermiticity, trace-1, and positivity hold by construction
-- the same algebraic-renormalize-every-forward-pass move as
:class:`NormConservationField`. First-order derivatives use the product
/ quotient rule; higher orders are refused.

Do not conflate this with founding bias collapse (``delta -> 0``) or
temperature collapse (``beta -> inf``, feasibility sense).
"""

from __future__ import annotations

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.cage.incompressible import _CageFieldBase
from omnibias.pinn.torch.fields.base import FieldBase, _import_torch_ops
from omnibias.qpinn._core.density import parse_rho_entry_name, rho_entry_names
from torch import Tensor


def _assemble_G(inner: FieldState, group: str, dim: int, *, axis=None, order: int = 0) -> Tensor:
    rows: list[list[Tensor]] = []
    for i in range(dim):
        row: list[Tensor] = []
        for j in range(dim):
            re_name, im_name = rho_entry_names(group, i, j)
            if axis is None or order == 0:
                re_v = inner.ops.value(inner, re_name)
                im_v = inner.ops.value(inner, im_name)
            else:
                re_v = inner.ops.derivative(inner, re_name, axis=axis, order=order)
                im_v = inner.ops.derivative(inner, im_name, axis=axis, order=order)
            row.append(torch.complex(re_v, im_v))
        rows.append(row)
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def _rho_from_G(gen: Tensor) -> Tensor:
    dag = gen.conj().transpose(-2, -1)
    gram = gen @ dag
    trace = torch.einsum("...ii->...", gram).real
    eps = torch.finfo(trace.dtype).tiny
    return gram / (trace.clamp_min(eps).unsqueeze(-1).unsqueeze(-1))


def _drho_from_G(gen: Tensor, dgen: Tensor) -> Tensor:
    dag = gen.conj().transpose(-2, -1)
    ddag = dgen.conj().transpose(-2, -1)
    gram = gen @ dag
    dgram = dgen @ dag + gen @ ddag
    trace = torch.einsum("...ii->...", gram).real
    dtrace = torch.einsum("...ii->...", dgram).real
    eps = torch.finfo(trace.dtype).tiny
    nrm = trace.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    dn = dtrace.unsqueeze(-1).unsqueeze(-1)
    return (dgram * nrm - gram * dn) / (nrm * nrm)


class DensityMatrixField(_CageFieldBase):
    """Hard PSD / trace-1 cage for a split-real density matrix."""

    rho_group_name: str
    dim: int

    @property
    def _omnibias_readout_independent(self) -> bool:
        return False

    def __init__(self, *, base: FieldBase, group: str = "rho", dim: int = 2) -> None:
        if not base.components.is_group(group):
            raise ValueError(
                f"base does not have a density-matrix group {group!r}; "
                "build it with omnibias.qpinn.make_rho_components"
            )
        members = base.components.group_members(group)
        expected = 2 * dim * dim
        if len(members) != expected:
            raise ValueError(
                f"group {group!r} must have {expected} components; got {len(members)}"
            )
        passthrough = tuple(n for n in base.components.names if n not in members)
        super().__init__(
            base=base,
            velocity_names=members,
            passthrough_names=passthrough,
            groups={group: members},
        )
        self.rho_group_name = group
        self.dim = dim

    def evaluate(self, coords: Tensor) -> FieldState[Tensor]:
        if coords.dim() != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != "
                f"coordinate_spec.ndim {self.coordinate_spec.ndim}"
            )
        inner_state = self.base.evaluate(coords)
        gen = _assemble_G(inner_state, self.rho_group_name, self.dim)
        rho = _rho_from_G(gen)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_torch_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={"_cage_inner_state": inner_state, "_rho": rho, "_G": gen},
        )

    def value_component(self, state: FieldState, name: str) -> Tensor:
        if name in self.passthrough_names:
            inner = state.extra["_cage_inner_state"]
            return inner.ops.value(inner, name)
        part, i, j = parse_rho_entry_name(name, group=self.rho_group_name)
        entry = state.extra["_rho"][..., i, j]
        return entry.real if part == "re" else entry.imag

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1
    ) -> Tensor:
        if order == 0:
            return self.value_component(state, name)
        if name in self.passthrough_names:
            inner = state.extra["_cage_inner_state"]
            return inner.ops.derivative(inner, name, axis=axis, order=order)
        if order != 1:
            raise NotImplementedError("DensityMatrixField derivatives are first-order only")
        inner = state.extra["_cage_inner_state"]
        gen = state.extra["_G"]
        dgen = _assemble_G(inner, self.rho_group_name, self.dim, axis=axis, order=1)
        drho = _drho_from_G(gen, dgen)
        part, i, j = parse_rho_entry_name(name, group=self.rho_group_name)
        entry = drho[..., i, j]
        return entry.real if part == "re" else entry.imag

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        if not orders or max(orders) == 0:
            return self.value_component(state, name)
        if len(orders) == 1 and orders[0] == 1:
            return self.derivative(state, name, axis=axes[0], order=1)
        raise NotImplementedError("DensityMatrixField mixed partials are first-order only")


def make_density_matrix_field(
    *, base: FieldBase, group: str = "rho", dim: int = 2
) -> DensityMatrixField:
    """Build a :class:`DensityMatrixField` around a generator field."""
    return DensityMatrixField(base=base, group=group, dim=dim)


__all__ = ["DensityMatrixField", "make_density_matrix_field"]

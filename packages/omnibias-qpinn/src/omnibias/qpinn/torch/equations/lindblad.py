# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""GKSL / Lindblad density-matrix residual (torch backend).

The residual is ``d rho/dt - L[rho]`` on the split-real channels of a
``d x d`` density matrix, using closed-form time derivatives from
``state.ops.derivative``. The GKSL generator is a caller input, not a
Born–Markov derivation.

Do not conflate this with founding bias collapse (``delta -> 0``) or
temperature collapse (``beta -> inf``, feasibility sense).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.density import rho_entry_names
from omnibias.qpinn.torch.equations._types import LindbladOutput
from omnibias.torch.lindblad import apply_lindblad
from torch import Tensor


def _complex_entry(
    state: FieldState,
    group: str,
    i: int,
    j: int,
    *,
    axis: int | str | None = None,
    order: int = 0,
) -> Tensor:
    re_name, im_name = rho_entry_names(group, i, j)
    if axis is None or order == 0:
        re_v = state.ops.value(state, re_name)
        im_v = state.ops.value(state, im_name)
    else:
        re_v = state.ops.derivative(state, re_name, axis=axis, order=order)
        im_v = state.ops.derivative(state, im_name, axis=axis, order=order)
    return torch.complex(re_v, im_v)


def _assemble(state: FieldState, group: str, dim: int, *, axis=None, order: int = 0) -> Tensor:
    rows: list[list[Tensor]] = []
    for i in range(dim):
        row = [_complex_entry(state, group, i, j, axis=axis, order=order) for j in range(dim)]
        rows.append(row)
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


@dataclass
class Lindblad:
    """Configurable GKSL residual on a split-real density matrix."""

    hamiltonian: Tensor
    jumps: Tensor
    rates: Tensor
    group: str = "rho"
    dim: int = 2
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> LindbladOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("Lindblad residual requires a time axis in the coordinate spec")
        members = state.components.group_members(self.group)
        expected = 2 * self.dim * self.dim
        if len(members) != expected:
            raise KeyError(
                f"group {self.group!r} must have {expected} channels; "
                "build the field with omnibias.qpinn.make_rho_components"
            )
        rho = _assemble(state, self.group, self.dim)
        rho_t = _assemble(state, self.group, self.dim, axis=time, order=1)
        rhs = apply_lindblad(rho, self.hamiltonian, self.jumps, self.rates)
        residual_c = rho_t - rhs
        residual = torch.stack([residual_c.real, residual_c.imag], dim=-1)
        residual = residual.reshape(residual.shape[0], -1)
        if self.source is not None:
            residual = residual - self.source(state)
        return LindbladOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def lindblad(
    state: FieldState,
    hamiltonian: Tensor,
    jumps: Tensor,
    rates: Tensor,
    *,
    group: str = "rho",
    dim: int = 2,
    source: Callable[[FieldState], Tensor] | None = None,
) -> LindbladOutput:
    """Stateless one-shot wrapper around :class:`Lindblad`."""
    return Lindblad(
        hamiltonian=hamiltonian,
        jumps=jumps,
        rates=rates,
        group=group,
        dim=dim,
        source=source,
    )(state)


__all__ = ["Lindblad", "lindblad"]

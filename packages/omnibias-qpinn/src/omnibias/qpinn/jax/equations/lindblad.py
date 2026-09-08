# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""GKSL / Lindblad density-matrix residual (JAX backend).

JAX twin of :mod:`omnibias.qpinn.torch.equations.lindblad`.

Do not conflate this with founding bias collapse (``delta -> 0``) or
temperature collapse (``beta -> inf``, feasibility sense).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.jax.lindblad import apply_lindblad
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.density import rho_entry_names
from omnibias.qpinn.jax.equations._types import LindbladOutput


def _complex_entry(
    state: FieldState,
    group: str,
    i: int,
    j: int,
    *,
    axis: int | str | None = None,
    order: int = 0,
) -> Array:
    re_name, im_name = rho_entry_names(group, i, j)
    if axis is None or order == 0:
        re_v = state.ops.value(state, re_name)
        im_v = state.ops.value(state, im_name)
    else:
        re_v = state.ops.derivative(state, re_name, axis=axis, order=order)
        im_v = state.ops.derivative(state, im_name, axis=axis, order=order)
    return re_v + 1j * im_v


def _assemble(state: FieldState, group: str, dim: int, *, axis=None, order: int = 0) -> Array:
    rows: list[Array] = []
    for i in range(dim):
        row = [_complex_entry(state, group, i, j, axis=axis, order=order) for j in range(dim)]
        rows.append(jnp.stack(row, axis=-1))
    return jnp.stack(rows, axis=-2)


@dataclass
class Lindblad:
    """Configurable GKSL residual on a split-real density matrix (JAX)."""

    hamiltonian: Array
    jumps: Array
    rates: Array
    group: str = "rho"
    dim: int = 2
    source: Callable[[FieldState], Array] | None = None

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
        residual = jnp.stack([jnp.real(residual_c), jnp.imag(residual_c)], axis=-1)
        residual = residual.reshape((residual.shape[0], -1))
        if self.source is not None:
            residual = residual - self.source(state)
        return LindbladOutput(
            residual=residual,
            diag={"mean_sq_residual": float(jnp.mean(residual**2))},
        )


def lindblad(
    state: FieldState,
    hamiltonian: Array,
    jumps: Array,
    rates: Array,
    *,
    group: str = "rho",
    dim: int = 2,
    source: Callable[[FieldState], Array] | None = None,
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

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable GKSL / Lindblad dynamics (torch twin; theory 09-32).

Bit-identical (given the same matrix exponential) twin of
:mod:`omnibias.core.lindblad`. The superoperator is assembled with
``torch.kron``; the propagator is ``torch.linalg.matrix_exp``. This is
the linear-semigroup analogue of the sigma tower -- one propagator
evaluation regardless of derivative order -- and is labelled as such,
not as the activation tower.

``order`` is a static Python ``int`` (branches only at trace time), so
:func:`derivative_tower` stays ``jit`` / functorch-vmap safe. Default
dtype is the framework default, never a hardcoded ``float32``.

Two collapse senses are named (mirroring the core module) and must not
be conflated: the founding bias collapse (``delta -> 0``) never appears
here; ``beta -> inf`` is the founding **temperature collapse**
(feasibility sense) and is not evaluated or requested by anything here
-- see :mod:`omnibias.core.occupancy` for the honest T=0 reference and
:mod:`omnibias.core.collapse.relaxation` for the new ``relaxation``
slot. Do not conflate the two.

The GKSL form is a caller input. Not a Born–Markov derivation, not a
general closed form, no thermodynamic limit.
"""

from __future__ import annotations

from typing import cast

from omnibias.torch.occupancy import occupancy

import torch
from torch import Tensor


def _complex_dtype() -> torch.dtype:
    if torch.get_default_dtype() == torch.float64:
        return torch.complex128
    return torch.complex64


def liouvillian(hamiltonian: Tensor, jumps: Tensor, rates: Tensor) -> Tensor:
    """Complex superoperator ``L`` of shape ``(d**2, d**2)``, column-stacked."""
    ham = hamiltonian.to(_complex_dtype())
    dim = ham.shape[-1]
    eye = torch.eye(dim, dtype=ham.dtype, device=ham.device)
    gen = -1j * (torch.kron(eye, ham) - torch.kron(ham.transpose(-2, -1).contiguous(), eye))
    jump_ops = jumps.to(dtype=ham.dtype)
    rate_ops = rates.to(dtype=ham.real.dtype)
    for k in range(jump_ops.shape[0]):
        jump = jump_ops[k]
        dag = jump.conj().transpose(-2, -1).contiguous()
        left = dag @ jump
        dissipator = torch.kron(jump.conj().contiguous(), jump) - 0.5 * (
            torch.kron(eye, left) + torch.kron(left.transpose(-2, -1).contiguous(), eye)
        )
        gen = gen + rate_ops[k].to(dtype=ham.dtype) * dissipator
    return gen


def _vec(rho: Tensor) -> Tensor:
    dim = rho.shape[-1]
    return rho.transpose(-2, -1).contiguous().reshape(*rho.shape[:-2], dim * dim)


def _unvec(vector: Tensor, dim: int) -> Tensor:
    return vector.reshape(*vector.shape[:-1], dim, dim).transpose(-2, -1)


def propagator(hamiltonian: Tensor, jumps: Tensor, rates: Tensor, time: Tensor | float) -> Tensor:
    """``exp(t L)``."""
    gen = liouvillian(hamiltonian, jumps, rates)
    return cast(Tensor, torch.linalg.matrix_exp(time * gen))


def evolve(
    rho0: Tensor,
    hamiltonian: Tensor,
    jumps: Tensor,
    rates: Tensor,
    time: Tensor | float,
) -> Tensor:
    """``rho(t) = exp(t L) rho(0)``."""
    dim = rho0.shape[-1]
    prop = propagator(hamiltonian, jumps, rates, time)
    vec = _vec(rho0.to(dtype=prop.dtype))
    return _unvec(prop @ vec, dim)


def apply_lindblad(rho: Tensor, hamiltonian: Tensor, jumps: Tensor, rates: Tensor) -> Tensor:
    """Evaluate ``L[rho]`` by the GKSL formula."""
    ham = hamiltonian.to(_complex_dtype())
    rho_c = rho.to(dtype=ham.dtype)
    out = -1j * (ham @ rho_c - rho_c @ ham)
    jump_ops = jumps.to(dtype=ham.dtype)
    rate_ops = rates.to(dtype=ham.real.dtype)
    for k in range(jump_ops.shape[0]):
        jump = jump_ops[k]
        dag = jump.conj().transpose(-2, -1).contiguous()
        lindblad = jump @ rho_c @ dag - 0.5 * (dag @ jump @ rho_c + rho_c @ dag @ jump)
        out = out + rate_ops[k].to(dtype=ham.dtype) * lindblad
    return out


def derivative_tower(
    rho0: Tensor,
    hamiltonian: Tensor,
    jumps: Tensor,
    rates: Tensor,
    time: Tensor | float,
    *,
    order: int,
) -> Tensor:
    """Stack ``(rho, L rho, ..., L^order rho)`` at time ``t``. ``order`` is static."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    gen = liouvillian(hamiltonian, jumps, rates)
    dim = rho0.shape[-1]
    vec = propagator(hamiltonian, jumps, rates, time) @ _vec(rho0.to(dtype=gen.dtype))
    rows: list[Tensor] = []
    for _ in range(order + 1):
        rows.append(_unvec(vec, dim))
        vec = gen @ vec
    return torch.stack(rows, dim=0)


def steady_state(hamiltonian: Tensor, jumps: Tensor, rates: Tensor) -> Tensor:
    """Unique trace-1 kernel vector of ``L`` via a linear solve."""
    gen = liouvillian(hamiltonian, jumps, rates)
    dim = hamiltonian.shape[-1]
    dim2 = dim * dim
    augmented = gen.clone()
    zero_row = torch.zeros(dim2, dtype=gen.dtype, device=gen.device)
    augmented = torch.cat([augmented[:-1], zero_row.unsqueeze(0)], dim=0)
    idx = torch.arange(dim, device=gen.device)
    augmented[-1, idx + idx * dim] = 1
    rhs = torch.zeros(dim2, dtype=gen.dtype, device=gen.device)
    rhs[-1] = 1
    vec = torch.linalg.solve(augmented, rhs)
    return _unvec(vec, dim)


def thermal_population(omega: Tensor, beta: Tensor) -> Tensor:
    """Excited-state Fermi occupancy ``sigma(-beta * omega)``."""
    mu = torch.zeros_like(omega)
    return occupancy(omega, beta, mu)


__all__ = [
    "apply_lindblad",
    "derivative_tower",
    "evolve",
    "liouvillian",
    "propagator",
    "steady_state",
    "thermal_population",
]

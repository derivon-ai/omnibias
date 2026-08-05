# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gauge connection ops (torch): field strength, covariant divergence, Bianchi.

The connection components ``A_mu^a`` live in a :class:`FieldState`; their partial
derivatives ``d_rho A_nu^a`` (and second partials, for the Yang-Mills operator)
come from the omnibias **closed-form** activation-derivative tower -- not autodiff
or finite differences. The numeric formulas are the shared backend-agnostic
kernels, so this file and its jax twin compute bit-identical values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import derivative, mixed_partial, value
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.connection import GaugeConnectionSpec
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra
from omnibias.geometry.gauge.torch.ops.algebra import structure_constants
from omnibias.geometry.gauge.torch.ops.hodge import levi_civita, signature_diagonal
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def connection_value(state: FieldState, conn: GaugeConnectionSpec) -> Tensor:
    r"""Connection values ``A_mu^a`` of shape ``(B, d, n)`` from a field state."""
    cols = []
    for mu in range(conn.spacetime_dim):
        names = conn.form.comps[(mu,)]
        cols.append(torch.stack([value(state, nm) for nm in names], dim=-1))
    return torch.stack(cols, dim=1)


def connection_partials(state: FieldState, conn: GaugeConnectionSpec) -> Tensor:
    r"""Closed-form ``d_rho A_nu^a`` of shape ``(B, rho, nu, a)``."""
    d = conn.spacetime_dim
    rows = []
    for rho in range(d):
        cols = []
        for nu in range(d):
            names = conn.form.comps[(nu,)]
            cols.append(
                torch.stack(
                    [derivative(state, nm, axis=rho, order=1) for nm in names], dim=-1
                )
            )
        rows.append(torch.stack(cols, dim=1))
    return torch.stack(rows, dim=1)


def connection_second_partials(state: FieldState, conn: GaugeConnectionSpec) -> Tensor:
    r"""Closed-form ``d_rho d_mu A_nu^a`` of shape ``(B, rho, mu, nu, a)``."""
    d = conn.spacetime_dim
    out_rho = []
    for rho in range(d):
        out_mu = []
        for mu in range(d):
            out_nu = []
            for nu in range(d):
                names = conn.form.comps[(nu,)]
                out_nu.append(
                    torch.stack(
                        [
                            mixed_partial(state, nm, (rho, mu), (1, 1))
                            for nm in names
                        ],
                        dim=-1,
                    )
                )
            out_mu.append(torch.stack(out_nu, dim=1))
        out_rho.append(torch.stack(out_mu, dim=1))
    return torch.stack(out_rho, dim=1)


def field_strength(state: FieldState, conn: GaugeConnectionSpec) -> Tensor:
    r"""Field strength ``F_{mu nu}^a`` of shape ``(B, d, d, n)`` from a field state."""
    a = connection_value(state, conn)
    da = connection_partials(state, conn)
    return field_strength_from_arrays(
        a, da, algebra=conn.algebra, coupling=conn.coupling
    )


def field_strength_from_arrays(
    A: Tensor, dA: Tensor, *, algebra: LieAlgebra, coupling: float
) -> Tensor:
    r"""Field strength from explicit ``A`` and ``dA`` arrays (no field state)."""
    f = structure_constants(algebra, dtype=A.dtype, device=A.device)
    return kernels.field_strength(torch, A, dA, f, coupling)


def covariant_divergence(
    state: FieldState,
    conn: GaugeConnectionSpec,
    *,
    signature: tuple[int, ...] | None = None,
) -> Tensor:
    r"""Yang-Mills operator ``(D_mu F^{mu nu})^a`` of shape ``(B, nu, a)``."""
    a = connection_value(state, conn)
    da = connection_partials(state, conn)
    dda = connection_second_partials(state, conn)
    return covariant_divergence_from_arrays(
        a,
        da,
        dda,
        algebra=conn.algebra,
        coupling=conn.coupling,
        signature=signature if signature is not None else conn.signature,
    )


def covariant_divergence_from_arrays(
    A: Tensor,
    dA: Tensor,
    ddA: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
) -> Tensor:
    r"""Yang-Mills operator from explicit ``A``, ``dA``, ``ddA`` arrays."""
    f = structure_constants(algebra, dtype=A.dtype, device=A.device)
    eta = signature_diagonal(signature, dtype=A.dtype, device=A.device)
    return kernels.covariant_divergence(torch, A, dA, ddA, f, coupling, eta)


def bianchi(
    state: FieldState,
    conn: GaugeConnectionSpec,
    *,
    signature: tuple[int, ...] | None = None,
) -> Tensor:
    r"""Bianchi operator ``(D_mu \tilde F^{mu nu})^a`` (identically zero)."""
    a = connection_value(state, conn)
    da = connection_partials(state, conn)
    dda = connection_second_partials(state, conn)
    return bianchi_from_arrays(
        a,
        da,
        dda,
        algebra=conn.algebra,
        coupling=conn.coupling,
        signature=signature if signature is not None else conn.signature,
    )


def bianchi_from_arrays(
    A: Tensor,
    dA: Tensor,
    ddA: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
) -> Tensor:
    r"""Bianchi operator from explicit ``A``, ``dA``, ``ddA`` arrays."""
    f = structure_constants(algebra, dtype=A.dtype, device=A.device)
    eta = signature_diagonal(signature, dtype=A.dtype, device=A.device)
    eps = levi_civita(A.shape[1], dtype=A.dtype, device=A.device)
    return kernels.bianchi(torch, A, dA, ddA, f, coupling, eta, eps)


def gauge_covariant_derivative(
    phi: Tensor,
    dphi: Tensor,
    A: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
) -> Tensor:
    r"""``(D_mu phi)^a = d_mu phi^a + g f^{abc} A_mu^b phi^c`` for an adjoint scalar.

    ``phi[B, a]``, ``dphi[B, mu, a] = d_mu phi^a`` -> ``(B, mu, a)``.
    """
    f = structure_constants(algebra, dtype=A.dtype, device=A.device)
    return kernels.covariant_derivative_adjoint(torch, phi, dphi, A, f, coupling)


def gauge_flow_rhs(
    state: FieldState,
    conn: GaugeConnectionSpec,
    *,
    signature: tuple[int, ...] | None = None,
    deturck: float = 0.0,
) -> Tensor:
    r"""Yang-Mills gradient-flow drift ``(D_mu F^{mu nu})^a`` of shape ``(B, nu, a)``.

    The deterministic part of the Parisi-Wu stochastic-quantisation flow; pass
    ``deturck != 0`` to add the DeTurck-Zwanziger gauge term.
    """
    a = connection_value(state, conn)
    da = connection_partials(state, conn)
    dda = connection_second_partials(state, conn)
    return gauge_flow_rhs_from_arrays(
        a,
        da,
        dda,
        algebra=conn.algebra,
        coupling=conn.coupling,
        signature=signature if signature is not None else conn.signature,
        deturck=deturck,
    )


def gauge_flow_rhs_from_arrays(
    A: Tensor,
    dA: Tensor,
    ddA: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
    deturck: float = 0.0,
) -> Tensor:
    r"""Yang-Mills gradient-flow drift from explicit ``A``, ``dA``, ``ddA`` arrays."""
    f = structure_constants(algebra, dtype=A.dtype, device=A.device)
    eta = signature_diagonal(signature, dtype=A.dtype, device=A.device)
    return kernels.yang_mills_gradient_flow_rhs(
        torch, A, dA, ddA, f, coupling, eta, deturck=deturck
    )


def langevin_step(
    A: Tensor,
    dA: Tensor,
    ddA: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
    dt: float,
    deturck: float = 0.0,
    temperature: float = 1.0,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""One Euler-Maruyama Langevin step ``A + dt * drift + sqrt(2 dt T) * xi``.

    Realizes the Parisi-Wu stochastic quantisation of Yang-Mills: the stationary
    measure of this stochastic flow is ``exp(-S[A]/T)``. Pass an explicit
    ``noise`` array (otherwise drawn from ``generator``) for reproducible /
    cross-backend-comparable trajectories.
    """
    rhs = gauge_flow_rhs_from_arrays(
        A, dA, ddA, algebra=algebra, coupling=coupling, signature=signature, deturck=deturck
    )
    if noise is None:
        noise = torch.randn(A.shape, dtype=A.dtype, device=A.device, generator=generator)
    return A + dt * rhs + math.sqrt(2.0 * dt * temperature) * noise


__all__ = [
    "bianchi",
    "bianchi_from_arrays",
    "connection_partials",
    "connection_second_partials",
    "connection_value",
    "covariant_divergence",
    "covariant_divergence_from_arrays",
    "field_strength",
    "field_strength_from_arrays",
    "gauge_covariant_derivative",
    "gauge_flow_rhs",
    "gauge_flow_rhs_from_arrays",
    "langevin_step",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The nonintegrable derivative (torch): parallel transport, holonomy, commutator.

Two faces of gauge nonintegrability, both anchored to the closed-form
:func:`~omnibias.geometry.gauge.torch.ops.connection.field_strength`:

* the **finite** side -- the Wu-Yang nonintegrable phase factor
  ``U(C) = P exp(-i g \int_C A_mu^a T^a dx^mu)`` (:func:`parallel_transport`)
  and its gauge-invariant trace (:func:`wilson_loop`). The connection and its
  derivatives are closed-form (sigma tower); the path-ordered exponential is a
  numerical ordered product of backend matrix exponentials over ``substeps``
  midpoint segments, converging to the continuum holonomy;
* the **local** side -- the covariant-derivative commutator
  ``[D_mu, D_nu] phi = g f\,F\,phi`` (:func:`covariant_derivative_commutator`),
  an exact closed-form identity whose defect (:func:`curvature_commutator_defect`)
  vanishes to machine precision.

The numeric formulas are the shared backend-agnostic kernels, so this file and
its jax twin agree: bit-identical for the commutator, and to ``rtol ~ 1e-9`` for
the transport (the matrix exponential is each backend's native primitive).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import derivative, mixed_partial, value
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.connection import GaugeConnectionSpec
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra
from omnibias.geometry.gauge.torch.ops.algebra import generators as algebra_generators
from omnibias.geometry.gauge.torch.ops.algebra import structure_constants
from omnibias.geometry.gauge.torch.ops.connection import connection_partials, connection_value
from torch import Tensor
from torch.func import jacfwd, vmap

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.fields._core.state import FieldState


# ----------------------------------------------------------------------------
# finite side: parallel transport / Wilson line / holonomy
# ----------------------------------------------------------------------------
def _transport_nodes(t0: float, t1: float, substeps: int, *, like: Tensor) -> Tensor:
    """Ordered midpoint parameters of ``substeps`` equal sub-intervals -> ``(N, 1)``."""
    dt = (t1 - t0) / substeps
    idx = torch.arange(substeps, dtype=like.dtype, device=like.device)
    return (t0 + (idx + 0.5) * dt).unsqueeze(-1)


def _tangent(curve: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Batched curve tangent ``r'(t)`` of shape ``(N, d)`` at param nodes ``x`` (N, 1)."""
    jac: Tensor = vmap(jacfwd(curve))(x)  # (N, d, 1)
    return jac[..., 0]


def parallel_transport_from_arrays(
    A_path: Tensor,
    tangents: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
    dt: float,
    generators: Tensor | None = None,
) -> Tensor:
    r"""Holonomy ``U(C) = P exp(-i g \int A_mu^a T^a dx^mu)`` from explicit arrays.

    ``A_path[N, d, a]`` is the connection at ``N`` ordered path points and
    ``tangents[N, d] = \dot x^mu`` the curve tangents; ``dt`` is the (uniform)
    parameter step.     The representation defaults to the fundamental of ``algebra``;
    pass ``generators[a, i, j]`` (e.g. ``spin_matrices``) for any other irrep.
    Returns the ``(dim, dim)`` transport matrix. The path ordering places the
    earliest-parameter segment on the left (``U = e^{X_0} e^{X_1} \cdots
    e^{X_{N-1}}``) -- the convention fixed so the infinitesimal-loop holonomy
    reproduces this package's :func:`field_strength` (verified by the non-abelian
    Stokes test).
    """
    gens = (
        generators
        if generators is not None
        else algebra_generators(algebra, dtype=A_path.dtype, device=A_path.device)
    )
    exponents = kernels.wilson_line_exponents(torch, A_path, tangents, gens, coupling, dt)
    segments = torch.linalg.matrix_exp(exponents)  # (N, dim, dim)
    dim = segments.shape[-1]
    u = torch.eye(dim, dtype=segments.dtype, device=segments.device)
    for k in range(segments.shape[0]):
        u = u @ segments[k]
    return u


def parallel_transport(
    state: FieldState,
    conn: GaugeConnectionSpec,
    curve: Callable[[Tensor], Tensor],
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    substeps: int = 32,
    generators: Tensor | None = None,
) -> Tensor:
    r"""Holonomy ``U(C)`` of the connection ``conn`` along ``curve`` -> ``(dim, dim)``.

    ``state`` must hold the connection components evaluated at the ordered path
    midpoints ``curve(transport_nodes(t0, t1, substeps))`` (the ``line_integral``
    pre-evaluation convention); ``curve`` is a bare callable mapping a ``(1,)``
    parameter to an ambient point ``(d,)``. The gradient of the connection is not
    needed -- only its value along the curve and the exact autodiff tangent
    ``r'(t)``; the path integral is a midpoint ordered product converging as
    ``substeps -> inf``.
    """
    a_path = connection_value(state, conn)  # (N, d, a)
    if a_path.shape[0] != substeps:
        raise ValueError(
            f"parallel_transport: state has {a_path.shape[0]} points but substeps="
            f"{substeps}; evaluate the field at curve(transport_nodes(t0, t1, substeps))"
        )
    mids = _transport_nodes(t0, t1, substeps, like=a_path)
    tangents = _tangent(curve, mids)  # (N, d)
    if tangents.shape[-1] != a_path.shape[1]:
        raise ValueError(
            f"parallel_transport: curve maps to {tangents.shape[-1]} ambient dims but "
            f"the connection spans {a_path.shape[1]} spacetime axes"
        )
    dt = (t1 - t0) / substeps
    return parallel_transport_from_arrays(
        a_path,
        tangents,
        algebra=conn.algebra,
        coupling=conn.coupling,
        dt=dt,
        generators=generators,
    )


def wilson_loop(
    state: FieldState,
    conn: GaugeConnectionSpec,
    curve: Callable[[Tensor], Tensor],
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    substeps: int = 32,
    generators: Tensor | None = None,
) -> Tensor:
    r"""Gauge-invariant Wilson loop ``(1/dim) Re tr U(C)`` for a closed ``curve``.

    Invariant under a global gauge transform ``A -> U A U^{-1}`` (the holonomy
    conjugates, its trace does not change). For a closed curve this is the
    nonintegrable phase factor of Wu & Yang; for an infinitesimal loop it detects
    the enclosed field strength. Returns a scalar tensor.
    """
    holonomy = parallel_transport(
        state, conn, curve, t0=t0, t1=t1, substeps=substeps, generators=generators
    )
    dim = holonomy.shape[-1]
    return torch.trace(holonomy).real / dim


# ----------------------------------------------------------------------------
# local side: covariant-derivative commutator = curvature action
# ----------------------------------------------------------------------------
def covariant_derivative_commutator_from_arrays(
    phi: Tensor,
    dphi: Tensor,
    ddphi: Tensor,
    A: Tensor,
    dA: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
) -> Tensor:
    r"""``([D_mu, D_nu] phi)^a`` for an adjoint scalar from explicit arrays -> ``(B, mu, nu, a)``."""
    f = structure_constants(algebra, dtype=A.dtype, device=A.device)
    return kernels.covariant_derivative_commutator(torch, phi, dphi, ddphi, A, dA, f, coupling)


def curvature_commutator_defect_from_arrays(
    phi: Tensor,
    dphi: Tensor,
    ddphi: Tensor,
    A: Tensor,
    dA: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
) -> Tensor:
    r"""Ricci-identity defect ``[D_mu, D_nu] phi - g f\,F\,phi`` (~ 0, closed form)."""
    f = structure_constants(algebra, dtype=A.dtype, device=A.device)
    commutator = kernels.covariant_derivative_commutator(torch, phi, dphi, ddphi, A, dA, f, coupling)
    fld = kernels.field_strength(torch, A, dA, f, coupling)
    action = kernels.curvature_action_on_adjoint(torch, fld, phi, f, coupling)
    return commutator - action


def _phi_arrays(
    state: FieldState, conn: GaugeConnectionSpec, phi_names: tuple[str, ...]
) -> tuple[Tensor, Tensor, Tensor]:
    """Adjoint scalar ``phi`` value / first / second partials from a field state."""
    if len(phi_names) != conn.algebra.dim:
        raise ValueError(
            f"phi_names has {len(phi_names)} components but algebra dim is {conn.algebra.dim}"
        )
    d = conn.spacetime_dim
    phi = torch.stack([value(state, nm) for nm in phi_names], dim=-1)
    dphi = torch.stack(
        [
            torch.stack([derivative(state, nm, axis=mu, order=1) for nm in phi_names], dim=-1)
            for mu in range(d)
        ],
        dim=1,
    )
    ddphi = torch.stack(
        [
            torch.stack(
                [
                    torch.stack(
                        [mixed_partial(state, nm, (mu, nu), (1, 1)) for nm in phi_names],
                        dim=-1,
                    )
                    for nu in range(d)
                ],
                dim=1,
            )
            for mu in range(d)
        ],
        dim=1,
    )
    return phi, dphi, ddphi


def covariant_derivative_commutator(
    state: FieldState, conn: GaugeConnectionSpec, phi_names: tuple[str, ...]
) -> Tensor:
    r"""``([D_mu, D_nu] phi)^a`` from a field state holding ``A`` and the adjoint scalar ``phi``."""
    phi, dphi, ddphi = _phi_arrays(state, conn, phi_names)
    a = connection_value(state, conn)
    da = connection_partials(state, conn)
    return covariant_derivative_commutator_from_arrays(
        phi, dphi, ddphi, a, da, algebra=conn.algebra, coupling=conn.coupling
    )


def curvature_commutator_defect(
    state: FieldState, conn: GaugeConnectionSpec, phi_names: tuple[str, ...]
) -> Tensor:
    r"""Ricci-identity defect from a field state (~ 0, closed form)."""
    phi, dphi, ddphi = _phi_arrays(state, conn, phi_names)
    a = connection_value(state, conn)
    da = connection_partials(state, conn)
    return curvature_commutator_defect_from_arrays(
        phi, dphi, ddphi, a, da, algebra=conn.algebra, coupling=conn.coupling
    )


def transport_nodes(t0: float, t1: float, substeps: int, *, like: Tensor) -> Tensor:
    """Public ordered midpoint parameters ``(N, 1)`` for the transport pre-evaluation."""
    return _transport_nodes(t0, t1, substeps, like=like)


__all__ = [
    "covariant_derivative_commutator",
    "covariant_derivative_commutator_from_arrays",
    "curvature_commutator_defect",
    "curvature_commutator_defect_from_arrays",
    "parallel_transport",
    "parallel_transport_from_arrays",
    "transport_nodes",
    "wilson_loop",
]

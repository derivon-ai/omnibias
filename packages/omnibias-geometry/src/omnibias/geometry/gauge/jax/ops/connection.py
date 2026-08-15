# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gauge connection ops (jax): field strength, covariant divergence, Bianchi.

The jax twin of :mod:`omnibias.geometry.gauge.torch.ops.connection`. Connection partials
``d_rho A_nu^a`` come from the omnibias closed-form derivative tower via the jax
:class:`FieldState` ops, and the numerics are the shared backend-agnostic kernels.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from omnibias.fields.jax.ops.basic import derivative, mixed_partial, value
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.connection import GaugeConnectionSpec
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra
from omnibias.geometry.gauge.jax.ops.algebra import structure_constants
from omnibias.geometry.gauge.jax.ops.hodge import levi_civita, signature_diagonal

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState

Array = Any


def connection_value(state: FieldState, conn: GaugeConnectionSpec) -> Array:
    r"""Connection values ``A_mu^a`` of shape ``(B, d, n)`` from a field state."""
    cols = []
    for mu in range(conn.spacetime_dim):
        names = conn.form.comps[(mu,)]
        cols.append(jnp.stack([value(state, nm) for nm in names], axis=-1))
    return jnp.stack(cols, axis=1)


def connection_partials(state: FieldState, conn: GaugeConnectionSpec) -> Array:
    r"""Closed-form ``d_rho A_nu^a`` of shape ``(B, rho, nu, a)``."""
    d = conn.spacetime_dim
    rows = []
    for rho in range(d):
        cols = []
        for nu in range(d):
            names = conn.form.comps[(nu,)]
            cols.append(
                jnp.stack(
                    [derivative(state, nm, axis=rho, order=1) for nm in names], axis=-1
                )
            )
        rows.append(jnp.stack(cols, axis=1))
    return jnp.stack(rows, axis=1)


def connection_second_partials(state: FieldState, conn: GaugeConnectionSpec) -> Array:
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
                    jnp.stack(
                        [mixed_partial(state, nm, (rho, mu), (1, 1)) for nm in names],
                        axis=-1,
                    )
                )
            out_mu.append(jnp.stack(out_nu, axis=1))
        out_rho.append(jnp.stack(out_mu, axis=1))
    return jnp.stack(out_rho, axis=1)


def field_strength(state: FieldState, conn: GaugeConnectionSpec) -> Array:
    r"""Field strength ``F_{mu nu}^a`` of shape ``(B, d, d, n)`` from a field state."""
    a = connection_value(state, conn)
    da = connection_partials(state, conn)
    return field_strength_from_arrays(
        a, da, algebra=conn.algebra, coupling=conn.coupling
    )


def field_strength_from_arrays(
    A: Array, dA: Array, *, algebra: LieAlgebra, coupling: float
) -> Array:
    r"""Field strength from explicit ``A`` and ``dA`` arrays (no field state)."""
    f = structure_constants(algebra, dtype=A.dtype)
    return kernels.field_strength(jnp, A, dA, f, coupling)


def covariant_derivative_field_strength(
    state: FieldState,
    conn: GaugeConnectionSpec,
) -> Array:
    r"""Covariant derivative ``(D_rho F_{mu nu})^a`` of shape ``(B, rho, mu, nu, a)``."""
    a = connection_value(state, conn)
    da = connection_partials(state, conn)
    dda = connection_second_partials(state, conn)
    return covariant_derivative_field_strength_from_arrays(
        a, da, dda, algebra=conn.algebra, coupling=conn.coupling
    )


def covariant_derivative_field_strength_from_arrays(
    A: Array,
    dA: Array,
    ddA: Array,
    *,
    algebra: LieAlgebra,
    coupling: float,
) -> Array:
    r"""``(D_rho F_{mu nu})^a`` from explicit ``A``, ``dA``, ``ddA`` arrays."""
    f = structure_constants(algebra, dtype=A.dtype)
    return kernels.covariant_derivative_field_strength(jnp, A, dA, ddA, f, coupling)


def covariant_divergence(
    state: FieldState,
    conn: GaugeConnectionSpec,
    *,
    signature: tuple[int, ...] | None = None,
) -> Array:
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
    A: Array,
    dA: Array,
    ddA: Array,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
) -> Array:
    r"""Yang-Mills operator from explicit ``A``, ``dA``, ``ddA`` arrays."""
    f = structure_constants(algebra, dtype=A.dtype)
    eta = signature_diagonal(signature, dtype=A.dtype)
    return kernels.covariant_divergence(jnp, A, dA, ddA, f, coupling, eta)


def bianchi(
    state: FieldState,
    conn: GaugeConnectionSpec,
    *,
    signature: tuple[int, ...] | None = None,
) -> Array:
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
    A: Array,
    dA: Array,
    ddA: Array,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
) -> Array:
    r"""Bianchi operator from explicit ``A``, ``dA``, ``ddA`` arrays."""
    f = structure_constants(algebra, dtype=A.dtype)
    eta = signature_diagonal(signature, dtype=A.dtype)
    eps = levi_civita(A.shape[1], dtype=A.dtype)
    return kernels.bianchi(jnp, A, dA, ddA, f, coupling, eta, eps)


def gauge_covariant_derivative(
    phi: Array,
    dphi: Array,
    A: Array,
    *,
    algebra: LieAlgebra,
    coupling: float,
) -> Array:
    r"""``(D_mu phi)^a = d_mu phi^a + g f^{abc} A_mu^b phi^c`` for an adjoint scalar."""
    f = structure_constants(algebra, dtype=A.dtype)
    return kernels.covariant_derivative_adjoint(jnp, phi, dphi, A, f, coupling)


def gauge_flow_rhs(
    state: FieldState,
    conn: GaugeConnectionSpec,
    *,
    signature: tuple[int, ...] | None = None,
    deturck: float = 0.0,
) -> Array:
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
    A: Array,
    dA: Array,
    ddA: Array,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
    deturck: float = 0.0,
) -> Array:
    r"""Yang-Mills gradient-flow drift from explicit ``A``, ``dA``, ``ddA`` arrays."""
    f = structure_constants(algebra, dtype=A.dtype)
    eta = signature_diagonal(signature, dtype=A.dtype)
    return kernels.yang_mills_gradient_flow_rhs(
        jnp, A, dA, ddA, f, coupling, eta, deturck=deturck
    )


def langevin_step(
    A: Array,
    dA: Array,
    ddA: Array,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
    dt: float,
    deturck: float = 0.0,
    temperature: float = 1.0,
    noise: Array | None = None,
    key: Array | None = None,
) -> Array:
    r"""One Euler-Maruyama Langevin step ``A + dt * drift + sqrt(2 dt T) * xi``.

    The jax twin of :func:`omnibias.geometry.gauge.torch.ops.langevin_step`; with the same
    explicit ``noise`` array it returns a bit-identical update. Supply a
    ``jax.random`` ``key`` when ``noise`` is not given.
    """
    rhs = gauge_flow_rhs_from_arrays(
        A, dA, ddA, algebra=algebra, coupling=coupling, signature=signature, deturck=deturck
    )
    if noise is None:
        if key is None:
            msg = "langevin_step requires either an explicit `noise` array or a `key`"
            raise ValueError(msg)
        noise = jax.random.normal(key, A.shape, dtype=A.dtype)
    return A + dt * rhs + math.sqrt(2.0 * dt * temperature) * noise


__all__ = [
    "bianchi",
    "bianchi_from_arrays",
    "connection_partials",
    "connection_second_partials",
    "connection_value",
    "covariant_derivative_field_strength",
    "covariant_derivative_field_strength_from_arrays",
    "covariant_divergence",
    "covariant_divergence_from_arrays",
    "field_strength",
    "field_strength_from_arrays",
    "gauge_covariant_derivative",
    "gauge_flow_rhs",
    "gauge_flow_rhs_from_arrays",
    "langevin_step",
]

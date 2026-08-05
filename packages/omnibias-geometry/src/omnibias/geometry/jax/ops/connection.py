# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Metric, connection, curvature and geodesics (jax).

Bit-identical twin of :mod:`omnibias.geometry.torch.ops.connection`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.geometry._core.manifold import ManifoldSpec


def _gfn(manifold: ManifoldSpec) -> object:
    return manifold.metric.g_point


def metric(coords: Array, manifold: ManifoldSpec) -> Array:
    """Metric ``g_ij`` of shape ``(B, d, d)`` evaluated at ``coords``."""
    return jax.vmap(manifold.metric.g_point)(coords)


def inverse_metric(coords: Array, manifold: ManifoldSpec) -> Array:
    """Inverse metric ``g^{ij}`` of shape ``(B, d, d)``."""
    return jnp.linalg.inv(metric(coords, manifold))


def sqrt_det_metric(coords: Array, manifold: ManifoldSpec) -> Array:
    r""":math:`\sqrt{|\det g|}` of shape ``(B,)``."""
    return jnp.sqrt(jnp.abs(jnp.linalg.det(metric(coords, manifold))))


def _christoffel_point(x: Array, gfn: object) -> Array:
    """``Gamma^k_{ij}`` of shape ``(d, d, d)`` at a single point ``x``."""
    g = gfn(x)  # type: ignore[operator]
    ginv = jnp.linalg.inv(g)
    dg = jax.jacfwd(gfn)(x)  # (d, d, d): dg[i, j, k] = d g_ij / d x_k
    d_i_g_lj = jnp.einsum("acd->adc", dg)
    d_j_g_li = dg
    d_l_g_ij = jnp.einsum("acd->dac", dg)
    bracket = d_i_g_lj + d_j_g_li - d_l_g_ij
    return 0.5 * jnp.einsum("kl,lij->kij", ginv, bracket)


def christoffel(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""Christoffel symbols :math:`\Gamma^k_{ij}` of shape ``(B, d, d, d)``."""
    gfn = _gfn(manifold)
    return jax.vmap(lambda x: _christoffel_point(x, gfn))(coords)


def _riemann_point(x: Array, gfn: object) -> Array:
    """``R^rho_{sigma mu nu}`` of shape ``(d, d, d, d)`` at a point."""
    gamma = _christoffel_point(x, gfn)
    dgamma = jax.jacfwd(lambda y: _christoffel_point(y, gfn))(x)  # [k, i, j, m]
    term1 = jnp.einsum("rnsm->rsmn", dgamma)
    term2 = jnp.einsum("rmsn->rsmn", dgamma)
    term3 = jnp.einsum("rml,lns->rsmn", gamma, gamma)
    term4 = jnp.einsum("rnl,lms->rsmn", gamma, gamma)
    return term1 - term2 + term3 - term4


def riemann_tensor(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""Riemann tensor :math:`R^\rho{}_{\sigma\mu\nu}`, shape ``(B, d, d, d, d)``."""
    gfn = _gfn(manifold)
    return jax.vmap(lambda x: _riemann_point(x, gfn))(coords)


def ricci_tensor(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""Ricci tensor :math:`R_{\sigma\nu} = R^\rho{}_{\sigma\rho\nu}`, ``(B, d, d)``."""
    riem = riemann_tensor(coords, manifold)
    return jnp.einsum("brsrn->bsn", riem)


def scalar_curvature(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""Scalar curvature :math:`R = g^{\sigma\nu} R_{\sigma\nu}`, shape ``(B,)``."""
    ginv = inverse_metric(coords, manifold)
    ric = ricci_tensor(coords, manifold)
    return jnp.einsum("bsn,bsn->b", ginv, ric)


def _metric_jacobian(coords: Array, manifold: ManifoldSpec) -> Array:
    """``dg[b, k, l, i] = d g_kl / d x_i`` of shape ``(B, d, d, d)``."""
    gfn = _gfn(manifold)
    return jax.vmap(jax.jacfwd(gfn))(coords)


def metric_density_divergence(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""``A^j = (1/sqrt|g|) d_i ( sqrt|g| g^{ij} )`` of shape ``(B, d)``.

    Computed from the metric derivative (no determinant under autodiff); see the
    torch twin for the identity.
    """
    ginv = inverse_metric(coords, manifold)
    dg = _metric_jacobian(coords, manifold)  # [b, k, l, i] = d_i g_kl
    term_dginv = -jnp.einsum("bik,bjl,bkli->bj", ginv, ginv, dg)
    term_logsqrt = 0.5 * jnp.einsum("bij,bac,baci->bj", ginv, ginv, dg)
    return term_dginv + term_logsqrt


def geodesic_rhs(coords: Array, velocity: Array, manifold: ManifoldSpec) -> Array:
    r"""Geodesic acceleration :math:`\ddot x^k = -\Gamma^k_{ij}\dot x^i \dot x^j`."""
    gamma = christoffel(coords, manifold)
    return -jnp.einsum("bkij,bi,bj->bk", gamma, velocity, velocity)


def lowered_riemann(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""Fully-lowered Riemann tensor :math:`R_{\rho\sigma\mu\nu}=g_{\rho a}R^a{}_{\sigma\mu\nu}`."""
    g = metric(coords, manifold)
    riem = riemann_tensor(coords, manifold)
    return jnp.einsum("bra,basmn->brsmn", g, riem)


def einstein_tensor(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""Einstein tensor :math:`G_{\mu\nu}=R_{\mu\nu}-\tfrac12 R\,g_{\mu\nu}`, ``(B, d, d)``."""
    ric = ricci_tensor(coords, manifold)
    g = metric(coords, manifold)
    sc = scalar_curvature(coords, manifold)
    return ric - 0.5 * sc[:, None, None] * g


def einstein_equation_residual(
    coords: Array,
    manifold: ManifoldSpec,
    stress_energy: Array | None = None,
    *,
    cosmological_constant: float = 0.0,
    kappa: float = 1.0,
) -> Array:
    r"""Einstein field-equation residual :math:`G_{\mu\nu}+\Lambda g_{\mu\nu}-\kappa T_{\mu\nu}`."""
    g = metric(coords, manifold)
    res = einstein_tensor(coords, manifold) + cosmological_constant * g
    if stress_energy is not None:
        res = res - kappa * stress_energy
    return res


def kretschmann_scalar(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""Kretschmann scalar :math:`K=R_{\rho\sigma\mu\nu}R^{\rho\sigma\mu\nu}`, ``(B,)``."""
    ginv = inverse_metric(coords, manifold)
    rd = lowered_riemann(coords, manifold)
    ru = jnp.einsum("bpa,bqc,bre,bsf,bacef->bpqrs", ginv, ginv, ginv, ginv, rd)
    return jnp.einsum("bpqrs,bpqrs->b", rd, ru)


def weyl_tensor(coords: Array, manifold: ManifoldSpec) -> Array:
    r"""Weyl conformal tensor :math:`C_{\rho\sigma\mu\nu}`, ``(B, d, d, d, d)`` (zeros for ``d < 3``)."""
    rd = lowered_riemann(coords, manifold)
    d = manifold.dim
    if d < 3:
        return jnp.zeros_like(rd)
    g = metric(coords, manifold)
    ric = ricci_tensor(coords, manifold)
    sc = scalar_curvature(coords, manifold)
    gr = (
        jnp.einsum("brm,bsn->brsmn", g, ric)
        - jnp.einsum("brn,bsm->brsmn", g, ric)
        - jnp.einsum("bsm,brn->brsmn", g, ric)
        + jnp.einsum("bsn,brm->brsmn", g, ric)
    )
    gg = jnp.einsum("brm,bsn->brsmn", g, g) - jnp.einsum("brn,bsm->brsmn", g, g)
    scale = sc[:, None, None, None, None] / ((d - 1) * (d - 2))
    return rd - (1.0 / (d - 2)) * gr + scale * gg


def geodesic_deviation(
    coords: Array, velocity: Array, separation: Array, manifold: ManifoldSpec,
) -> Array:
    r"""Tidal (Jacobi) acceleration :math:`a^\rho=-R^\rho{}_{\sigma\mu\nu}u^\sigma\xi^\mu u^\nu`."""
    riem = riemann_tensor(coords, manifold)
    return -jnp.einsum("brsmn,bs,bm,bn->br", riem, velocity, separation, velocity)


__all__ = [
    "christoffel",
    "einstein_equation_residual",
    "einstein_tensor",
    "geodesic_deviation",
    "geodesic_rhs",
    "inverse_metric",
    "kretschmann_scalar",
    "lowered_riemann",
    "metric",
    "metric_density_divergence",
    "ricci_tensor",
    "riemann_tensor",
    "scalar_curvature",
    "sqrt_det_metric",
]

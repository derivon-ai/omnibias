# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Metric, connection, curvature and geodesics (torch).

All quantities are functions of the coordinate point only (they depend on the
metric, not on a field). Metric derivatives are obtained by forward-mode autodiff
of the analytic per-point metric ``g_point`` -- exact for analytic metrics. See
``GEOMETRY_DERIVATIONS.md`` for the index conventions and identities.

Index convention: ``christoffel`` returns ``Gamma[..., k, i, j] = Gamma^k_{ij}``;
``riemann_tensor`` returns ``R[..., rho, sigma, mu, nu] = R^rho_{sigma mu nu}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.func import jacfwd, vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.geometry._core.manifold import ManifoldSpec


def _gfn(manifold: ManifoldSpec) -> object:
    return manifold.metric.g_point


def metric(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    """Metric ``g_ij`` of shape ``(B, d, d)`` evaluated at ``coords``."""
    return vmap(manifold.metric.g_point)(coords)


def inverse_metric(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    """Inverse metric ``g^{ij}`` of shape ``(B, d, d)``."""
    return torch.linalg.inv(metric(coords, manifold))


def sqrt_det_metric(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r""":math:`\sqrt{|\det g|}` of shape ``(B,)``."""
    return torch.sqrt(torch.abs(torch.linalg.det(metric(coords, manifold))))


def _christoffel_point(x: Tensor, gfn: object) -> Tensor:
    """``Gamma^k_{ij}`` of shape ``(d, d, d)`` at a single point ``x``."""
    g = gfn(x)  # type: ignore[operator]
    ginv = torch.linalg.inv(g)
    dg = jacfwd(gfn)(x)  # (d, d, d): dg[i, j, k] = d g_ij / d x_k
    d_i_g_lj = torch.einsum("acd->adc", dg)  # [l, i, j]
    d_j_g_li = dg                            # [l, i, j]
    d_l_g_ij = torch.einsum("acd->dac", dg)  # [l, i, j]
    bracket = d_i_g_lj + d_j_g_li - d_l_g_ij
    return 0.5 * torch.einsum("kl,lij->kij", ginv, bracket)


def christoffel(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Christoffel symbols :math:`\Gamma^k_{ij}` of shape ``(B, d, d, d)``."""
    gfn = _gfn(manifold)
    return vmap(lambda x: _christoffel_point(x, gfn))(coords)


def _riemann_point(x: Tensor, gfn: object) -> Tensor:
    """``R^rho_{sigma mu nu}`` of shape ``(d, d, d, d)`` at a point."""
    gamma = _christoffel_point(x, gfn)  # [k, i, j]
    dgamma = jacfwd(lambda y: _christoffel_point(y, gfn))(x)  # [k, i, j, m]
    # R^rho_{sigma mu nu} = d_mu G^rho_{nu sigma} - d_nu G^rho_{mu sigma}
    #                       + G^rho_{mu lam} G^lam_{nu sigma}
    #                       - G^rho_{nu lam} G^lam_{mu sigma}
    term1 = torch.einsum("rnsm->rsmn", dgamma)
    term2 = torch.einsum("rmsn->rsmn", dgamma)
    term3 = torch.einsum("rml,lns->rsmn", gamma, gamma)
    term4 = torch.einsum("rnl,lms->rsmn", gamma, gamma)
    return term1 - term2 + term3 - term4


def riemann_tensor(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Riemann tensor :math:`R^\rho{}_{\sigma\mu\nu}`, shape ``(B, d, d, d, d)``."""
    gfn = _gfn(manifold)
    return vmap(lambda x: _riemann_point(x, gfn))(coords)


def ricci_tensor(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Ricci tensor :math:`R_{\sigma\nu} = R^\rho{}_{\sigma\rho\nu}`, ``(B, d, d)``."""
    riem = riemann_tensor(coords, manifold)
    return torch.einsum("brsrn->bsn", riem)


def scalar_curvature(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Scalar curvature :math:`R = g^{\sigma\nu} R_{\sigma\nu}`, shape ``(B,)``."""
    ginv = inverse_metric(coords, manifold)
    ric = ricci_tensor(coords, manifold)
    return torch.einsum("bsn,bsn->b", ginv, ric)


def _metric_jacobian(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    """``dg[b, k, l, i] = d g_kl / d x_i`` of shape ``(B, d, d, d)``."""
    gfn = _gfn(manifold)
    return vmap(jacfwd(gfn))(coords)


def metric_density_divergence(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""``A^j = (1/sqrt|g|) d_i ( sqrt|g| g^{ij} )`` of shape ``(B, d)``.

    Computed from the metric derivative via
    :math:`A^j = \partial_i g^{ij} + g^{ij}\,\tfrac12 g^{ab}\partial_i g_{ab}`
    (equivalently the contracted Christoffel identity
    :math:`A^j = -g^{ik}\Gamma^j_{ik}`). This uses only the first metric
    derivative -- no determinant under autodiff -- so it is a code path
    independent of :func:`christoffel`, doubling as a connection cross-check.
    """
    ginv = inverse_metric(coords, manifold)
    dg = _metric_jacobian(coords, manifold)  # [b, k, l, i] = d_i g_kl
    term_dginv = -torch.einsum("bik,bjl,bkli->bj", ginv, ginv, dg)
    term_logsqrt = 0.5 * torch.einsum("bij,bac,baci->bj", ginv, ginv, dg)
    return term_dginv + term_logsqrt


def geodesic_rhs(coords: Tensor, velocity: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Geodesic acceleration :math:`\ddot x^k = -\Gamma^k_{ij}\dot x^i \dot x^j`.

    Parameters
    ----------
    coords
        Positions, shape ``(B, d)``.
    velocity
        Velocities :math:`\dot x`, shape ``(B, d)``.
    manifold
        The manifold supplying the metric.

    Returns
    -------
    Tensor
        The acceleration ``(B, d)``.
    """
    gamma = christoffel(coords, manifold)  # (B, k, i, j)
    return -torch.einsum("bkij,bi,bj->bk", gamma, velocity, velocity)


def lowered_riemann(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Fully-lowered Riemann tensor :math:`R_{\rho\sigma\mu\nu}=g_{\rho a}R^a{}_{\sigma\mu\nu}`.

    Shape ``(B, d, d, d, d)`` with all four indices down.
    """
    g = metric(coords, manifold)
    riem = riemann_tensor(coords, manifold)  # R^a_{sigma mu nu}
    return torch.einsum("bra,basmn->brsmn", g, riem)


def einstein_tensor(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Einstein tensor :math:`G_{\mu\nu}=R_{\mu\nu}-\tfrac12 R\,g_{\mu\nu}`, ``(B, d, d)``.

    The trace-reversed Ricci tensor; in ``d=2`` it is identically zero, and its
    contraction ``g^{mu nu} G_{mu nu} = (2-d)/2 * R`` (an independent tie back to
    :func:`scalar_curvature`). Metric derivatives are autodiff-exact.
    """
    ric = ricci_tensor(coords, manifold)
    g = metric(coords, manifold)
    sc = scalar_curvature(coords, manifold)
    return ric - 0.5 * sc[:, None, None] * g


def einstein_equation_residual(
    coords: Tensor,
    manifold: ManifoldSpec,
    stress_energy: Tensor | None = None,
    *,
    cosmological_constant: float = 0.0,
    kappa: float = 1.0,
) -> Tensor:
    r"""Einstein field-equation residual :math:`G_{\mu\nu}+\Lambda g_{\mu\nu}-\kappa T_{\mu\nu}`.

    ``stress_energy`` is the supplied stress-energy tensor ``T_{mu nu}`` of shape
    ``(B, d, d)`` (``None`` means vacuum, ``T = 0``). Returns shape ``(B, d, d)``;
    a vacuum solution with the right ``Lambda`` encloses zero.
    """
    g = metric(coords, manifold)
    res = einstein_tensor(coords, manifold) + cosmological_constant * g
    if stress_energy is not None:
        res = res - kappa * stress_energy
    return res


def kretschmann_scalar(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Kretschmann scalar :math:`K=R_{\rho\sigma\mu\nu}R^{\rho\sigma\mu\nu}`, ``(B,)``.

    A coordinate invariant that stays finite where components blow up (e.g.
    Schwarzschild ``K = 48 M^2 / r^6``), so it certifies genuine curvature
    rather than a coordinate artefact.
    """
    ginv = inverse_metric(coords, manifold)
    rd = lowered_riemann(coords, manifold)  # R_{rho sigma mu nu}
    ru = torch.einsum("bpa,bqc,bre,bsf,bacef->bpqrs", ginv, ginv, ginv, ginv, rd)
    return torch.einsum("bpqrs,bpqrs->b", rd, ru)


def weyl_tensor(coords: Tensor, manifold: ManifoldSpec) -> Tensor:
    r"""Weyl conformal tensor :math:`C_{\rho\sigma\mu\nu}`, ``(B, d, d, d, d)``.

    The totally trace-free part of the Riemann tensor,

    .. math::

        C_{abcd} = R_{abcd}
            - \tfrac{1}{d-2}\big(g_{ac}R_{bd}-g_{ad}R_{bc}-g_{bc}R_{ad}+g_{bd}R_{ac}\big)
            + \tfrac{R}{(d-1)(d-2)}\big(g_{ac}g_{bd}-g_{ad}g_{bc}\big).

    It is a conformal invariant; for ``d < 3`` (and for conformally-flat or
    maximally-symmetric spaces) it is identically zero, returned as zeros.
    """
    rd = lowered_riemann(coords, manifold)
    d = manifold.dim
    if d < 3:
        return torch.zeros_like(rd)
    g = metric(coords, manifold)
    ric = ricci_tensor(coords, manifold)
    sc = scalar_curvature(coords, manifold)
    gr = (
        torch.einsum("brm,bsn->brsmn", g, ric)
        - torch.einsum("brn,bsm->brsmn", g, ric)
        - torch.einsum("bsm,brn->brsmn", g, ric)
        + torch.einsum("bsn,brm->brsmn", g, ric)
    )
    gg = torch.einsum("brm,bsn->brsmn", g, g) - torch.einsum("brn,bsm->brsmn", g, g)
    scale = sc[:, None, None, None, None] / ((d - 1) * (d - 2))
    return rd - (1.0 / (d - 2)) * gr + scale * gg


def geodesic_deviation(
    coords: Tensor, velocity: Tensor, separation: Tensor, manifold: ManifoldSpec,
) -> Tensor:
    r"""Tidal (Jacobi) acceleration :math:`a^\rho=-R^\rho{}_{\sigma\mu\nu}u^\sigma\xi^\mu u^\nu`.

    The relative acceleration of neighbouring geodesics with tangent ``velocity``
    ``u`` and separation ``separation`` ``xi``; all shape ``(B, d)``. Vanishes in
    flat space and probes the full Riemann tensor.
    """
    riem = riemann_tensor(coords, manifold)  # R^rho_{sigma mu nu}
    return -torch.einsum("brsmn,bs,bm,bn->br", riem, velocity, separation, velocity)


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
    "weyl_tensor",
]

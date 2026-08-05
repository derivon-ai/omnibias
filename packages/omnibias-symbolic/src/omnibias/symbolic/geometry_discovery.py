# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differential-geometry operator columns for multivariate neural-jet discovery.

:mod:`omnibias.symbolic.field_discovery` lifts the closed-form neural jet to the
*flat* vector-calculus surface (gradient, divergence, curl, Laplacian, Ito
generator, constant-metric anisotropic Laplacian). This module lifts it one
further, to **Riemannian geometry on a curved manifold**: the Laplace--Beltrami
operator, Christoffel symbols, covariant Hessian, and the Riemann / Ricci /
scalar curvature -- plus the *pullback metric of a learned chart* ``g = J^T h J``
that is the headline of :mod:`omnibias.geometry`.

Honesty contract (mirrors :mod:`omnibias.geometry`):

* **Field-function derivatives are exact closed form** -- every ``partial^alpha u``
  comes from the activation tower via :func:`~omnibias.symbolic.field_discovery.extract_field_jet`.
* **The metric and its derivatives are inputs.** A :class:`MetricField` carries
  ``g_ij`` and ``partial_k g_ij`` (and optionally ``partial_l partial_k g_ij`` for
  curvature) at the sample points. For an *analytic* metric these are exact; for a
  *learned chart* ``phi`` they are the exact closed-form pullback
  ``g = J^T J`` assembled from the chart components' own neural jets
  (:func:`pullback_metric_field`). Either way the geometric operators below are
  exact -- no finite differences.

The discovery face: with the Laplace--Beltrami column injected as an exact
operator atom (``FieldLawDiscoverer(..., extra_columns_fn=...)``), the sparse
search recovers genuinely geometric laws such as the **heat flow on the
hyperbolic plane** ``u_t = Delta_g u`` (where ``Delta_g`` carries the Christoffel
drift), not merely its flat coordinate expansion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from omnibias.symbolic.field_discovery import (
    FieldJet,
    FieldLawDiscoverer,
    analytic_field_jet,
    field_gradient,
    field_hessian,
)

MultiIndex = tuple[int, ...]


# --------------------------------------------------------------------------- #
# Metric field: g_ij, d_k g_ij, (optional) d_l d_k g_ij at the sample points
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricField:
    r"""A Riemannian metric and its derivatives sampled on a point cloud.

    ``g[n]`` is the symmetric positive-definite matrix ``g_ij`` at point ``X[n]``;
    ``dg[n, k, i, j] = partial_k g_ij``; and the optional
    ``ddg[n, l, k, i, j] = partial_l partial_k g_ij`` (required only for the
    curvature operators). All exact -- analytic for a closed-form metric, or the
    closed-form pullback of a learned chart (see :func:`pullback_metric_field`).
    """

    X: np.ndarray  # (n, m)
    g: np.ndarray  # (n, m, m)
    dg: np.ndarray  # (n, m, m, m): dg[:, k, i, j] = d_k g_ij
    var_names: tuple[str, ...]
    ddg: np.ndarray | None = None  # (n, m, m, m, m): ddg[:, l, k, i, j] = d_l d_k g_ij

    @property
    def dim(self) -> int:
        return int(self.g.shape[-1])

    @property
    def n(self) -> int:
        return int(self.g.shape[0])

    def __post_init__(self) -> None:
        n, m = self.n, self.dim
        if self.g.shape != (n, m, m):
            raise ValueError(f"g must be (n, m, m), got {self.g.shape}")
        if self.dg.shape != (n, m, m, m):
            raise ValueError(f"dg must be (n, m, m, m), got {self.dg.shape}")
        if self.X.shape != (n, m):
            raise ValueError(f"X must be ({n}, {m}), got {self.X.shape}")
        if len(self.var_names) != m:
            raise ValueError("var_names must match the metric dimension")
        if self.ddg is not None and self.ddg.shape != (n, m, m, m, m):
            raise ValueError(f"ddg must be (n, m, m, m, m), got {self.ddg.shape}")


def analytic_metric_field(
    X: np.ndarray,
    g: np.ndarray,
    dg: np.ndarray,
    *,
    ddg: np.ndarray | None = None,
    var_names: tuple[str, ...] | None = None,
) -> MetricField:
    """Assemble a :class:`MetricField` from explicit (exact) metric arrays."""
    X = np.asarray(X, dtype=float)
    g = np.asarray(g, dtype=float)
    dg = np.asarray(dg, dtype=float)
    ddg_arr = None if ddg is None else np.asarray(ddg, dtype=float)
    m = g.shape[-1]
    names = var_names if var_names is not None else tuple(f"x{i}" for i in range(m))
    return MetricField(X=X, g=g, dg=dg, var_names=names, ddg=ddg_arr)


def flat_metric_field(
    X: np.ndarray, *, var_names: tuple[str, ...] | None = None, with_second: bool = True
) -> MetricField:
    """The Euclidean metric ``g = I`` (zero derivatives, zero curvature)."""
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    g = np.broadcast_to(np.eye(m), (n, m, m)).copy()
    dg = np.zeros((n, m, m, m))
    ddg = np.zeros((n, m, m, m, m)) if with_second else None
    return analytic_metric_field(X, g, dg, ddg=ddg, var_names=var_names)


def warped_product_metric_field(
    X: np.ndarray,
    f: np.ndarray,
    fp: np.ndarray,
    fpp: np.ndarray | None = None,
    *,
    var_names: tuple[str, ...] = ("x", "y"),
) -> MetricField:
    r"""The 2-D warped-product metric ``ds^2 = dx^2 + f(x)^2 dy^2``.

    A surface of revolution / geodesic-polar form whose Gaussian curvature is
    ``K = -f''/f`` (so scalar curvature ``R = 2K``). Recovers the sphere
    (``f = sin x``, ``K = +1``), the hyperbolic plane (``f = e^x``, ``K = -1``),
    and the flat plane (``f = 1``). ``f, fp(=f'), fpp(=f'')`` are the warp and its
    derivatives sampled at ``X[:, 0]``; ``fpp`` is required only for curvature.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise ValueError(f"warped-product metric is 2-D, got {X.shape[1]} columns")
    n = X.shape[0]
    f = np.asarray(f, dtype=float).reshape(-1)
    fp = np.asarray(fp, dtype=float).reshape(-1)
    g = np.zeros((n, 2, 2))
    g[:, 0, 0] = 1.0
    g[:, 1, 1] = f**2
    dg = np.zeros((n, 2, 2, 2))
    dg[:, 0, 1, 1] = 2.0 * f * fp  # d_x g_yy
    ddg = None
    if fpp is not None:
        fpp = np.asarray(fpp, dtype=float).reshape(-1)
        ddg = np.zeros((n, 2, 2, 2, 2))
        ddg[:, 0, 0, 1, 1] = 2.0 * (fp**2 + f * fpp)  # d_x d_x g_yy
    return analytic_metric_field(X, g, dg, ddg=ddg, var_names=var_names)


# --------------------------------------------------------------------------- #
# Metric algebra
# --------------------------------------------------------------------------- #
def metric_inverse(metric: MetricField) -> np.ndarray:
    r"""Inverse metric ``g^{ij}``, shape ``(n, m, m)``."""
    return np.asarray(np.linalg.inv(metric.g), dtype=float)


def metric_determinant(metric: MetricField) -> np.ndarray:
    r"""Metric determinant ``det g``, shape ``(n,)``."""
    return np.asarray(np.linalg.det(metric.g), dtype=float)


def christoffel_symbols(metric: MetricField) -> np.ndarray:
    r"""Christoffel symbols of the second kind ``Gamma^k_{ij}``, shape ``(n, m, m, m)``.

    ``Gamma^k_{ij} = 1/2 g^{kl}(partial_i g_{lj} + partial_j g_{li} - partial_l g_{ij})``,
    indexed ``Gamma[:, k, i, j]``. Exact given the metric jet.
    """
    ginv = metric_inverse(metric)
    dg = metric.dg
    # term[:, l, i, j] = d_i g_lj + d_j g_li - d_l g_ij
    term = (
        np.einsum("nilj->nlij", dg)
        + np.einsum("njli->nlij", dg)
        - dg
    )
    gamma: np.ndarray = 0.5 * np.einsum("nkl,nlij->nkij", ginv, term)
    return np.asarray(gamma, dtype=float)


def _christoffel_gradient(metric: MetricField, ginv: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    r"""``partial_m Gamma^k_{ij}``, shape ``(n, m, m, m, m)`` (needs second metric derivs)."""
    if metric.ddg is None:
        raise ValueError("curvature requires second metric derivatives (ddg)")
    dg = metric.dg
    ddg = metric.ddg
    # d_m g^{kl} = - g^{ka} (d_m g_{ab}) g^{bl}
    dginv = -np.einsum("nka,nmab,nbl->nmkl", ginv, dg, ginv)
    # T[:, l, i, j] = d_i g_lj + d_j g_li - d_l g_ij  (same kernel as christoffel)
    term = np.einsum("nilj->nlij", dg) + np.einsum("njli->nlij", dg) - dg
    # d_m T[:, l, i, j] = d_m d_i g_lj + d_m d_j g_li - d_m d_l g_ij
    dterm = (
        np.einsum("nmilj->nmlij", ddg)
        + np.einsum("nmjli->nmlij", ddg)
        - np.einsum("nmlij->nmlij", ddg)
    )
    dgamma = 0.5 * (
        np.einsum("nmkl,nlij->nmkij", dginv, term)
        + np.einsum("nkl,nmlij->nmkij", ginv, dterm)
    )
    return np.asarray(dgamma, dtype=float)


def riemann_tensor(metric: MetricField) -> np.ndarray:
    r"""Riemann curvature ``R^{rho}_{sigma mu nu}``, shape ``(n, m, m, m, m)``.

    ``R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
    + Gamma^rho_{mu lambda} Gamma^lambda_{nu sigma}
    - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}``, indexed
    ``R[:, rho, sigma, mu, nu]``. Requires :attr:`MetricField.ddg`.
    """
    ginv = metric_inverse(metric)
    gamma = christoffel_symbols(metric)
    dgamma = _christoffel_gradient(metric, ginv, gamma)
    # d_mu Gamma^rho_{nu sigma} -> [rho, sigma, mu, nu]
    r1 = np.einsum("nmrvs->nrsmv", dgamma)
    r2 = np.einsum("nvrms->nrsmv", dgamma)
    # Gamma^rho_{mu l} Gamma^l_{nu sigma} -> [rho, sigma, mu, nu]
    t3 = np.einsum("nrml,nlvs->nrsmv", gamma, gamma)
    t4 = np.einsum("nrvl,nlms->nrsmv", gamma, gamma)
    return np.asarray(r1 - r2 + t3 - t4, dtype=float)


def ricci_tensor(metric: MetricField) -> np.ndarray:
    r"""Ricci tensor ``R_{sigma nu} = R^{rho}_{sigma rho nu}``, shape ``(n, m, m)``."""
    riem = riemann_tensor(metric)
    return np.asarray(np.einsum("nrsrv->nsv", riem), dtype=float)


def scalar_curvature(metric: MetricField) -> np.ndarray:
    r"""Scalar curvature ``R = g^{sigma nu} R_{sigma nu}``, shape ``(n,)``.

    For a surface ``R = 2K`` (twice the Gaussian curvature): ``+2`` on the unit
    sphere, ``0`` on the plane, ``-2`` on the hyperbolic plane.
    """
    ginv = metric_inverse(metric)
    ric = ricci_tensor(metric)
    return np.asarray(np.einsum("nsv,nsv->n", ginv, ric), dtype=float)


# --------------------------------------------------------------------------- #
# Metric-aware field operators (exact closed-form field derivatives)
# --------------------------------------------------------------------------- #
def _resolve_spatial(jet: FieldJet, metric: MetricField, spatial_axes: Sequence[int] | None) -> tuple[int, ...]:
    if spatial_axes is None:
        ax = tuple(range(metric.dim))
    else:
        ax = tuple(int(a) for a in spatial_axes)
    if len(ax) != metric.dim:
        raise ValueError(f"spatial_axes length {len(ax)} != metric dim {metric.dim}")
    for a in ax:
        if a < 0 or a >= jet.dim:
            raise ValueError(f"axis {a} out of range for jet dim {jet.dim}")
    if metric.n != jet.n:
        raise ValueError(f"metric has {metric.n} points but jet has {jet.n}")
    return ax


def covariant_hessian(
    jet: FieldJet, metric: MetricField, *, spatial_axes: Sequence[int] | None = None
) -> np.ndarray:
    r"""Covariant Hessian ``nabla^2_{ij} u = partial^2_{ij} u - Gamma^k_{ij} partial_k u``.

    Shape ``(n, m, m)``. The geometric correction ``-Gamma^k_{ij} partial_k u`` is
    what makes the Hessian a genuine ``(0, 2)``-tensor on the manifold. Field
    partials are exact; the Christoffel symbols come from the metric jet.
    """
    ax = _resolve_spatial(jet, metric, spatial_axes)
    grad = field_gradient(jet, axes=ax)  # (n, m)
    hess = field_hessian(jet, axes=ax)  # (n, m, m)
    gamma = christoffel_symbols(metric)  # (n, k, i, j)
    correction = np.einsum("nkij,nk->nij", gamma, grad)
    return np.asarray(hess - correction, dtype=float)


def laplace_beltrami(
    jet: FieldJet, metric: MetricField, *, spatial_axes: Sequence[int] | None = None
) -> np.ndarray:
    r"""Laplace--Beltrami operator ``Delta_g u = g^{ij} nabla^2_{ij} u``, shape ``(n,)``.

    Equivalently ``(1/sqrt|g|) partial_i(sqrt|g| g^{ij} partial_j u)``. The full
    position-dependent Riemannian Laplacian -- the Christoffel drift term
    distinguishes it from the flat / constant-metric
    :func:`~omnibias.symbolic.field_discovery.field_anisotropic_laplacian`. Field
    derivatives are exact closed form; the metric is the (exact) input.
    """
    ginv = metric_inverse(metric)
    cov_hess = covariant_hessian(jet, metric, spatial_axes=spatial_axes)
    return np.asarray(np.einsum("nij,nij->n", ginv, cov_hess), dtype=float)


def metric_grad_norm_sq(
    jet: FieldJet, metric: MetricField, *, spatial_axes: Sequence[int] | None = None
) -> np.ndarray:
    r"""Riemannian squared gradient norm ``|nabla u|_g^2 = g^{ij} partial_i u partial_j u``.

    Shape ``(n,)`` -- the geometric eikonal term (``= 1`` for a signed geodesic
    distance). Reduces to :func:`~omnibias.symbolic.field_discovery.field_grad_norm_sq`
    when ``g = I``.
    """
    ax = _resolve_spatial(jet, metric, spatial_axes)
    ginv = metric_inverse(metric)
    grad = field_gradient(jet, axes=ax)
    return np.asarray(np.einsum("nij,ni,nj->n", ginv, grad, grad), dtype=float)


# --------------------------------------------------------------------------- #
# Pullback metric of a learned chart: g = J^T J (and its exact derivatives)
# --------------------------------------------------------------------------- #
def _component_jacobian(chart_jets: Sequence[FieldJet], m: int) -> np.ndarray:
    """J[:, a, i] = partial_i phi_a, shape (n, ambient, m)."""
    n = chart_jets[0].n
    amb = len(chart_jets)
    jac = np.empty((n, amb, m), dtype=float)
    for a, comp in enumerate(chart_jets):
        for i in range(m):
            jac[:, a, i] = comp.partial(_unit(comp.dim, i))
    return jac


def _component_hessian(chart_jets: Sequence[FieldJet], m: int) -> np.ndarray:
    """H[:, a, k, i] = partial_k partial_i phi_a, shape (n, ambient, m, m)."""
    n = chart_jets[0].n
    amb = len(chart_jets)
    hess = np.empty((n, amb, m, m), dtype=float)
    for a, comp in enumerate(chart_jets):
        for k in range(m):
            for i in range(m):
                hess[:, a, k, i] = comp.partial(_pair(comp.dim, k, i))
    return hess


def _component_third(chart_jets: Sequence[FieldJet], m: int) -> np.ndarray:
    """T[:, a, l, k, i] = partial_l partial_k partial_i phi_a, shape (n, ambient, m, m, m)."""
    n = chart_jets[0].n
    amb = len(chart_jets)
    third = np.empty((n, amb, m, m, m), dtype=float)
    for a, comp in enumerate(chart_jets):
        for ell in range(m):
            for k in range(m):
                for i in range(m):
                    third[:, a, ell, k, i] = comp.partial(_triple(comp.dim, ell, k, i))
    return third


def _unit(dim: int, i: int) -> MultiIndex:
    return tuple(1 if j == i else 0 for j in range(dim))


def _pair(dim: int, i: int, j: int) -> MultiIndex:
    out = [0] * dim
    out[i] += 1
    out[j] += 1
    return tuple(out)


def _triple(dim: int, i: int, j: int, k: int) -> MultiIndex:
    out = [0] * dim
    out[i] += 1
    out[j] += 1
    out[k] += 1
    return tuple(out)


def pullback_metric_field(
    chart_jets: Sequence[FieldJet],
    *,
    var_names: tuple[str, ...] | None = None,
    with_curvature: bool = False,
) -> MetricField:
    r"""Closed-form pullback metric ``g = J^T J`` of a learned chart ``phi: M -> R^N``.

    Given the chart's component jets ``phi_a`` (each a
    :class:`~omnibias.symbolic.field_discovery.FieldJet` over the ``m`` manifold
    coordinates), the induced (Euclidean ambient ``h = I``) metric and its exact
    derivatives are pure products of the components' own closed-form partials:

    .. math::

        g_{ij} = \sum_a (\partial_i \phi_a)(\partial_j \phi_a), \quad
        \partial_k g_{ij} = \sum_a (\partial_{ki}\phi_a)(\partial_j\phi_a)
                                  + (\partial_i\phi_a)(\partial_{kj}\phi_a),

    and similarly for ``partial_l partial_k g_{ij}`` (needs order-3 chart jets,
    enabled by ``with_curvature=True``). This is the symbolic-engine twin of the
    :mod:`omnibias.geometry` pullback ``g = J^T h J`` -- the metric of a *learned*
    coordinate chart, exact because the chart is an omnibias field.
    """
    if len(chart_jets) == 0:
        raise ValueError("need at least one chart component")
    m = chart_jets[0].dim
    need = 3 if with_curvature else 2
    for comp in chart_jets:
        if comp.dim != m:
            raise ValueError("all chart components must share the manifold dimension")
        if comp.order < need:
            raise ValueError(
                f"chart jets need order >= {need} (got {comp.order}); "
                "use with_curvature only with order-3 jets"
            )
        if comp.n != chart_jets[0].n:
            raise ValueError("all chart components must share sample points")

    jac = _component_jacobian(chart_jets, m)  # (n, a, i)
    hess = _component_hessian(chart_jets, m)  # (n, a, k, i)
    g = np.einsum("nai,naj->nij", jac, jac)
    dg = np.einsum("naki,naj->nkij", hess, jac) + np.einsum("nai,nakj->nkij", jac, hess)

    ddg = None
    if with_curvature:
        third = _component_third(chart_jets, m)  # (n, a, l, k, i)
        ddg = (
            np.einsum("nalki,naj->nlkij", third, jac)
            + np.einsum("naki,nalj->nlkij", hess, hess)
            + np.einsum("nali,nakj->nlkij", hess, hess)
            + np.einsum("nai,nalkj->nlkij", jac, third)
        )
    names = var_names if var_names is not None else chart_jets[0].var_names
    return analytic_metric_field(chart_jets[0].X, g, dg, ddg=ddg, var_names=names)


# --------------------------------------------------------------------------- #
# Geometric heat flow on the round sphere (exact dataset + discovery)
# --------------------------------------------------------------------------- #
def _sphere_metric_at(X: np.ndarray) -> MetricField:
    r"""The S^2 metric ``g = diag(1, sin^2 theta)`` at ``X[:, :2]`` (Gaussian K=+1)."""
    tp = np.asarray(X, dtype=float)[:, :2]
    theta = tp[:, 0]
    return warped_product_metric_field(
        tp, f=np.sin(theta), fp=np.cos(theta), fpp=-np.sin(theta), var_names=("theta", "phi")
    )


def _legendre_mode(theta: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""``(f, f_theta, f_theta_theta)`` of the zonal harmonic ``f = P_l(cos theta)``."""
    from numpy.polynomial.legendre import Legendre

    p = np.cos(theta)
    sp = np.sin(theta)
    poly = Legendre.basis(degree)
    p0 = np.asarray(poly(p), dtype=float)
    p1 = np.asarray(poly.deriv(1)(p), dtype=float)
    p2 = np.asarray(poly.deriv(2)(p), dtype=float)
    f = p0
    f_theta = -sp * p1
    f_theta_theta = -p * p1 + sp**2 * p2
    return f, f_theta, f_theta_theta


def _spherical_heat_jet(
    X: np.ndarray, *, degrees: tuple[int, ...], amps: tuple[float, ...]
) -> FieldJet:
    r"""Exact jet of a zonal heat solution on ``S^2`` over ``(theta, phi, t)``.

    ``u = sum_l a_l e^{-l(l+1) t} P_l(cos theta)`` (phi-independent). Each zonal
    harmonic is an eigenfunction of the spherical Laplace--Beltrami operator
    ``Delta_g f = f_theta_theta + cot(theta) f_theta`` with eigenvalue
    ``-l(l+1)``, so ``u_t = Delta_g u`` holds exactly. The ``cot(theta)`` drift is
    *position dependent*, so this law is irreducible to any constant-coefficient
    flat relation -- the geometric atom is genuinely necessary.
    """
    names = ("theta", "phi", "t")
    theta, t = X[:, 0], X[:, 2]
    keys = [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2),
    ]
    acc = {k: np.zeros_like(theta) for k in keys}
    for degree, a in zip(degrees, amps, strict=True):
        lam = degree * (degree + 1)
        f, f_th, f_thth = _legendre_mode(theta, degree)
        d = a * np.exp(-lam * t)
        acc[(0, 0, 0)] += d * f
        acc[(1, 0, 0)] += d * f_th
        acc[(2, 0, 0)] += d * f_thth
        acc[(0, 0, 1)] += -lam * d * f
        acc[(1, 0, 1)] += -lam * d * f_th
        acc[(0, 0, 2)] += lam**2 * d * f
    return analytic_field_jet(X, acc, order=2, var_names=names)


def make_geometric_heat_split(
    *,
    seed: int = 0,
    counts: tuple[int, int, int] = (500, 320, 320),
    degrees: tuple[int, ...] = (1, 2),
    amps: tuple[float, ...] = (1.0, 0.5),
) -> tuple[FieldJet, FieldJet, FieldJet, tuple[MetricField, MetricField, MetricField], str]:
    r"""Heat flow on the round sphere: train/val/test jets + their ``S^2`` metrics.

    The exact law is ``u_t = Delta_g u`` on ``S^2`` (metric ``diag(1, sin^2
    theta)``). Two zonal eigenmodes (degrees ``l = 1, 2``) break the single-mode
    degeneracy. Because the Laplace--Beltrami drift ``cot(theta) u_theta`` is
    *position dependent*, no constant-coefficient flat relation reproduces ``u_t``;
    the injected ``Delta_g`` atom is the unique one-term law.
    """
    rng = np.random.default_rng(seed)
    box_lo = np.array([0.4, 0.0, 0.0])
    box_hi = np.array([np.pi - 0.4, 2.0 * np.pi, 0.25])
    grids = [rng.uniform(box_lo, box_hi, size=(n, 3)) for n in counts]
    jets = tuple(_spherical_heat_jet(X, degrees=degrees, amps=amps) for X in grids)
    metrics = tuple(_sphere_metric_at(X) for X in grids)
    return (
        jets[0], jets[1], jets[2],
        (metrics[0], metrics[1], metrics[2]),
        "u_t = Delta_g u  (heat flow on the round sphere S^2)",
    )


def discover_geometric_heat_law(
    train: FieldJet,
    val: FieldJet,
    test: FieldJet,
    metrics: tuple[MetricField, MetricField, MetricField],
    *,
    spatial_axes: tuple[int, ...] = (0, 1),
    time_axis: int = 2,
    lhs_index: MultiIndex = (0, 0, 1),
) -> dict[str, object]:
    r"""Recover ``u_t = Delta_g u`` by injecting the exact Laplace--Beltrami atom.

    The discoverer searches the operator library *plus* the geometric column
    ``lap_g(u) = laplace_beltrami(u, g)`` (evaluated per split at its own points),
    selecting the compact geometric law over the longer flat expansion.
    """
    paired = {id(train): metrics[0], id(val): metrics[1], id(test): metrics[2]}

    def extra(jet: FieldJet) -> dict[str, np.ndarray]:
        metric = paired[id(jet)]
        return {"lap_g(u)": laplace_beltrami(jet, metric, spatial_axes=spatial_axes)}

    discoverer = FieldLawDiscoverer(max_degree=1, time_axis=time_axis)
    result = discoverer.discover(
        train, val, test, lhs_index=lhs_index, extra_columns_fn=extra
    )
    return {
        "equation": result.formula(),
        "selected_terms": result.active_terms(),
        "validation_rmse": result.validation_rmse,
        "test_rmse": result.test_rmse,
        "target_scale": result.target_scale,
    }


def evaluate_geometric_discovery(*, seed: int = 0) -> dict[str, object]:
    """Smoke run: recover the hyperbolic-plane heat law from exact jets."""
    train, val, test, metrics, hidden = make_geometric_heat_split(seed=seed)
    return {
        "geometric_heat": {
            "hidden_law": hidden,
            **discover_geometric_heat_law(train, val, test, metrics),
        }
    }


def gaussian_curvature_2d(metric: MetricField) -> np.ndarray:
    r"""Gaussian curvature ``K = R / 2`` of a 2-D metric, shape ``(n,)``."""
    if metric.dim != 2:
        raise ValueError(f"Gaussian curvature is 2-D only, got dim {metric.dim}")
    return np.asarray(0.5 * scalar_curvature(metric), dtype=float)

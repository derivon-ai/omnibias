# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multivariate (vector-calculus) neural-jet discovery: PDEs, not just ODEs.

:mod:`omnibias.symbolic.discovery` discovers *one-dimensional* implicit identities
among ``x, y, dy, d2y, ...``. This module lifts the same closed-form machinery to
**many input variables**, unlocking the full vector-calculus / partial-differential
surface: gradient, divergence, curl, Laplacian, Hessian, the Ito / Fokker--Planck
generator and the (constant-metric) Laplace--Beltrami operator -- and the
multivariate / space--time PDE laws they express (Laplace, heat, wave, advection,
Burgers, Helmholtz, ...).

The key primitive is :func:`extract_field_jet`. For a random-feature field

.. math::

    u(x) = \sum_h c_h\,\sigma(z_h) + b,\qquad z_h = \sum_i W_{hi}\,\tilde x_i + \beta_h,
    \quad \tilde x_i = (x_i - m_i)/s_i,

the inner map is *affine*, so the multivariate Faa di Bruno chain rule collapses to
a single surviving term and **every** mixed partial is exact closed form:

.. math::

    \partial^\alpha u(x) = \sum_h c_h\,\sigma^{(|\alpha|)}(z_h)
        \prod_i \bigl(W_{hi}/s_i\bigr)^{\alpha_i}.

One activation-tower evaluation per total order ``k = |\alpha|`` yields *all*
order-``k`` partials at once -- the multivariate twin of the omnibias contract
(one ``sigma`` eval regardless of order). The multi-index bookkeeping reuses the
canonical ordering of :mod:`omnibias.core.multi_index`, so the columns line up
row-for-row with the :mod:`omnibias.jax.jet_mv` / :mod:`omnibias.torch.jet_mv`
kernels.

:class:`FieldLawDiscoverer` then runs the same train/validation/test sparse
relation search as :class:`~omnibias.symbolic.discovery.NeuralJetDiscoverer` over
the operator columns, recovering PDE coefficients (e.g. ``u_t = 0.12*u_xx``)
without ever being handed a named differential operator.

Honesty: the operator columns are **exact closed form**, but the sparse relation
search (STLSQ, via :mod:`omnibias.symbolic.discovery`) is a **numerical,
non-differentiable** numpy least-squares step -- the recovered PDE coefficients
are a numerical fit, not a closed-form identity.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations_with_replacement
from typing import Any

import numpy as np
from omnibias.core.multi_index import multi_indices
from omnibias.symbolic.diagnostics import (
    divergence_objective_term,
    residual_dependence_report,
    residual_distribution_report,
)
from omnibias.symbolic.discovery import (
    SparseEquation,
    fit_sparse_equation,
    rmse,
)

MultiIndex = tuple[int, ...]


# --------------------------------------------------------------------------- #
# Multivariate neural field + closed-form mixed-partial extraction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NeuralFieldND:
    """A ``d``-input random-feature omnibias field with closed-form partials.

    ``u(x) = c . sigma(W xtilde + beta) + b`` where ``xtilde = (x - x_mean)/x_scale``
    is the per-coordinate standardisation. All mixed partials are exact (see
    :func:`extract_field_jet`); the field is fit by solving the linear readout in
    :func:`fit_neural_field_nd`.
    """

    W: np.ndarray  # (hidden, d)
    beta: np.ndarray  # (hidden,)
    c: np.ndarray  # (hidden,)
    b: float
    x_mean: np.ndarray  # (d,)
    x_scale: np.ndarray  # (d,)
    activation: str
    var_names: tuple[str, ...]
    train_rmse: float = 0.0

    @property
    def dim(self) -> int:
        return int(self.W.shape[1])


@dataclass(frozen=True)
class FieldJet:
    """Samples of a scalar field and *all* its mixed partials up to ``order``.

    ``partials[alpha]`` is the length-``n`` array of ``partial^alpha u`` at the
    sample points :attr:`X` (shape ``(n, d)``), keyed by multi-index ``alpha`` in
    the canonical :func:`omnibias.core.multi_index.multi_indices` order. ``var_names``
    name the coordinates (e.g. ``("x", "t")``) for human-readable operator labels.
    """

    X: np.ndarray  # (n, d)
    order: int
    partials: dict[MultiIndex, np.ndarray]
    var_names: tuple[str, ...]

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def value(self) -> np.ndarray:
        return self.partials[(0,) * self.dim]

    def partial(self, alpha: MultiIndex) -> np.ndarray:
        if len(alpha) != self.dim:
            raise ValueError(f"alpha {alpha} has wrong length for dim {self.dim}")
        if sum(alpha) > self.order:
            raise ValueError(f"alpha {alpha} exceeds jet order {self.order}")
        return self.partials[tuple(alpha)]


def _default_var_names(dim: int) -> tuple[str, ...]:
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    return tuple(f"x{i}" for i in range(dim))


def fit_neural_field_nd(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 256,
    ridge: float = 1e-5,
    activation: str = "tanh",
    bandwidth: float = 1.0,
    var_names: tuple[str, ...] | None = None,
    seed: int = 0,
) -> NeuralFieldND:
    """Fit a smooth ``d``-input omnibias random-feature field by ridge readout.

    The multivariate twin of
    :func:`omnibias.symbolic.discovery.fit_neural_field_1d`: random hidden weights
    ``W`` (scaled by ``bandwidth / sqrt(d)`` so the pre-activation has unit-ish
    variance) and biases ``beta`` are frozen, and only the output layer ``c, b`` is
    solved. The resulting field exposes *exact* closed-form mixed partials via
    :func:`extract_field_jet`.
    """
    jnp = _jax_numpy()
    from omnibias.jax import get_activation

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n, d), got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must share the sample axis")
    if hidden < 1:
        raise ValueError(f"hidden must be >= 1, got {hidden}")
    dim = X.shape[1]
    names = var_names if var_names is not None else _default_var_names(dim)
    if len(names) != dim:
        raise ValueError("var_names must match the number of columns of X")

    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    xs = (X - x_mean) / x_scale
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0, size=(hidden, dim)) * (bandwidth / math.sqrt(dim))
    beta = rng.normal(0.0, 1.0, size=hidden)
    spec = get_activation(activation)
    z = xs @ W.T + beta[None, :]
    phi = np.asarray(spec.forward(jnp.asarray(z)))
    design = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)
    reg = ridge * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    pred = design @ coef
    return NeuralFieldND(
        W=W,
        beta=beta,
        c=coef[:-1],
        b=float(coef[-1]),
        x_mean=x_mean,
        x_scale=x_scale,
        activation=activation,
        var_names=tuple(names),
        train_rmse=rmse(y, pred),
    )


def extract_field_jet(
    field_nd: NeuralFieldND,
    X: np.ndarray,
    *,
    max_order: int = 2,
) -> FieldJet:
    r"""Exact closed-form mixed partials of a :class:`NeuralFieldND` up to ``max_order``.

    Because the inner map ``z = W xtilde + beta`` is affine, the only surviving
    term of the multivariate Faa di Bruno expansion is the top one, so

    .. math::

        \partial^\alpha u(x) = \sum_h c_h\,\sigma^{(|\alpha|)}(z_h)
            \prod_i \bigl(W_{hi}/s_i\bigr)^{\alpha_i}.

    Each total order ``k`` needs a *single* activation-tower evaluation
    ``sigma^{(k)}`` (via the omnibias fastpath); every order-``k`` partial is then a
    cheap contraction against the weight monomials. The returned :class:`FieldJet`
    keys partials by the canonical multi-index ordering.
    """
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    jnp = _jax_numpy()
    from omnibias.jax import get_activation

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n, d), got shape {X.shape}")
    dim = field_nd.dim
    if X.shape[1] != dim:
        raise ValueError(f"X has {X.shape[1]} columns but field has dim {dim}")
    xs = (X - field_nd.x_mean) / field_nd.x_scale
    z = xs @ field_nd.W.T + field_nd.beta[None, :]
    spec = get_activation(field_nd.activation)
    # Scaled hidden weights Ws[h, i] = W[h, i] / s_i (chain factor per coordinate).
    ws = field_nd.W / field_nd.x_scale[None, :]

    # sigma^(k) for k = 0..max_order, one tower evaluation each.
    sigma_orders: list[np.ndarray] = []
    for k in range(max_order + 1):
        if k == 0:
            sigma_orders.append(np.asarray(spec.forward(jnp.asarray(z))))
        else:
            if spec.fastpath is None:
                raise TypeError(
                    f"activation {field_nd.activation!r} has no derivative fastpath"
                )
            sigma_orders.append(np.asarray(spec.fastpath(jnp.asarray(z), k)))

    partials: dict[MultiIndex, np.ndarray] = {}
    for alpha in multi_indices(dim, max_order):
        k = sum(alpha)
        sig = sigma_orders[k]
        wmono = np.ones(field_nd.W.shape[0], dtype=float)
        for i, a in enumerate(alpha):
            if a:
                wmono = wmono * ws[:, i] ** a
        col = (sig * wmono[None, :]) @ field_nd.c
        if k == 0:
            col = col + field_nd.b
        partials[alpha] = np.asarray(col, dtype=float)
    return FieldJet(X=X, order=max_order, partials=partials, var_names=field_nd.var_names)


def analytic_field_jet(
    X: np.ndarray,
    partials: dict[MultiIndex, np.ndarray],
    *,
    order: int,
    var_names: tuple[str, ...] | None = None,
) -> FieldJet:
    """Assemble a :class:`FieldJet` from explicitly supplied partial arrays.

    Used for exact synthetic PDE datasets and tests: every multi-index ``alpha``
    with ``|alpha| <= order`` must be present in ``partials`` and shaped ``(n,)``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n, d), got shape {X.shape}")
    dim = X.shape[1]
    names = var_names if var_names is not None else _default_var_names(dim)
    if len(names) != dim:
        raise ValueError("var_names must match the number of columns of X")
    out: dict[MultiIndex, np.ndarray] = {}
    for alpha in multi_indices(dim, order):
        if alpha not in partials:
            raise ValueError(f"missing partial for multi-index {alpha}")
        col = np.asarray(partials[alpha], dtype=float).reshape(-1)
        if col.shape[0] != X.shape[0]:
            raise ValueError(f"partial {alpha} has wrong length {col.shape[0]}")
        out[alpha] = col
    return FieldJet(X=X, order=order, partials=out, var_names=tuple(names))


# --------------------------------------------------------------------------- #
# Exact vector-calculus operators (read off the closed-form partials)
# --------------------------------------------------------------------------- #
def _unit(dim: int, i: int) -> MultiIndex:
    return tuple(1 if j == i else 0 for j in range(dim))


def _pair(dim: int, i: int, j: int) -> MultiIndex:
    out = [0] * dim
    out[i] += 1
    out[j] += 1
    return tuple(out)


def _resolve_axes(jet: FieldJet, axes: Sequence[int] | None) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(jet.dim))
    resolved = tuple(int(a) for a in axes)
    for a in resolved:
        if a < 0 or a >= jet.dim:
            raise ValueError(f"axis {a} out of range for dim {jet.dim}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate axes in {resolved}")
    return resolved


def field_derivative_jet(jet: FieldJet, axis: int) -> FieldJet:
    r"""The :class:`FieldJet` of ``partial u / partial x_axis`` (order ``order - 1``).

    Shifts every stored partial up by one along ``axis`` (``partial^alpha(partial_a u)
    = partial^{alpha + e_a} u``), so the result is the exact derivative field one order
    lower. This is the primitive for building a vector field from a potential
    (``grad phi = [field_derivative_jet(phi, i) for i]``) and for composing
    :func:`field_divergence` / :func:`field_curl` of derived fields.
    """
    if jet.order < 1:
        raise ValueError(f"derivative jet needs order >= 1, got {jet.order}")
    if axis < 0 or axis >= jet.dim:
        raise ValueError(f"axis {axis} out of range for dim {jet.dim}")
    new_order = jet.order - 1
    out: dict[MultiIndex, np.ndarray] = {}
    for alpha in multi_indices(jet.dim, new_order):
        shifted = list(alpha)
        shifted[axis] += 1
        out[alpha] = jet.partials[tuple(shifted)]
    return FieldJet(X=jet.X, order=new_order, partials=out, var_names=jet.var_names)


def field_value(jet: FieldJet) -> np.ndarray:
    """The field value ``u`` (the zeroth partial)."""
    return jet.value()


def field_gradient(jet: FieldJet, *, axes: Sequence[int] | None = None) -> np.ndarray:
    r"""Gradient :math:`\nabla u`, shape ``(n, len(axes))`` (all axes by default)."""
    if jet.order < 1:
        raise ValueError(f"gradient needs order >= 1, got {jet.order}")
    ax = _resolve_axes(jet, axes)
    return np.stack([jet.partial(_unit(jet.dim, i)) for i in ax], axis=1)


def field_hessian(jet: FieldJet, *, axes: Sequence[int] | None = None) -> np.ndarray:
    r"""Hessian :math:`\partial^2_{ij} u`, shape ``(n, m, m)`` with ``m = len(axes)``."""
    if jet.order < 2:
        raise ValueError(f"hessian needs order >= 2, got {jet.order}")
    ax = _resolve_axes(jet, axes)
    m = len(ax)
    n = jet.n
    out = np.empty((n, m, m), dtype=float)
    for a, i in enumerate(ax):
        for b, j in enumerate(ax):
            out[:, a, b] = jet.partial(_pair(jet.dim, i, j))
    return out


def field_laplacian(jet: FieldJet, *, axes: Sequence[int] | None = None) -> np.ndarray:
    r"""Laplacian :math:`\Delta u = \sum_{i} \partial^2_{ii} u` over ``axes``."""
    if jet.order < 2:
        raise ValueError(f"laplacian needs order >= 2, got {jet.order}")
    ax = _resolve_axes(jet, axes)
    total = np.zeros(jet.n, dtype=float)
    for i in ax:
        total = total + jet.partial(_pair(jet.dim, i, i))
    return total


def field_grad_norm_sq(
    jet: FieldJet, *, axes: Sequence[int] | None = None
) -> np.ndarray:
    r""":math:`|\nabla u|^2 = \sum_i (\partial_i u)^2` over ``axes`` (the eikonal term)."""
    g = field_gradient(jet, axes=axes)
    return np.asarray(np.sum(g * g, axis=1), dtype=float)


def field_divergence(
    components: Sequence[FieldJet], *, axes: Sequence[int] | None = None
) -> np.ndarray:
    r"""Divergence :math:`\nabla\cdot F = \sum_i \partial_i F_i` of a vector field.

    ``components[i]`` is the :class:`FieldJet` of the ``i``-th vector component; all
    must share sample points and dimension. ``axes`` picks the differentiation axis
    per component (defaults to ``0, 1, ...``); ``len(components) == len(axes)``.
    """
    if len(components) == 0:
        raise ValueError("divergence needs at least one component")
    dim = components[0].dim
    ax = tuple(range(len(components))) if axes is None else tuple(int(a) for a in axes)
    if len(ax) != len(components):
        raise ValueError("axes length must match number of components")
    total = np.zeros(components[0].n, dtype=float)
    for comp, a in zip(components, ax, strict=True):
        if comp.dim != dim:
            raise ValueError("all components must share dimension")
        if comp.order < 1:
            raise ValueError("divergence needs order >= 1 components")
        total = total + comp.partial(_unit(dim, a))
    return total


def field_curl(components: Sequence[FieldJet]) -> np.ndarray:
    r"""Curl of a 2-D (scalar vorticity) or 3-D (vector) field.

    * 2-D: ``[F0, F1]`` -> scalar ``omega = partial_0 F1 - partial_1 F0``, shape ``(n,)``;
    * 3-D: ``[F0, F1, F2]`` -> ``(n, 3)`` vector :math:`\nabla\times F`.
    """
    m = len(components)
    dim = components[0].dim
    for comp in components:
        if comp.dim != dim:
            raise ValueError("all components must share dimension")
        if comp.order < 1:
            raise ValueError("curl needs order >= 1 components")
    if m == 2 and dim == 2:
        f0, f1 = components
        return np.asarray(f1.partial(_unit(2, 0)) - f0.partial(_unit(2, 1)), dtype=float)
    if m == 3 and dim == 3:
        f0, f1, f2 = components
        cx = f2.partial(_unit(3, 1)) - f1.partial(_unit(3, 2))
        cy = f0.partial(_unit(3, 2)) - f2.partial(_unit(3, 0))
        cz = f1.partial(_unit(3, 0)) - f0.partial(_unit(3, 1))
        return np.stack([cx, cy, cz], axis=1)
    raise ValueError(
        f"curl supports 2-D (2 comps) or 3-D (3 comps); got {m} comps in dim {dim}"
    )


def field_ito_generator(
    jet: FieldJet,
    drift: np.ndarray,
    diffusion: np.ndarray,
    *,
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    r"""Ito / backward-Kolmogorov generator :math:`\mathcal L u`.

    .. math::

        \mathcal L u = \sum_i b_i\,\partial_i u
            + \tfrac12 \sum_{ij} (\sigma\sigma^\top)_{ij}\,\partial^2_{ij} u,

    the drift--diffusion operator of an Ito SDE ``dX = b\,dt + \sigma\,dW`` (and the
    Fokker--Planck adjoint's principal part). ``drift`` is ``(m,)``, ``diffusion``
    is the ``(m, m)`` matrix ``sigma sigma^T`` (a.k.a. ``2D``); both constant here,
    so the operator is *exact* given the closed-form partials. ``m = len(axes)``.
    """
    if jet.order < 2:
        raise ValueError(f"ito generator needs order >= 2, got {jet.order}")
    ax = _resolve_axes(jet, axes)
    m = len(ax)
    b = np.asarray(drift, dtype=float).reshape(-1)
    cov = np.asarray(diffusion, dtype=float)
    if b.shape[0] != m:
        raise ValueError(f"drift must have length {m}, got {b.shape[0]}")
    if cov.shape != (m, m):
        raise ValueError(f"diffusion must be ({m}, {m}), got {cov.shape}")
    grad = field_gradient(jet, axes=ax)
    hess = field_hessian(jet, axes=ax)
    drift_term = grad @ b
    diff_term = 0.5 * np.einsum("ij,nij->n", cov, hess)
    return np.asarray(drift_term + diff_term, dtype=float)


def field_anisotropic_laplacian(
    jet: FieldJet,
    metric_inv: np.ndarray,
    *,
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    r"""Constant-metric Laplace--Beltrami principal part :math:`\sum_{ij} g^{ij}\partial^2_{ij}u`.

    For a *constant* inverse metric ``g^{ij}`` (and unit metric determinant), the
    Laplace--Beltrami operator reduces to this anisotropic Laplacian, computed
    exactly from the Hessian. The full Riemannian operator (with the
    ``\partial_j(\sqrt{g} g^{ij})`` Christoffel term for a position-dependent metric)
    lives in :mod:`omnibias.geometry`; this is the honest closed-form constant-metric
    special case. ``metric_inv`` is the symmetric ``(m, m)`` matrix, ``m = len(axes)``.
    """
    if jet.order < 2:
        raise ValueError(f"anisotropic laplacian needs order >= 2, got {jet.order}")
    ax = _resolve_axes(jet, axes)
    m = len(ax)
    g = np.asarray(metric_inv, dtype=float)
    if g.shape != (m, m):
        raise ValueError(f"metric_inv must be ({m}, {m}), got {g.shape}")
    hess = field_hessian(jet, axes=ax)
    return np.asarray(np.einsum("ij,nij->n", g, hess), dtype=float)


def field_wirtinger(
    u_jet: FieldJet,
    v_jet: FieldJet | None = None,
    *,
    axes: tuple[int, int] = (0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    r"""Wirtinger derivatives :math:`(\partial_z f,\ \partial_{\bar z} f)` of ``f = u + i v``.

    With ``z = x + i y`` over the two ``axes`` ``(x, y)`` and the complex field
    ``f = u + i v`` (``v`` defaults to ``0``),

    .. math::

        \partial_z f = \tfrac12 (f_x - i f_y), \qquad
        \partial_{\bar z} f = \tfrac12 (f_x + i f_y).

    Both are exact from the closed-form first partials. The holomorphy /
    Cauchy--Riemann test is ``partial_{\bar z} f = 0`` (i.e. ``u_x = v_y`` and
    ``u_y = -v_x``); ``partial_z f`` is then the complex derivative ``f'(z)``. This is
    the complex-analysis face of the field surface (the twin of the
    :mod:`omnibias.fields` Wirtinger ops).
    """
    ax0, ax1 = axes
    ux = u_jet.partial(_unit(u_jet.dim, ax0))
    uy = u_jet.partial(_unit(u_jet.dim, ax1))
    if v_jet is None:
        fx: np.ndarray = ux.astype(complex)
        fy: np.ndarray = uy.astype(complex)
    else:
        vx = v_jet.partial(_unit(v_jet.dim, ax0))
        vy = v_jet.partial(_unit(v_jet.dim, ax1))
        fx = ux + 1j * vx
        fy = uy + 1j * vy
    d_z = 0.5 * (fx - 1j * fy)
    d_zbar = 0.5 * (fx + 1j * fy)
    return np.asarray(d_z), np.asarray(d_zbar)


# --------------------------------------------------------------------------- #
# Operator-column dictionary + multivariate relation library
# --------------------------------------------------------------------------- #
def _partial_suffix(alpha: MultiIndex, var_names: tuple[str, ...]) -> str:
    return "".join(var_names[i] * alpha[i] for i in range(len(alpha)))


def field_partial_name(alpha: MultiIndex, var_names: tuple[str, ...], *, lhs: str = "u") -> str:
    """Readable operator name for a partial, e.g. ``u``, ``u_x``, ``u_xt``, ``u_xx``."""
    if sum(alpha) == 0:
        return lhs
    return f"{lhs}_{_partial_suffix(alpha, var_names)}"


def field_operator_columns(
    jet: FieldJet,
    *,
    lhs: str = "u",
    max_partial_order: int | None = None,
    spatial_axes: Sequence[int] | None = None,
    include_laplacian: bool = True,
    include_grad_norm_sq: bool = False,
) -> dict[str, np.ndarray]:
    """Named base operator columns of a scalar field jet.

    Always includes the value ``u`` and every mixed partial up to
    ``max_partial_order`` (default: the jet order). Optionally adds the spatial
    Laplacian ``lap(u)`` (over ``spatial_axes``) and the eikonal term
    ``|grad u|^2``. These are the atoms multiplied together by
    :func:`build_field_relation_library`.
    """
    order = jet.order if max_partial_order is None else int(max_partial_order)
    if order < 0 or order > jet.order:
        raise ValueError(f"max_partial_order must be in [0, {jet.order}], got {order}")
    cols: dict[str, np.ndarray] = {}
    for alpha in multi_indices(jet.dim, order):
        cols[field_partial_name(alpha, jet.var_names, lhs=lhs)] = jet.partial(alpha)
    if include_laplacian and jet.order >= 2:
        cols[f"lap({lhs})"] = field_laplacian(jet, axes=spatial_axes)
    if include_grad_norm_sq and jet.order >= 1:
        cols[f"|grad {lhs}|^2"] = field_grad_norm_sq(jet, axes=spatial_axes)
    return cols


def build_field_relation_library(
    jet: FieldJet,
    *,
    lhs_index: MultiIndex,
    max_degree: int = 1,
    lhs: str = "u",
    time_axis: int | None = None,
    rhs_orders: Sequence[int] | None = None,
    max_partial_order: int | None = None,
    spatial_axes: Sequence[int] | None = None,
    include_laplacian: bool = False,
    include_grad_norm_sq: bool = False,
    extra_columns: dict[str, np.ndarray] | None = None,
    exclude: Sequence[str] = (),
) -> tuple[np.ndarray, list[str]]:
    r"""Polynomial design over operator columns, excluding the LHS partial.

    The multivariate twin of
    :func:`omnibias.symbolic.discovery.build_jet_relation_library`. Base atoms are
    the partials of ``jet`` (plus the optional ``lap``/``|grad|^2`` composites and
    any ``extra_columns``); the chosen RHS atoms are multiplied into monomials up to
    total ``max_degree``.

    Two physically-motivated restrictions tame the well-known PDE-library
    degeneracy (correlated derivative columns admit several exact relations):

    * ``time_axis`` -- the *method-of-lines* restriction for evolution PDEs: drop
      every partial whose derivative order along ``time_axis`` is ``>=`` that of the
      LHS, so the RHS contains no equal/higher time derivative (e.g. ``u_t = F`` with
      ``F`` free of ``u_t, u_tt, u_xt``).
    * ``rhs_orders`` -- keep only partials whose *total* order lies in this set (e.g.
      ``(2,)`` recovers the elliptic principal part ``u_xx = -u_yy``).
    """
    if max_degree < 1:
        raise ValueError(f"max_degree must be >= 1, got {max_degree}")
    order = jet.order if max_partial_order is None else int(max_partial_order)
    if order < 0 or order > jet.order:
        raise ValueError(f"max_partial_order must be in [0, {jet.order}], got {order}")
    if len(lhs_index) != jet.dim:
        raise ValueError(f"lhs_index {lhs_index} has wrong length for dim {jet.dim}")
    lhs_name = field_partial_name(lhs_index, jet.var_names, lhs=lhs)
    orders_set = None if rhs_orders is None else {int(o) for o in rhs_orders}
    drop = set(exclude) | {lhs_name}

    atom_names: list[str] = []
    atoms: list[np.ndarray] = []
    for alpha in multi_indices(jet.dim, order):
        name = field_partial_name(alpha, jet.var_names, lhs=lhs)
        if name in drop:
            continue
        if orders_set is not None and sum(alpha) not in orders_set:
            continue
        if time_axis is not None and alpha[time_axis] >= lhs_index[time_axis]:
            continue
        atom_names.append(name)
        atoms.append(jet.partial(alpha))
    if include_laplacian and jet.order >= 2:
        name = f"lap({lhs})"
        if name not in drop and (orders_set is None or 2 in orders_set):
            atom_names.append(name)
            atoms.append(field_laplacian(jet, axes=spatial_axes))
    if include_grad_norm_sq and jet.order >= 1:
        name = f"|grad {lhs}|^2"
        if name not in drop and (orders_set is None or 2 in orders_set):
            atom_names.append(name)
            atoms.append(field_grad_norm_sq(jet, axes=spatial_axes))
    if extra_columns:
        for name, col in extra_columns.items():
            if name not in drop:
                atom_names.append(name)
                atoms.append(np.asarray(col, dtype=float).reshape(-1))
    if not atoms:
        raise ValueError("relation library has no candidate columns after exclusions")

    design_cols: list[np.ndarray] = []
    term_names: list[str] = []
    for degree in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(len(atoms)), degree):
            design_cols.append(np.prod([atoms[k] for k in combo], axis=0))
            term_names.append(
                "*".join(
                    _power_name(atom_names[k], combo.count(k)) for k in sorted(set(combo))
                )
            )
    return np.stack(design_cols, axis=1), term_names


# --------------------------------------------------------------------------- #
# Multivariate PDE-law discoverer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldLawResult:
    """Best compressed PDE-style law found among the operator columns."""

    lhs_name: str
    equation: SparseEquation
    validation_rmse: float
    test_rmse: float
    selection_score: float
    target_scale: float
    family: str = "field_operator_relation"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def formula(self) -> str:
        return str(self.equation.formula(lhs=self.lhs_name))

    def active_terms(self) -> list[dict[str, float | str]]:
        terms: list[dict[str, float | str]] = self.equation.active_terms()
        return terms


@dataclass(frozen=True)
class FieldLawDiscoverer:
    """Sparse PDE-law discovery over the exact vector-calculus operator columns.

    Mirrors :class:`~omnibias.symbolic.discovery.NeuralJetDiscoverer` but in the
    multivariate operator space: given a left-hand operator ``lhs_name`` (e.g.
    ``"u_t"``), it searches a sparse relation among the remaining operator
    monomials (``u``, ``u_x``, ``u_xx``, ``lap(u)``, ...) over a ridge/threshold
    grid, selecting on validation RMSE + complexity (+ optional divergence
    objective) and reporting test residual diagnostics.
    """

    max_degree: int = 1
    alphas: tuple[float, ...] = (1e-12, 1e-10, 1e-8, 1e-6)
    thresholds: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-3)
    complexity_weight: float = 2e-3
    time_axis: int | None = None
    rhs_orders: tuple[int, ...] | None = None
    max_partial_order: int | None = None
    spatial_axes: tuple[int, ...] | None = None
    include_laplacian: bool = False
    include_grad_norm_sq: bool = False
    divergence_objective: str | None = None
    divergence_weight: float = 1.0
    diagnostics_bins: int = 32
    dependence_bins: int = 16

    def discover(
        self,
        train: FieldJet,
        val: FieldJet,
        test: FieldJet,
        *,
        lhs_index: MultiIndex,
        lhs: str = "u",
        exclude: Sequence[str] = (),
        extra_columns_fn: Callable[[FieldJet], dict[str, np.ndarray]] | None = None,
    ) -> FieldLawResult:
        """Search a sparse PDE law for ``lhs_index`` over the operator columns.

        ``extra_columns_fn`` injects custom *exact* operator columns (one dict per
        jet, so split-specific evaluation points are honoured) as extra degree-1
        atoms -- e.g. a Laplace--Beltrami column ``{"lap_g(u)": laplace_beltrami(jet,
        metric)}`` to recover geometric heat flow ``u_t = lap_g(u)``.
        """
        lhs_name = field_partial_name(lhs_index, train.var_names, lhs=lhs)

        def _library(jet: FieldJet) -> tuple[np.ndarray, list[str]]:
            extra = None if extra_columns_fn is None else extra_columns_fn(jet)
            return build_field_relation_library(
                jet,
                lhs_index=lhs_index,
                max_degree=self.max_degree,
                lhs=lhs,
                time_axis=self.time_axis,
                rhs_orders=self.rhs_orders,
                max_partial_order=self.max_partial_order,
                spatial_axes=self.spatial_axes,
                include_laplacian=self.include_laplacian,
                include_grad_norm_sq=self.include_grad_norm_sq,
                extra_columns=extra,
                exclude=exclude,
            )

        train_design, names = _library(train)
        val_design, _ = _library(val)
        test_design, _ = _library(test)
        target_train = train.partial(lhs_index)
        target_val = val.partial(lhs_index)
        target_test = test.partial(lhs_index)
        scale = float(np.std(target_val))
        if scale < 1e-12:
            scale = 1.0
        val_feat = val_design
        test_feat = test_design

        best: FieldLawResult | None = None
        best_diagnostics: dict[str, object] = {}
        for alpha in self.alphas:
            for threshold in self.thresholds:
                equation = fit_sparse_equation(
                    train_design, target_train, names, alpha=alpha, threshold=threshold
                )
                val_pred = equation.predict(val_design)
                val_rmse = rmse(target_val, val_pred)
                active_count = len(equation.active_terms())
                score = val_rmse / scale + self.complexity_weight * active_count
                if self.divergence_objective is not None:
                    score += self.divergence_weight * divergence_objective_term(
                        self.divergence_objective,
                        val_feat,
                        target_val - val_pred,
                        bins=self.diagnostics_bins,
                        dependence_bins=self.dependence_bins,
                    )
                test_pred = equation.predict(test_design)
                result = FieldLawResult(
                    lhs_name=lhs_name,
                    equation=equation,
                    validation_rmse=val_rmse,
                    test_rmse=rmse(target_test, test_pred),
                    selection_score=score,
                    target_scale=scale,
                )
                if best is None or result.selection_score < best.selection_score:
                    best = result
                    resid = target_test - test_pred
                    best_diagnostics = residual_distribution_report(
                        resid, bins=self.diagnostics_bins
                    )
                    best_diagnostics.update(
                        residual_dependence_report(
                            test_feat, resid, bins=self.dependence_bins
                        )
                    )
        if best is None:
            raise RuntimeError(
                "FieldLawDiscoverer.search produced no candidates; "
                "check alphas and thresholds"
            )
        return replace(best, diagnostics=best_diagnostics)


# --------------------------------------------------------------------------- #
# Exact synthetic PDE datasets (analytic fields -> machine-precision recovery)
# --------------------------------------------------------------------------- #
def _three_point_grids(
    box: Sequence[tuple[float, float]],
    *,
    counts: tuple[int, int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three independent uniform point clouds in an axis-aligned box (train/val/test)."""
    rng = np.random.default_rng(seed)
    lows = np.asarray([lo for lo, _ in box], dtype=float)
    highs = np.asarray([hi for _, hi in box], dtype=float)
    out = []
    for n in counts:
        out.append(rng.uniform(lows, highs, size=(n, len(box))))
    return out[0], out[1], out[2]


def make_laplace_field_split(
    *, seed: int = 0, counts: tuple[int, int, int] = (300, 200, 200)
) -> tuple[FieldJet, FieldJet, FieldJet, str]:
    r"""Two-mode harmonic field over ``(x, y)`` with ``Delta u = 0`` (so ``u_xx = -u_yy``).

    ``u = e^{x}\sin y + 0.5\,e^{2x}\sin 2y``; each summand is harmonic, and the two
    distinct frequencies break the single-mode degeneracy ``u_xx = u`` so that
    ``u_xx = -u_yy`` is the *unique* one-term relation recoverable from the exact
    partials.
    """
    names = ("x", "y")
    tr, va, te = _three_point_grids([(-1.0, 1.0), (-1.0, 1.0)], counts=counts, seed=seed)

    def jet(X: np.ndarray) -> FieldJet:
        x, y = X[:, 0], X[:, 1]
        e1, e2 = np.exp(x), np.exp(2.0 * x)
        s1, c1 = np.sin(y), np.cos(y)
        s2, c2 = np.sin(2.0 * y), np.cos(2.0 * y)
        m1 = e1 * s1
        m2 = 0.5 * e2 * s2
        partials: dict[MultiIndex, np.ndarray] = {
            (0, 0): m1 + m2,
            (1, 0): m1 + 2.0 * m2,
            (0, 1): e1 * c1 + 0.5 * e2 * 2.0 * c2,
            (2, 0): m1 + 4.0 * m2,
            (1, 1): e1 * c1 + 2.0 * e2 * c2,
            (0, 2): -m1 - 4.0 * m2,
        }
        return analytic_field_jet(X, partials, order=2, var_names=names)

    return jet(tr), jet(va), jet(te), "u_xx = -u_yy  (Laplace: Delta u = 0)"


def _heat_modes(
    X: np.ndarray, *, k: float, amps: tuple[float, ...], names: tuple[str, str]
) -> FieldJet:
    """Multi-mode 1-D heat field jet: u = sum_m a_m e^{-k m^2 pi^2 t} sin(m pi x)."""
    x, t = X[:, 0], X[:, 1]
    u = np.zeros_like(x)
    ux = np.zeros_like(x)
    ut = np.zeros_like(x)
    uxx = np.zeros_like(x)
    uxt = np.zeros_like(x)
    utt = np.zeros_like(x)
    for m, a in enumerate(amps, start=1):
        w = m * math.pi
        decay = np.exp(-k * w**2 * t)
        sx, cx = np.sin(w * x), np.cos(w * x)
        base = a * decay
        u += base * sx
        ux += base * w * cx
        uxx += -(w**2) * base * sx
        ut += -k * w**2 * base * sx
        uxt += -k * w**2 * base * w * cx
        utt += (k * w**2) ** 2 * base * sx
    partials: dict[MultiIndex, np.ndarray] = {
        (0, 0): u, (1, 0): ux, (0, 1): ut, (2, 0): uxx, (1, 1): uxt, (0, 2): utt,
    }
    return analytic_field_jet(X, partials, order=2, var_names=names)


def make_heat_field_split(
    *,
    diffusivity: float = 0.12,
    seed: int = 0,
    counts: tuple[int, int, int] = (400, 250, 250),
) -> tuple[FieldJet, FieldJet, FieldJet, str]:
    r"""Two-mode 1-D heat field over ``(x, t)`` satisfying ``u_t = k u_xx``.

    ``u = sin(pi x) e^{-k pi^2 t} + 0.5 sin(2 pi x) e^{-4 k pi^2 t}``. The second
    mode breaks the single-mode degeneracy ``u_t = -k pi^2 u`` so the unique
    one-term relation is the heat law ``u_t = k u_xx``.
    """
    names = ("x", "t")
    k = diffusivity
    tr, va, te = _three_point_grids([(0.0, 1.0), (0.0, 0.4)], counts=counts, seed=seed)
    amps = (1.0, 0.5)
    return (
        _heat_modes(tr, k=k, amps=amps, names=names),
        _heat_modes(va, k=k, amps=amps, names=names),
        _heat_modes(te, k=k, amps=amps, names=names),
        f"u_t = {k:g}*u_xx  (heat equation)",
    )


def _wave_modes(
    X: np.ndarray, *, c: float, amps: tuple[float, ...], names: tuple[str, str]
) -> FieldJet:
    """Multi-mode 1-D wave field jet: u = sum_m a_m sin(m pi x) cos(c m pi t)."""
    x, t = X[:, 0], X[:, 1]
    u = np.zeros_like(x)
    ux = np.zeros_like(x)
    ut = np.zeros_like(x)
    uxx = np.zeros_like(x)
    uxt = np.zeros_like(x)
    utt = np.zeros_like(x)
    for m, a in enumerate(amps, start=1):
        w = m * math.pi
        sx, cx = np.sin(w * x), np.cos(w * x)
        ct, st = np.cos(c * w * t), np.sin(c * w * t)
        u += a * sx * ct
        ux += a * w * cx * ct
        ut += -a * c * w * sx * st
        uxx += -a * w**2 * sx * ct
        uxt += -a * c * w**2 * cx * st
        utt += -a * (c * w) ** 2 * sx * ct
    partials: dict[MultiIndex, np.ndarray] = {
        (0, 0): u, (1, 0): ux, (0, 1): ut, (2, 0): uxx, (1, 1): uxt, (0, 2): utt,
    }
    return analytic_field_jet(X, partials, order=2, var_names=names)


def make_wave_field_split(
    *,
    speed: float = 1.3,
    seed: int = 0,
    counts: tuple[int, int, int] = (400, 250, 250),
) -> tuple[FieldJet, FieldJet, FieldJet, str]:
    r"""Two-mode 1-D wave field over ``(x, t)`` satisfying ``u_tt = c^2 u_xx``.

    ``u = sin(pi x) cos(c pi t) + 0.5 sin(2 pi x) cos(2 c pi t)`` -- two modes so the
    unique one-term relation is the wave law ``u_tt = c^2 u_xx``.
    """
    names = ("x", "t")
    c = speed
    tr, va, te = _three_point_grids([(0.0, 1.0), (0.0, 1.0)], counts=counts, seed=seed)
    amps = (1.0, 0.5)
    return (
        _wave_modes(tr, c=c, amps=amps, names=names),
        _wave_modes(va, c=c, amps=amps, names=names),
        _wave_modes(te, c=c, amps=amps, names=names),
        f"u_tt = {c**2:g}*u_xx  (wave equation)",
    )


def _burgers_wave_jet(
    X: np.ndarray, *, nu: float, c0: float, amp: float, names: tuple[str, str]
) -> dict[MultiIndex, np.ndarray]:
    """Exact partials of one Cole--Hopf travelling wave of viscous Burgers."""
    x, t = X[:, 0], X[:, 1]
    kx = amp / (2.0 * nu)
    kt = -amp * c0 / (2.0 * nu)
    xi = kx * x + kt * t
    th = np.tanh(xi)
    sech2 = 1.0 - th**2
    d1 = sech2  # d/dxi tanh
    d2 = -2.0 * th * sech2  # d2/dxi2 tanh
    u = c0 - amp * th
    ux = -amp * d1 * kx
    ut = -amp * d1 * kt
    uxx = -amp * d2 * kx**2
    uxt = -amp * d2 * kx * kt
    utt = -amp * d2 * kt**2
    return {(0, 0): u, (1, 0): ux, (0, 1): ut, (2, 0): uxx, (1, 1): uxt, (0, 2): utt}


def _stack_burgers(
    X: np.ndarray, *, nu: float, waves: tuple[tuple[float, float], ...], names: tuple[str, str]
) -> FieldJet:
    """Stack several Burgers travelling waves (same nu) so only the PDE law is shared."""
    chunks = np.array_split(X, len(waves), axis=0)
    parts: dict[MultiIndex, list[np.ndarray]] = {a: [] for a in multi_indices(2, 2)}
    xs: list[np.ndarray] = []
    for chunk, (c0, amp) in zip(chunks, waves, strict=True):
        xs.append(chunk)
        wj = _burgers_wave_jet(chunk, nu=nu, c0=c0, amp=amp, names=names)
        for a in parts:
            parts[a].append(wj[a])
    Xs = np.concatenate(xs, axis=0)
    partials = {a: np.concatenate(parts[a]) for a in parts}
    return analytic_field_jet(Xs, partials, order=2, var_names=names)


def make_burgers_field_split(
    *,
    viscosity: float = 0.1,
    seed: int = 0,
    counts: tuple[int, int, int] = (600, 360, 360),
) -> tuple[FieldJet, FieldJet, FieldJet, str]:
    r"""Stacked viscous-Burgers travelling waves over ``(x, t)`` -- the law is nonlinear.

    Each ``u(x, t) = c0 - A tanh(A (x - c0 t)/(2 nu))`` exactly satisfies
    ``u_t + u u_x = nu u_xx`` (an on-brand ``tanh`` shock). A single shock also
    trivially satisfies the *linear* travelling-wave relation ``u_t = -c0 u_x``; we
    therefore **stack several shocks with different speeds/amplitudes but the same
    nu**, so the only relation that holds across the whole dataset is the genuine
    nonlinear Burgers law (degree-2 in the operator library).
    """
    names = ("x", "t")
    nu = viscosity
    waves = ((0.6, 0.8), (-0.4, 0.6), (0.9, 1.0))
    tr, va, te = _three_point_grids([(-1.5, 1.5), (0.0, 1.0)], counts=counts, seed=seed)
    return (
        _stack_burgers(tr, nu=nu, waves=waves, names=names),
        _stack_burgers(va, nu=nu, waves=waves, names=names),
        _stack_burgers(te, nu=nu, waves=waves, names=names),
        f"u_t = -u*u_x + {nu:g}*u_xx  (viscous Burgers)",
    )


def make_heat2d_field_split(
    *,
    diffusivity: float = 0.1,
    seed: int = 0,
    counts: tuple[int, int, int] = (500, 320, 320),
) -> tuple[FieldJet, FieldJet, FieldJet, str]:
    r"""Two-mode 2-D heat field over ``(x, y, t)`` satisfying ``u_t = k (u_xx + u_yy)``.

    A genuinely multivariate (two spatial axes) law; the recovered equation
    ``u_t = k u_xx + k u_yy`` is the discrete Laplacian. ``u = sin(pi x) sin(pi y)
    e^{-2 k pi^2 t} + 0.5 sin(2 pi x) sin(pi y) e^{-5 k pi^2 t}``.
    """
    names = ("x", "y", "t")
    k = diffusivity
    tr, va, te = _three_point_grids(
        [(0.0, 1.0), (0.0, 1.0), (0.0, 0.3)], counts=counts, seed=seed
    )
    # Three independent spatial profiles so that {u, u_xx, u_yy} are linearly
    # independent and ``u_t = k(u_xx + u_yy)`` is the unique recoverable relation.
    modes = ((1, 1, 1.0), (2, 1, 0.6), (1, 2, 0.4))

    def jet(X: np.ndarray) -> FieldJet:
        x, y, t = X[:, 0], X[:, 1], X[:, 2]
        acc = {a: np.zeros_like(x) for a in multi_indices(3, 2)}
        for mx, my, a in modes:
            wx, wy = mx * math.pi, my * math.pi
            lam = k * (wx**2 + wy**2)
            decay = a * np.exp(-lam * t)
            sx, cx = np.sin(wx * x), np.cos(wx * x)
            sy, cy = np.sin(wy * y), np.cos(wy * y)
            base = decay * sx * sy
            acc[(0, 0, 0)] += base
            acc[(1, 0, 0)] += decay * wx * cx * sy
            acc[(0, 1, 0)] += decay * sx * wy * cy
            acc[(0, 0, 1)] += -lam * base
            acc[(2, 0, 0)] += -(wx**2) * base
            acc[(0, 2, 0)] += -(wy**2) * base
            acc[(0, 0, 2)] += lam**2 * base
            acc[(1, 1, 0)] += decay * wx * cx * wy * cy
            acc[(1, 0, 1)] += -lam * decay * wx * cx * sy
            acc[(0, 1, 1)] += -lam * decay * sx * wy * cy
        return analytic_field_jet(X, acc, order=2, var_names=names)

    return jet(tr), jet(va), jet(te), f"u_t = {k:g}*u_xx + {k:g}*u_yy  (2-D heat)"


def discover_field_pde_law(
    train: FieldJet,
    val: FieldJet,
    test: FieldJet,
    *,
    lhs_index: MultiIndex,
    max_degree: int = 1,
    time_axis: int | None = None,
    rhs_orders: Sequence[int] | None = None,
    spatial_axes: Sequence[int] | None = None,
    include_laplacian: bool = False,
    exclude: Sequence[str] = (),
) -> dict[str, object]:
    """Discover and report a single PDE law for the ``lhs_index`` partial."""
    discoverer = FieldLawDiscoverer(
        max_degree=max_degree,
        time_axis=time_axis,
        rhs_orders=None if rhs_orders is None else tuple(rhs_orders),
        spatial_axes=None if spatial_axes is None else tuple(spatial_axes),
        include_laplacian=include_laplacian,
    )
    result = discoverer.discover(train, val, test, lhs_index=lhs_index, exclude=exclude)
    return {
        "equation": result.formula(),
        "selected_terms": result.active_terms(),
        "validation_rmse": result.validation_rmse,
        "test_rmse": result.test_rmse,
        "target_scale": result.target_scale,
    }


def evaluate_field_pde_discovery(*, seed: int = 0) -> dict[str, object]:
    """Recover the canonical PDE laws from exact analytic field jets (a smoke run).

    Each case uses the physically-motivated restriction: ``rhs_orders=(2,)`` for the
    elliptic Laplace principal part, and the ``time_axis`` method-of-lines
    restriction for the evolution equations (heat / wave / Burgers / 2-D heat).
    """
    laplace = make_laplace_field_split(seed=seed)
    heat = make_heat_field_split(seed=seed)
    wave = make_wave_field_split(seed=seed)
    burgers = make_burgers_field_split(seed=seed)
    heat2d = make_heat2d_field_split(seed=seed)
    return {
        "laplace": {
            "hidden_law": laplace[3],
            **discover_field_pde_law(
                laplace[0], laplace[1], laplace[2],
                lhs_index=(2, 0), max_degree=1, rhs_orders=(2,),
            ),
        },
        "heat": {
            "hidden_law": heat[3],
            **discover_field_pde_law(
                heat[0], heat[1], heat[2],
                lhs_index=(0, 1), max_degree=1, time_axis=1,
            ),
        },
        "wave": {
            "hidden_law": wave[3],
            **discover_field_pde_law(
                wave[0], wave[1], wave[2],
                lhs_index=(0, 2), max_degree=1, time_axis=1,
            ),
        },
        "burgers": {
            "hidden_law": burgers[3],
            **discover_field_pde_law(
                burgers[0], burgers[1], burgers[2],
                lhs_index=(0, 1), max_degree=2, time_axis=1,
            ),
        },
        "heat2d": {
            "hidden_law": heat2d[3],
            **discover_field_pde_law(
                heat2d[0], heat2d[1], heat2d[2],
                lhs_index=(0, 0, 1), max_degree=1, time_axis=2,
            ),
        },
    }


#: A field kernel ``K(X, T)`` called with query points ``X`` of shape ``(n, d)``
#: and quadrature nodes ``T`` of shape ``(m, d)``, returning ``(n, m)``. The
#: kernel does its own broadcasting, which is what keeps it readable in more than
#: one dimension.
FieldIntegralKernel = Callable[[np.ndarray, np.ndarray], np.ndarray]


def measure_integral_columns(
    measure: object,
    kernels: Mapping[str, FieldIntegralKernel],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> Callable[[FieldJet], dict[str, np.ndarray]]:
    r"""Fredholm columns ``int_Omega K(x, t) u(t) dmu(t)`` for a field law search.

    The multi-dimensional counterpart of
    :func:`~omnibias.symbolic.discovery.build_jet_integral_features`, packaged as
    an ``extra_columns_fn`` so it plugs into
    :meth:`FieldLawDiscoverer.discover` with no new plumbing. It is what lets a
    PDE search express a **nonlocal** term -- a convolution, a screened Coulomb
    interaction, a memory kernel -- next to the local differential atoms.

    ``measure`` is an :class:`omnibias.measure.Measure` over the domain, or a
    factory ``X -> Measure`` invoked with each jet's sample points. Its nodes must
    equal the jet's ``X``, for the same reason as in the 1-D builder: ``u`` is
    known at the samples and nowhere else, so any other node set would require
    interpolation. Splits normally carry different points, which is why the
    factory form is usually the one you want.

    Only the global (Fredholm) family is offered. The causal Volterra and running
    families have no counterpart here: they need an ordering of the domain, and
    scattered points in ``d > 1`` do not have one.

    Honesty label: **numerical** -- the column is a quadrature, as accurate as the
    rule the measure carries. The differential atoms it sits beside remain exact.
    """
    kernels = dict(kernels)
    if not kernels:
        raise ValueError("measure_integral_columns needs at least one kernel")

    def columns(jet: FieldJet) -> dict[str, np.ndarray]:
        weights = _field_measure_weights(measure, jet.X, atol=atol, rtol=rtol)
        u = jet.value()
        out: dict[str, np.ndarray] = {}
        for name, kernel in kernels.items():
            k = np.asarray(kernel(jet.X, jet.X), dtype=float)
            if k.shape != (jet.n, jet.n):
                raise ValueError(
                    f"kernel {name!r} returned shape {k.shape}, expected "
                    f"{(jet.n, jet.n)} for (query points, quadrature nodes)"
                )
            out[f"F[{name}](u)"] = (k * weights[None, :]) @ u
        return out

    return columns


def _field_measure_weights(
    measure: object, points: np.ndarray, *, atol: float, rtol: float
) -> np.ndarray:
    """The measure's weights, after checking its nodes are the jet's samples."""
    try:  # pragma: no cover -- exercised, but the import guard itself is not
        from omnibias.measure import Measure
    except ModuleNotFoundError as exc:  # pragma: no cover -- optional dependency
        raise ModuleNotFoundError(
            "measure_integral_columns needs the omnibias-measure package; "
            "install 'omnibias-measure'."
        ) from exc
    resolved = measure(points) if not isinstance(measure, Measure) else measure
    if not isinstance(resolved, Measure):
        raise TypeError(
            "measure must be an omnibias.measure.Measure or a factory returning "
            f"one, got {type(resolved).__name__}"
        )
    nodes = np.asarray(resolved.nodes, dtype=float)
    if nodes.shape != points.shape or not np.allclose(
        nodes, points, rtol=rtol, atol=atol
    ):
        raise ValueError(
            f"the measure's nodes {nodes.shape} must be the jet's sample points "
            f"{points.shape}: u is only known at the samples, so a measure on "
            "other nodes could only be used by interpolating. Pass a factory "
            "'X -> Measure' so each split builds its rule on its own points"
        )
    return np.asarray(resolved.weights, dtype=float)


def _power_name(name: str, power: int) -> str:
    return name if power == 1 else f"{name}^{power}"


def _jax_numpy() -> Any:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    return jnp

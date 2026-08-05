# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Verified parameter-jet: a rigorous order-``N`` Taylor model of the training objective.

The rigorous multivariate jet :func:`omnibias.core.verified.jet_mv.mlp_jet_mv` differentiates a
network read-out with respect to its *inputs* (weights frozen). This module is its **parameter-space**
counterpart of *arbitrary order* ``N``: it builds the order-``N``
:class:`~omnibias.core.verified.taylor_model_mv.TaylorModelMV` of the training objective
``J(theta0 + delta) = L(theta0 + delta) + l2 ||theta0 + delta||^2`` over a box in the parameter
directions, either the **full** parameter space (``delta in [-r, r]^P``) or a low-dimensional
**subspace** (``delta = Q a``, ``a in [-r, r]^k``, given an orthonormal basis ``Q``).

The polynomial part is the *exact* Taylor expansion (its coefficients are :class:`Interval`
enclosures, tight up to outward rounding) and the scalar :attr:`ParamJet.remainder` rigorously bounds
the model-vs-truth error ``J(theta0 + delta) - m(delta)`` over the whole box -- so a single Taylor-
model pass yields every mixed partial up to total order ``N`` together with a sound tail bound.
Reading off order ``m`` gives the symmetric ``m``-th derivative tensor: :meth:`ParamJet.grad`
(``m = 1``), :meth:`ParamJet.hessian` (``m = 2``), or :meth:`ParamJet.tensor` (any ``m <= N``).

The one numerical kernel is the closed-form activation composition ``_compose_sigma_tm`` (Makino--Berz:
the exact ``sigma^(n)`` from :func:`~omnibias.core.verified.sigma.sigma_tower_interval` plus a Lagrange
remainder for the truncated series), so every term of total degree ``> N`` lands soundly in the
remainder. It is order-generic, which is what lets this module serve every order in one place;
:mod:`omnibias.verify._core.subspace_step` is the order-3 subspace special case built on top of it.

**Concrete win.** The full-``P`` order-2 jet yields the *entire* interval Hessian of the objective in a
**single** Taylor-model pass, versus the ``O(P^2)`` hyper-dual sweeps of
:class:`~omnibias.verify._core.param_loss.ParamSpaceLoss` -- an alternate Hessian provider for the
strict-local-min and global certificates.

Honest scope: the coefficient count is ``C(P + N, P)`` (full) or ``C(k + N, k)`` (subspace), so this is
for **tiny** networks, **low** order, or a **small** subspace -- ``tanh`` / ``sigmoid`` activations,
fixed data. It is a rigorous *local* Taylor model about ``theta0``, not a global or open-problem claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.multi_index import (
    index_position,
    multi_index_factorial,
    num_multi_indices,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.core.verified.taylor_model_mv import TaylorModelMV
from omnibias.verify._core.param_loss import MLPArchitecture

Data = Sequence[tuple[Sequence[float], Sequence[float]]]


# --------------------------------------------------------------------------- #
# Taylor-model activation composition + forward pass (shared, order-generic).
# --------------------------------------------------------------------------- #
def _compose_sigma_tm(
    u: TaylorModelMV, name: str, center: Sequence[float], radius: Sequence[float]
) -> TaylorModelMV:
    r"""Compose an activation ``sigma`` onto a Taylor model: ``sigma(u)``.

    Makino-Berz Taylor-model composition.  Expanding about the midpoint ``u0`` of ``u``'s constant
    coefficient, ``sigma(u) = sum_{n=0}^{N} sigma^(n)(u0)/n! * (u - u0)^n + Rc`` where ``N = u.order``.
    The finite sum is evaluated in Taylor-model arithmetic (so every term of total degree ``> N`` is
    absorbed rigorously into the running remainder), and the truncated-series Lagrange remainder

    .. math::

        Rc = \frac{\sigma^{(N+1)}(\mathrm{range}(u))}{(N+1)!}\,(u - u0)^{N+1}

    is added on top -- ``sigma^(N+1)`` enclosed over the *range* of ``u`` (which contains the true
    expansion point), times the ``(N+1)``-th power of the range of ``u - u0``.  Both use the exact
    closed-form :func:`~omnibias.core.verified.sigma.sigma_tower_interval` tower, so the composition
    needs a single transcendental enclosure per derivative order.
    """
    order = u.order
    dim = len(center)
    u0 = u.coeffs[0].mid
    u0_iv = Interval.point(u0)
    tower = sigma_tower_interval(name, u0_iv, order)  # sigma^(0..N)(u0)
    w = u - u0_iv  # (u - u0): constant coefficient is ~0 (residual rounding only)

    result = TaylorModelMV.constant(tower[0], center, radius, order)
    w_power = TaylorModelMV.constant(1.0, center, radius, order)
    fact = 1
    for n in range(1, order + 1):
        fact *= n
        w_power = w_power * w
        deriv_coeff = tower[n] * Interval.from_rational(Fraction(1, fact))
        result = result + w_power * deriv_coeff

    # Lagrange remainder of the truncated Taylor series (degree n >= N + 1 tail):
    # sigma^(N+1) over the range of u, times |u - u0|^{N+1} / (N+1)!.
    range_u = u.bound()
    sigma_tail = sigma_tower_interval(name, range_u, order + 1)[order + 1]
    fact_tail = fact * (order + 1)
    w_range = w.bound()
    series_remainder = (
        sigma_tail * Interval.from_rational(Fraction(1, fact_tail)) * w_range.pow_int(order + 1)
    )
    n_coeffs = num_multi_indices(dim, order)
    tail_tm = TaylorModelMV(
        center, radius, order, [Interval.point(0.0)] * n_coeffs, series_remainder
    )
    return result + tail_tm


def _normalize_data(arch: MLPArchitecture, data: Data) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    out: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for x, y in data:
        xt = tuple(float(v) for v in x)
        yt = tuple(float(v) for v in y)
        if len(xt) != arch.dims[0]:
            raise ValueError(f"input dim {len(xt)} != arch input dim {arch.dims[0]}")
        if len(yt) != arch.out_dim:
            raise ValueError(f"target dim {len(yt)} != arch output dim {arch.out_dim}")
        out.append((xt, yt))
    if not out:
        raise ValueError("data must be non-empty")
    return out


def _forward_tm(
    arch: MLPArchitecture,
    theta: Sequence[TaylorModelMV],
    x: Sequence[float],
    center: Sequence[float],
    radius: Sequence[float],
    order: int,
) -> list[TaylorModelMV]:
    """MLP forward pass in Taylor-model arithmetic (the weights are Taylor models)."""
    off = 0
    layers: list[tuple[list[list[TaylorModelMV]], list[TaylorModelMV]]] = []
    for n_out, n_in in arch.layer_shapes:
        weight = [[theta[off + r * n_in + c] for c in range(n_in)] for r in range(n_out)]
        off += n_out * n_in
        bias = [theta[off + r] for r in range(n_out)]
        off += n_out
        layers.append((weight, bias))
    a: list[TaylorModelMV] = [
        TaylorModelMV.constant(float(xi), center, radius, order) for xi in x
    ]
    last = len(layers) - 1
    for depth, (weight, bias) in enumerate(layers):
        z: list[TaylorModelMV] = []
        for o in range(len(weight)):
            acc = bias[o]
            row = weight[o]
            for j in range(len(row)):
                acc = acc + row[j] * a[j]
            z.append(acc)
        a = z if depth == last else [_compose_sigma_tm(zo, arch.activation, center, radius) for zo in z]
    return a


def _loss_tm(
    arch: MLPArchitecture,
    theta: Sequence[TaylorModelMV],
    data: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
    center: Sequence[float],
    radius: Sequence[float],
    order: int,
) -> TaylorModelMV:
    """Mean-squared-error training loss ``L(theta0 + delta)`` as an order-``N`` Taylor model."""
    n = len(data)
    acc = TaylorModelMV.constant(0.0, center, radius, order)
    for x, y in data:
        out = _forward_tm(arch, theta, x, center, radius, order)
        for o, yo in enumerate(y):
            diff = out[o] - Interval.point(float(yo))
            acc = acc + diff * diff
    return acc * Interval.from_rational(Fraction(1, n))


def _objective_tm(
    arch: MLPArchitecture,
    theta_tms: Sequence[TaylorModelMV],
    data: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
    center: Sequence[float],
    radius: Sequence[float],
    order: int,
    l2: float,
) -> TaylorModelMV:
    """``J = L + l2 ||theta||^2`` as an order-``N`` Taylor model (regulariser over all parameters)."""
    psi = _loss_tm(arch, theta_tms, data, center, radius, order)
    if l2 != 0.0:
        reg = TaylorModelMV.constant(0.0, center, radius, order)
        for tp in theta_tms:
            reg = reg + tp * tp
        psi = psi + reg * Interval.point(float(l2))
    return psi


# --------------------------------------------------------------------------- #
# The parameter-jet primitive.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParamJet:
    r"""An order-``N`` Taylor model of the training objective about ``theta0``.

    :attr:`tm` is the order-``order`` :class:`TaylorModelMV` of ``J(theta0 + delta)`` over the box in
    the ``dim`` parameter directions (``dim = P`` for a full-parameter jet, ``dim = k`` for a subspace
    jet with :attr:`basis` ``P x k``).  The polynomial coefficients are exact (up to outward rounding)
    and :attr:`remainder` rigorously bounds the model-vs-truth error over the box.  Read off the
    symmetric ``m``-th derivative tensor with :meth:`tensor` (``m <= order``); :meth:`grad` and
    :meth:`hessian` are the ``m = 1`` / ``m = 2`` shortcuts.
    """

    theta0: tuple[float, ...]
    order: int
    dim: int
    radius: float
    basis: tuple[tuple[float, ...], ...] | None
    tm: TaylorModelMV

    @property
    def remainder(self) -> Interval:
        """The rigorous model-vs-truth remainder over the whole box."""
        return self.tm.remainder

    def bound(self) -> Interval:
        """A sound enclosure of ``J`` itself over the box (polynomial range + remainder)."""
        return self.tm.bound()

    def _entry(self, idx: Sequence[int], pos: dict[tuple[int, ...], int]) -> Interval:
        r"""The mixed partial ``d^m J / dtheta_{idx}`` = ``alpha! * coeff_alpha`` at ``theta0``."""
        alpha = [0] * self.dim
        for i in idx:
            alpha[i] += 1
        alpha_t = tuple(alpha)
        coeff = self.tm.coeffs[pos[alpha_t]]
        fact = multi_index_factorial(alpha_t)
        return coeff if fact == 1 else coeff * Interval.from_rational(fact)

    def value(self) -> Interval:
        """The order-0 term ``J(theta0)`` (the constant coefficient)."""
        pos = index_position(self.dim, self.order)
        return self.tm.coeffs[pos[(0,) * self.dim]]

    def grad(self) -> tuple[Interval, ...]:
        """The gradient ``grad J(theta0)`` as a length-``dim`` tuple of interval enclosures."""
        pos = index_position(self.dim, self.order)
        return tuple(self._entry((i,), pos) for i in range(self.dim))

    def hessian(self) -> tuple[tuple[Interval, ...], ...]:
        """The Hessian ``Hess J(theta0)`` as a ``dim x dim`` tuple of interval enclosures."""
        pos = index_position(self.dim, self.order)
        return tuple(
            tuple(self._entry((i, j), pos) for j in range(self.dim)) for i in range(self.dim)
        )

    def tensor(self, m: int) -> Any:
        r"""The symmetric ``m``-th derivative tensor of ``J`` at ``theta0`` (nested tuples of Interval).

        ``tensor(0)`` is the scalar ``J(theta0)``, ``tensor(1)`` the gradient, ``tensor(2)`` the
        Hessian, and so on up to ``m = order``.  Entry ``[i_1, ..., i_m]`` is the mixed partial
        ``d^m J / dtheta_{i_1} ... dtheta_{i_m}`` (an interval enclosure valid at ``theta0``).
        """
        if m < 0:
            raise ValueError(f"tensor order must be >= 0, got {m}")
        if m > self.order:
            raise ValueError(f"tensor order {m} exceeds the jet order {self.order}")
        pos = index_position(self.dim, self.order)

        def _build(prefix: tuple[int, ...], depth: int) -> Any:
            if depth == m:
                return self._entry(prefix, pos)
            return tuple(_build((*prefix, i), depth + 1) for i in range(self.dim))

        return _build((), 0)


def param_jet(
    arch: MLPArchitecture,
    data: Data,
    theta0: Sequence[float],
    *,
    order: int,
    radius: float,
    l2: float = 0.0,
    basis: Sequence[Sequence[float]] | None = None,
) -> ParamJet:
    r"""Build the order-``N`` parameter-space Taylor model of the training objective.

    Encloses ``J(theta0 + delta) = L(theta0 + delta) + l2 ||theta0 + delta||^2`` as an order-``order``
    :class:`TaylorModelMV` over a box:

    * **full parameter space** (``basis=None``): ``delta in [-radius, radius]^P`` -- every parameter
      gets its own Taylor variable, so one pass yields the whole gradient / Hessian / higher tensors;
    * **subspace** (``basis`` a ``P x k`` matrix): ``delta = Q a``, ``a in [-radius, radius]^k`` -- the
      map ``theta_p = theta0_p + sum_j basis[p][j] a_j`` is exact and affine, so the ``k`` shared
      subspace variables keep the enclosure tight (the mechanism :mod:`subspace_step` relies on).

    Returns a :class:`ParamJet`; read off derivatives with :meth:`ParamJet.grad` /
    :meth:`ParamJet.hessian` / :meth:`ParamJet.tensor` and the sound tail bound with
    :attr:`ParamJet.remainder`.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if l2 < 0.0:
        raise ValueError(f"l2 (weight decay) must be >= 0, got {l2}")
    theta = [float(t) for t in theta0]
    p_dim = arch.n_params
    if len(theta) != p_dim:
        raise ValueError(f"theta0 has {len(theta)} entries but the model has {p_dim} parameters")
    data_t = _normalize_data(arch, data)

    basis_out: tuple[tuple[float, ...], ...] | None
    if basis is None:
        dim = p_dim
        center = [0.0] * dim
        radius_vec = [float(radius)] * dim
        coords = [TaylorModelMV.coordinate(p, center, radius_vec, order) for p in range(dim)]
        theta_tms = [
            TaylorModelMV.constant(theta[p], center, radius_vec, order) + coords[p]
            for p in range(p_dim)
        ]
        basis_out = None
    else:
        if len(basis) != p_dim:
            raise ValueError(f"basis has {len(basis)} rows but the model has {p_dim} parameters")
        k = len(basis[0])
        if k < 1:
            raise ValueError("basis must have at least one column")
        for row in basis:
            if len(row) != k:
                raise ValueError("all basis rows must have the same length")
        dim = k
        center = [0.0] * dim
        radius_vec = [float(radius)] * dim
        coords = [TaylorModelMV.coordinate(j, center, radius_vec, order) for j in range(dim)]
        theta_tms = []
        for row_i, base in zip(basis, theta, strict=True):
            tp = TaylorModelMV.constant(base, center, radius_vec, order)
            for j in range(dim):
                q = float(row_i[j])
                if q != 0.0:
                    tp = tp + coords[j] * Interval.point(q)
            theta_tms.append(tp)
        basis_out = tuple(tuple(float(v) for v in row) for row in basis)

    psi = _objective_tm(arch, theta_tms, data_t, center, radius_vec, order, l2)
    return ParamJet(
        theta0=tuple(theta),
        order=order,
        dim=dim,
        radius=float(radius),
        basis=basis_out,
        tm=psi,
    )


__all__ = [
    "Data",
    "ParamJet",
    "param_jet",
]

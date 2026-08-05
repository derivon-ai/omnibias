# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Moment <-> cumulant combinatorics and the analytic delta method.

Cumulants and (raw / central) moments are linked by the exponential Bell
polynomials that already power Faa di Bruno in :mod:`omnibias.core.bell`:

.. math::

    m_n = B_n(\kappa_1, \dots, \kappa_n), \qquad
    \kappa_n = \sum_{k=1}^{n} (-1)^{k-1}(k-1)!\,B_{n,k}(m_1, \dots, m_{n-k+1}).

The pay-off is the *analytic delta method*: given the closed-form derivative
tower of a map ``f`` at the input mean (which omnibias produces exactly) and the
central moments of the input, :func:`delta_method_central_moments` returns the
output mean, variance and higher central moments in closed form -- a
deterministic replacement for Monte-Carlo uncertainty propagation, exact to the
chosen truncation order.

All routines here are pure Python and ring-generic: they use only ``+``, ``*``
and division by integers, so they evaluate identically on ``float``,
:class:`~fractions.Fraction`, or backend tensors (which is exactly how the
``omnibias.{torch,jax}.moments`` twins reuse :func:`delta_method_central_moments`
elementwise).

Conventions
-----------
Cumulant and moment sequences are **1-indexed** lists ``[order-1, order-2, ...]``
(no order-0 entry).  Central-moment sequences returned here keep
``mu_1 = 0`` in position 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb, factorial
from typing import Any

from omnibias.core.bell import bell_complete, bell_partial

# A ring element: ``float``, :class:`~fractions.Fraction`, or a backend tensor.
# The routines below use only ``+``, ``*``, ``**`` and integer division, so they
# are polymorphic over any such type; ``Any`` captures that honestly.
Ring = Any


# --------------------------------------------------------------------------- #
# Ring-generic truncated polynomial helpers (coeff[k] is the u**k coefficient).
# --------------------------------------------------------------------------- #
def _is_zero(x: object) -> bool:
    """Structural-zero test that is safe for array/tensor coefficients.

    Only the integer ``0`` placeholders introduced by this module are treated as
    zero; tensors and other numerics are never short-circuited (correct, since
    the skip is purely an optimisation).
    """
    return type(x) is int and x == 0


def _poly_mul(a: Sequence[Ring], b: Sequence[Ring], max_deg: int) -> list[Ring]:
    out: list[Ring] = [0] * (min(len(a) + len(b) - 2, max_deg) + 1)
    for i, ai in enumerate(a):
        if i > max_deg or _is_zero(ai):
            continue
        for j, bj in enumerate(b):
            if i + j > max_deg:
                break
            if _is_zero(bj):
                continue
            out[i + j] = out[i + j] + ai * bj
    return out


def _poly_pow(a: Sequence[Ring], p: int, max_deg: int) -> list[Ring]:
    result: list[Ring] = [1]
    base = list(a)
    e = p
    while e > 0:
        if e & 1:
            result = _poly_mul(result, base, max_deg)
        e >>= 1
        if e > 0:
            base = _poly_mul(base, base, max_deg)
    return result


# --------------------------------------------------------------------------- #
# Moment <-> cumulant identities (exact, arbitrary order).
# --------------------------------------------------------------------------- #
def raw_moments_from_cumulants(kappa: Sequence[Ring]) -> list[Ring]:
    r"""Raw moments ``[m_1, ..., m_N]`` from cumulants ``[kappa_1, ..., kappa_N]``.

    Uses ``m_n = B_n(kappa_1, ..., kappa_n)`` (complete Bell polynomial).
    """
    n_max = len(kappa)
    out: list[Ring] = []
    for n in range(1, n_max + 1):
        acc: Ring = 0
        for exps, coeff in bell_complete(n).items():
            term: Ring = coeff
            for i, e in enumerate(exps):
                if e:
                    term = term * (kappa[i] ** e)
            acc = acc + term
        out.append(acc)
    return out


def cumulants_from_raw_moments(moments: Sequence[Ring]) -> list[Ring]:
    r"""Cumulants ``[kappa_1, ..., kappa_N]`` from raw moments ``[m_1, ..., m_N]``."""
    n_max = len(moments)
    out: list[Ring] = []
    for n in range(1, n_max + 1):
        acc: Ring = 0
        for k in range(1, n + 1):
            sign = 1 if (k - 1) % 2 == 0 else -1
            weight = sign * factorial(k - 1)
            sub: Ring = 0
            for exps, coeff in bell_partial(n, k).items():
                term: Ring = coeff
                for i, e in enumerate(exps):
                    if e:
                        term = term * (moments[i] ** e)
                sub = sub + term
            acc = acc + weight * sub
        out.append(acc)
    return out


def central_moments_from_cumulants(kappa: Sequence[Ring]) -> list[Ring]:
    r"""Central moments ``[mu_1=0, mu_2, ..., mu_N]`` from cumulants.

    Central moments are the moments of ``X - E[X]``, whose cumulants are
    ``(0, kappa_2, kappa_3, ...)``; hence ``mu_n = B_n(0, kappa_2, ..., kappa_n)``.
    """
    if not kappa:
        return []
    zero = kappa[0] * 0
    shifted = [zero, *list(kappa[1:])]
    return raw_moments_from_cumulants(shifted)


def raw_to_central_moments(raw: Sequence[Ring], mean: Ring) -> list[Ring]:
    r"""Central moments from raw moments via the binomial shift by ``-mean``."""
    n_max = len(raw)
    full = [1, *list(raw)]  # index 0 = m_0 = 1
    out: list[Ring] = []
    for n in range(1, n_max + 1):
        acc: Ring = 0
        for j in range(n + 1):
            acc = acc + comb(n, j) * full[j] * ((-mean) ** (n - j))
        out.append(acc)
    return out


def central_to_raw_moments(central: Sequence[Ring], mean: Ring) -> list[Ring]:
    r"""Raw moments from central moments via the binomial shift by ``+mean``."""
    n_max = len(central)
    full = [1, *list(central)]  # index 0 = mu_0 = 1
    out: list[Ring] = []
    for n in range(1, n_max + 1):
        acc: Ring = 0
        for j in range(n + 1):
            acc = acc + comb(n, j) * full[j] * (mean ** (n - j))
        out.append(acc)
    return out


def gaussian_central_moments(variance: Ring, order: int) -> list[Ring]:
    r"""Central moments ``[mu_1=0, ..., mu_order]`` of ``N(mu, variance)``.

    ``mu_{2k} = (2k-1)!! * variance**k`` and odd central moments vanish.
    """
    out: list[Ring] = []
    for n in range(1, order + 1):
        if n % 2 == 1:
            out.append(variance * 0)
        else:
            half = n // 2
            df = factorial(n) // (2**half * factorial(half))  # (n-1)!!
            out.append(df * variance**half)
    return out


# --------------------------------------------------------------------------- #
# Analytic delta method (univariate, arbitrary truncation order).
# --------------------------------------------------------------------------- #
def delta_method_central_moments(
    derivatives: Sequence[Ring],
    central_in: Sequence[Ring],
    order: int,
) -> dict[str, Ring]:
    r"""Closed-form moments of ``Y = f(X)`` from ``f``'s derivative tower.

    Parameters
    ----------
    derivatives:
        ``[f(mu), f'(mu), ..., f^{(order)}(mu)]`` -- the exact derivative tower
        at the input mean ``mu`` (length must be at least ``order + 1``).
    central_in:
        Input central moments ``[mu_1=0, mu_2, ..., mu_order]`` (1-indexed;
        length must be at least ``order``).
    order:
        Truncation / output order (``>= 1``).

    Returns
    -------
    dict
        ``{"mean": E[Y], "variance": Var[Y], "central": [nu_2, ..., nu_order]}``.

    Notes
    -----
    Writes the fluctuation ``S = f(X) - f(mu) = sum_k a_k (X-mu)^k`` with
    ``a_k = f^{(k)}(mu)/k!``, expands ``(S - E[S])^p`` as a polynomial in
    ``u = X-mu`` truncated at degree ``order``, then takes expectations with the
    input central moments.  Exact for any input whose central moments are given,
    to the truncation order.  For a *linear* ``f`` it reproduces the input
    moments exactly (the higher ``a_k`` vanish).
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if len(derivatives) < order + 1:
        raise ValueError(f"need >= {order + 1} derivatives, got {len(derivatives)}")
    if len(central_in) < order:
        raise ValueError(f"need >= {order} central moments, got {len(central_in)}")

    a = [derivatives[k] / factorial(k) for k in range(order + 1)]
    c: list[Ring] = [1, *list(central_in[:order])]  # c[0]=1, c[1]=mu_1=0, ...

    fluct: list[Ring] = [0] * (order + 1)  # coeff of u**k in S
    for k in range(1, order + 1):
        fluct[k] = a[k]
    e_s: Ring = 0
    for k in range(order + 1):
        e_s = e_s + fluct[k] * c[k]
    mean = a[0] + e_s

    centered = list(fluct)
    centered[0] = fluct[0] - e_s
    central_out: list[Ring] = []
    for p in range(2, order + 1):
        poly = _poly_pow(centered, p, order)
        nu: Ring = 0
        for j in range(min(len(poly), order + 1)):
            nu = nu + poly[j] * c[j]
        central_out.append(nu)

    variance: Ring = central_out[0] if central_out else derivatives[0] * 0
    return {"mean": mean, "variance": variance, "central": central_out}


def delta_method_from_cumulants(
    derivatives: Sequence[Ring],
    input_cumulants: Sequence[Ring],
    order: int,
) -> dict[str, Ring]:
    """:func:`delta_method_central_moments` fed by input *cumulants* instead.

    The input cumulants ``[kappa_1, ..., kappa_M]`` are converted to central
    moments first; ``kappa_1`` (the mean) is ignored by the central form.
    """
    central = central_moments_from_cumulants(input_cumulants)
    return delta_method_central_moments(derivatives, central, order)


# --------------------------------------------------------------------------- #
# Multivariate second-order delta method (reference; tensors handled in backends).
# --------------------------------------------------------------------------- #
def second_order_delta(
    value: Sequence[Ring],
    jacobian: Sequence[Sequence[Ring]],
    hessian: Sequence[Sequence[Sequence[Ring]]] | None,
    cov: Sequence[Sequence[Ring]],
) -> tuple[list[Ring], list[list[Ring]]]:
    r"""Second-order Gaussian delta method for a vector map ``f: R^D -> R^C``.

    ``jacobian[i][c] = d f_c / d x_i`` and ``hessian[i][j][c] = d^2 f_c / dx_i dx_j``
    at the mean; ``cov`` is the ``D x D`` input covariance.  Returns
    ``(out_mean, out_cov)`` with

    .. math::

        E[Y_c] \approx f_c + \tfrac12 \sum_{ij} H^c_{ij}\,\Sigma_{ij}, \qquad
        \operatorname{Cov}[Y_c, Y_d] \approx \sum_{ij} J^c_i \Sigma_{ij} J^d_j.
    """
    dim = len(jacobian)
    out_dim = len(value)
    out_mean: list[Ring] = list(value)
    if hessian is not None:
        for c in range(out_dim):
            corr: Ring = 0
            for i in range(dim):
                for j in range(dim):
                    corr = corr + hessian[i][j][c] * cov[i][j]
            out_mean[c] = value[c] + corr * 0.5
    out_cov: list[list[Ring]] = []
    for c in range(out_dim):
        row: list[Ring] = []
        for d in range(out_dim):
            acc: Ring = 0
            for i in range(dim):
                for j in range(dim):
                    acc = acc + jacobian[i][c] * cov[i][j] * jacobian[j][d]
            row.append(acc)
        out_cov.append(row)
    return out_mean, out_cov


__all__ = [
    "central_moments_from_cumulants",
    "central_to_raw_moments",
    "cumulants_from_raw_moments",
    "delta_method_central_moments",
    "delta_method_from_cumulants",
    "gaussian_central_moments",
    "raw_moments_from_cumulants",
    "raw_to_central_moments",
    "second_order_delta",
]

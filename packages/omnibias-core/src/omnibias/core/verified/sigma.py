# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous interval enclosure of the closed-form derivative tower.

This is the verified twin of omnibias's float ``fastpath`` kernels: it returns a
guaranteed enclosure of ``sigma^(k)(z)`` for ``k = 0 .. order`` given an interval
enclosure of the argument ``z``.  The same closed-form structure used everywhere
in omnibias is reused -- a single transcendental evaluation
(:func:`omnibias.core.verified.transcend`) followed by exact-integer polynomial
evaluation (:func:`omnibias.core.verified.coeffs.horner_interval`) -- so the
enclosure is tight and needs only **one** transcendental enclosure regardless of
the derivative order.

Supported activations: ``"tanh"``, ``"sigmoid"``, ``"gaussian"``
(``g(z) = exp(-z^2 / 2)``), the trigonometric pair ``"sin"`` / ``"cos"``, the
smooth closed-form neural activations ``"silu"`` (``z * sigmoid(z)``), ``"gelu"``
(exact ``z * Phi(z)``), and ``"softplus"`` (``ln(1 + e^z)``), and ``"sech"``
(``1 / cosh(z)``, the secant / Euler-number tower ``sech^(n) = Q_n(tanh) sech``).
The neural three reuse the sigmoid / Gaussian towers via the product / shift
identities (see the helper docstrings below), so they need no new polynomial
recurrence.

The trigonometric tower is closed form to *every* order with **no** polynomial
recurrence: ``cos`` and ``sin`` are their own fourth derivatives, so the whole
tower is the 4-cycle of phase shifts ``(cos, -sin, -cos, sin)`` /
``(sin, cos, -sin, -cos)`` built from two interval enclosures
(:func:`~omnibias.core.verified.transcend.cos_iv` and ``sin_iv``) regardless of
``order``.  This admits classical Fourier-mode / plane-wave exact solutions to the
certified jet.
"""

from __future__ import annotations

from omnibias.core.verified.coeffs import (
    hermite_coeffs_exact,
    horner_interval,
    sech_poly_coeffs_exact,
    sigmoid_poly_coeffs_exact,
    tanh_poly_coeffs_exact,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import (
    _INV_SQRT_2PI,
    cos_iv,
    exp_iv,
    gauss_cdf_iv,
    sech_iv,
    sigmoid_iv,
    sin_iv,
    softplus_iv,
    tanh_iv,
)

_SUPPORTED = (
    "tanh",
    "sigmoid",
    "gaussian",
    "sin",
    "cos",
    "silu",
    "gelu",
    "softplus",
    "sech",
)


def sigma_tower_interval(name: str, z: Interval, order: int) -> tuple[Interval, ...]:
    """Enclose ``(sigma(z), sigma'(z), ..., sigma^(order)(z))`` over interval ``z``.

    Parameters
    ----------
    name
        One of ``"tanh"``, ``"sigmoid"``, ``"gaussian"``, ``"sin"``, ``"cos"``,
        ``"silu"``, ``"gelu"`` (exact), ``"softplus"``, or ``"sech"``.
    z
        Interval enclosure of the scalar argument.
    order
        Highest derivative order ``N``; the tower has ``N + 1`` entries.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if name == "tanh":
        t = tanh_iv(z)
        return tuple(
            horner_interval(tanh_poly_coeffs_exact(k), t) for k in range(order + 1)
        )
    if name == "sech":
        t = tanh_iv(z)
        sech = sech_iv(z)
        return tuple(
            horner_interval(sech_poly_coeffs_exact(k), t) * sech for k in range(order + 1)
        )
    if name == "sigmoid":
        return tuple(_sigmoid_tower(z, order))
    if name == "silu":
        return _silu_tower(z, order)
    if name == "softplus":
        return _softplus_tower(z, order)
    if name == "gelu":
        return _gelu_tower(z, order)
    if name == "gaussian":
        return _gaussian_tower(z, order)
    if name in ("sin", "cos"):
        return _trig_tower(name, z, order)
    raise ValueError(f"unsupported activation {name!r}; expected one of {_SUPPORTED}")


def _trig_tower(name: str, z: Interval, order: int) -> tuple[Interval, ...]:
    r"""Tower for ``cos``/``sin`` via the 4-cycle ``f^(k)(z) = f(z + k pi/2)``.

    ``cos^(k)`` cycles through ``(cos, -sin, -cos, sin)`` and ``sin^(k)`` through
    ``(sin, cos, -sin, -cos)``; both reuse the same two interval enclosures
    ``cos_iv(z)`` and ``sin_iv(z)``, so the cost is independent of ``order``.
    """
    c = cos_iv(z)
    s = sin_iv(z)
    cycle = (c, -s, -c, s) if name == "cos" else (s, c, -s, -c)
    return tuple(cycle[k % 4] for k in range(order + 1))


def _sigmoid_tower(z: Interval, order: int) -> list[Interval]:
    """``[sigmoid(z), sigmoid'(z), ..., sigmoid^(order)(z)]`` via the exact ``s``-polynomials."""
    s = sigmoid_iv(z)
    return [horner_interval(sigmoid_poly_coeffs_exact(k), s) for k in range(order + 1)]


def _silu_tower(z: Interval, order: int) -> tuple[Interval, ...]:
    r"""Tower for ``silu(z) = z * sigmoid(z)`` via the Leibniz rule.

    ``silu^(k) = z * sigmoid^(k)(z) + k * sigmoid^(k-1)(z)`` (product rule of
    ``z`` with the sigmoid tower).
    """
    s = _sigmoid_tower(z, order)
    rows: list[Interval] = []
    for k in range(order + 1):
        term = z * s[k]
        if k >= 1:
            term = term + Interval.point(float(k)) * s[k - 1]
        rows.append(term)
    return tuple(rows)


def _softplus_tower(z: Interval, order: int) -> tuple[Interval, ...]:
    r"""Tower for ``softplus(z) = ln(1 + e^z)``.

    The value uses the stable :func:`softplus_iv` enclosure; every higher order is a
    shift of the sigmoid tower, since ``softplus'(z) = sigmoid(z)`` and hence
    ``softplus^(k)(z) = sigmoid^(k-1)(z)`` for ``k >= 1``.
    """
    rows: list[Interval] = [softplus_iv(z)]
    if order >= 1:
        s = _sigmoid_tower(z, order - 1)  # s[j] = sigmoid^(j)(z)
        rows.extend(s[k - 1] for k in range(1, order + 1))
    return tuple(rows)


def _gelu_tower(z: Interval, order: int) -> tuple[Interval, ...]:
    r"""Tower for exact ``gelu(z) = z * Phi(z)`` (``Phi`` the standard-normal CDF).

    ``Phi^(0) = gauss_cdf(z)`` and, for ``k >= 1``, ``Phi^(k) = phi^(k-1)`` with the
    standard-normal density ``phi(z) = (1/sqrt(2 pi)) g(z)`` (``g(z) = exp(-z^2/2)``);
    so ``Phi^(k) = INV_SQRT_2PI * g^(k-1)`` reuses the Gaussian tower.  The Leibniz
    rule then gives ``gelu^(k) = z * Phi^(k) + k * Phi^(k-1)``.
    """
    phi: list[Interval] = [gauss_cdf_iv(z)]
    if order >= 1:
        g = _gaussian_tower(z, order - 1)  # g[j] = g^(j)(z), j = 0 .. order-1
        phi.extend(_INV_SQRT_2PI * g[k - 1] for k in range(1, order + 1))
    rows: list[Interval] = []
    for k in range(order + 1):
        term = z * phi[k]
        if k >= 1:
            term = term + Interval.point(float(k)) * phi[k - 1]
        rows.append(term)
    return tuple(rows)


def _gaussian_tower(z: Interval, order: int) -> tuple[Interval, ...]:
    r"""Tower for ``g(z) = exp(-z^2/2)``: ``g^(n)(z) = (-1)^n He_n(z) g(z)``."""
    half_z2 = z.pow_int(2) * Interval.point(0.5)
    g = exp_iv(-half_z2)
    rows: list[Interval] = []
    sign = 1
    for k in range(order + 1):
        he = horner_interval(hermite_coeffs_exact(k), z)
        term = he * g
        rows.append(term if sign == 1 else -term)
        sign = -sign
    return tuple(rows)


def sigma_value_interval(name: str, z: Interval) -> Interval:
    """Enclosure of ``sigma(z)`` (the zeroth tower entry)."""
    return sigma_tower_interval(name, z, 0)[0]


__all__ = [
    "sigma_tower_interval",
    "sigma_value_interval",
]

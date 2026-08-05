# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Leading asymptotics of the analytic-combinatorics coefficients (pure ``math``).

Closed-form leading asymptotics for the numbers extracted off the towers. Each is
the genuine leading term (relative error ``-> 0`` as the index grows); the exact
values and a high-precision ``mpmath`` cross-check live in the tests.

* **Bernoulli** ``B_{2m} ~ (-1)^{m+1} 2 (2m)! / (2 pi)^{2m}`` (fast: relative
  error ``O(2^{-2m})``).
* **Euler (secant)** ``E_{2m} ~ (-1)^m 2^{2m+2} (2m)! / pi^{2m+1}`` (fast).
* **Stirling second kind, fixed ``k``** ``S(n, k) ~ k^n / k!`` as ``n -> inf``.
* **Bell numbers** -- the Dobinski saddle-point form
  ``B_n ~ e^{n/r - 1} n! / (r^n sqrt(2 pi n (r+1)))`` with ``r = W(n)`` (Lambert
  W); relative error ``~1%`` by ``n = 30`` and shrinking.

Values are returned as ``float`` and are computed in log space where possible, so
they stay finite far past the point where ``(2m)!`` would overflow a double; the
Bell helper also exposes its logarithm for very large ``n``.
"""

from __future__ import annotations

from math import exp, lgamma, log, pi


def bernoulli_asymptotic(n: int) -> float:
    r"""Signed leading asymptotic of ``B_n`` (even ``n = 2m >= 2`` only).

    ``B_{2m} ~ (-1)^{m+1} 2 (2m)! / (2 pi)^{2m}``.
    """
    if n < 2 or n % 2 == 1:
        raise ValueError(f"Bernoulli asymptotic is defined for even n >= 2, got {n}")
    m = n // 2
    log_mag = log(2.0) + lgamma(n + 1) - n * log(2.0 * pi)
    return (-1.0) ** (m + 1) * exp(log_mag)


def euler_asymptotic(n: int) -> float:
    r"""Signed leading asymptotic of the Euler (secant) number ``E_n`` (even ``n = 2m >= 2``).

    ``E_{2m} ~ (-1)^m 2^{2m+2} (2m)! / pi^{2m+1}``.
    """
    if n < 2 or n % 2 == 1:
        raise ValueError(f"Euler asymptotic is defined for even n >= 2, got {n}")
    m = n // 2
    log_mag = (2 * m + 2) * log(2.0) + lgamma(n + 1) - (2 * m + 1) * log(pi)
    return (-1.0) ** m * exp(log_mag)


def stirling_second_asymptotic(n: int, k: int) -> float:
    r"""Fixed-``k`` leading asymptotic ``S(n, k) ~ k^n / k!`` as ``n -> inf``."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return exp(n * log(k) - lgamma(k + 1))


def _lambert_w(x: float) -> float:
    """Principal real branch ``W(x)`` for ``x >= 0`` (Halley iteration)."""
    if x < 0.0:
        raise ValueError(f"_lambert_w supports x >= 0 only, got {x}")
    if x == 0.0:
        return 0.0
    w = log(1.0 + x)  # globally safe start for x > 0
    for _ in range(100):
        ew = exp(w)
        f = w * ew - x
        denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0)
        w_next = w - f / denom
        if abs(w_next - w) <= 1e-16 * (1.0 + abs(w_next)):
            return w_next
        w = w_next
    return w


def log_bell_number_asymptotic(n: int) -> float:
    r"""Natural log of the Bell-number saddle-point asymptotic (safe for large ``n``)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    r = _lambert_w(float(n))
    return (n / r - 1.0) + lgamma(n + 1) - n * log(r) - 0.5 * log(2.0 * pi * n * (r + 1.0))


def bell_number_asymptotic(n: int) -> float:
    r"""Dobinski saddle-point asymptotic of the Bell number ``Bell(n)``.

    ``Bell(n) ~ e^{n/r - 1} n! / (r^n sqrt(2 pi n (r+1)))`` with ``r = W(n)``. Use
    :func:`log_bell_number_asymptotic` when ``n`` is large enough to overflow.
    """
    return exp(log_bell_number_asymptotic(n))


__all__ = [
    "bell_number_asymptotic",
    "bernoulli_asymptotic",
    "euler_asymptotic",
    "log_bell_number_asymptotic",
    "stirling_second_asymptotic",
]

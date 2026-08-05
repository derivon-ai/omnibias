# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-Python polynomial-coefficient generators for the closed-form
derivative towers (sigmoid, tanh, Gaussian).

The torch and JAX backends both import these. The coefficient
sequences are independent of the array library; only the Horner
evaluation has to differ across backends (because it touches the
tensor type).

Three families:

* sigmoid / softplus: ``sigma^(n)(z) = P_n(sigmoid(z))`` where
  ``P_n`` is a polynomial in ``s = sigmoid(z)`` whose coefficients
  are produced by :func:`sigmoid_polynomial_coeffs`.
* tanh: ``tanh^(n)(z) = T_n(tanh(z))`` where ``T_n`` is a polynomial
  in ``t = tanh(z)`` produced by :func:`tanh_polynomial_coeffs`.
* Gaussian ``g(z) = exp(-z^2/2)``: ``g^(n)(z) = (-1)^n He_n(z) g(z)``
  with probabilist's Hermite polynomial coefficients
  :func:`hermite_coeffs`.

These are the same definitions used by the PyTorch fast paths in
:mod:`omnibias.fastpath.eulerian`, :mod:`omnibias.fastpath.legendre`,
:mod:`omnibias.fastpath.hermite`. Keeping them in one pure-Python
file guarantees backend parity by construction.

Exactness
---------
Every one of these recurrences has **integer** coefficients (signed Eulerian
numbers, the tanh / sech Riccati towers, probabilist's Hermite). They are
therefore accumulated in Python ``int`` -- which is unbounded -- and narrowed to
``float`` exactly once, on return. Running the recurrence in ``float`` instead
would compound rounding at every step and stop matching the correctly-rounded
coefficient from order 19 (sigmoid), 20 (sech), 22 (tanh) and 30 (Hermite)
onward, because the coefficients outgrow ``2**53``. Narrowing once means each
returned coefficient carries a single rounding, the best a ``float`` can do, and
the exact ``int`` twins in :mod:`omnibias.core.verified.coeffs` agree with these
by construction rather than by luck.

The recurrences are also iterative rather than self-recursive, so a large order
costs memory but never a ``RecursionError``, and each cache is explicitly bounded
(order is often user-controlled, and an unbounded memo keyed on it is a memory
exhaustion vector).
"""

from __future__ import annotations

from functools import lru_cache

#: Largest derivative order these generators accept. Well past the point where a
#: ``float`` coefficient can hold the answer at all (see :func:`_narrow`), so the
#: binding limit in practice is representability rather than this cap; the cap is
#: here so a hostile or mistyped order cannot allocate unboundedly.
MAX_ORDER: int = 512

#: Bound on each memo. The tower is walked from low order upward, so a small
#: window captures effectively all reuse without pinning every order ever asked
#: for in memory. An unbounded ``@cache`` keyed on a user-supplied order is a
#: memory-exhaustion vector.
_CACHE_SIZE: int = 256


def _check_order(n: int) -> None:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    if n > MAX_ORDER:
        raise ValueError(
            f"order n must be <= MAX_ORDER ({MAX_ORDER}), got {n}; raise "
            "omnibias.core.polynomials.MAX_ORDER if you genuinely need a taller tower"
        )


def _narrow(coeffs: tuple[int, ...], n: int, family: str) -> tuple[float, ...]:
    """Round exact integer coefficients to ``float``, once, with a clear ceiling.

    These towers grow super-exponentially: the sigmoid coefficients leave the
    ``float64`` range at order 160, tanh and sech at 164, Hermite at 297. Past
    that a ``tuple[float, ...]`` cannot represent the answer at all, so report it
    as such instead of letting ``float()`` surface a bare ``OverflowError``.
    """
    try:
        return tuple(float(c) for c in coeffs)
    except OverflowError:
        raise ValueError(
            f"{family} coefficients at order {n} exceed the float64 range, so they "
            "cannot be returned as floats; use the exact integer twin in "
            "omnibias.core.verified.coeffs (or omnibias.core.verified for interval "
            "arithmetic) at this order"
        ) from None


# ---------------------------------------------------------------------------
# sigmoid / softplus -- Eulerian recursion
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_CACHE_SIZE)
def _sigmoid_coeffs_int(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of ``P_n``; see :func:`sigmoid_polynomial_coeffs`."""
    coeffs = [0, 1]  # P_0(s) = s
    for _ in range(n):
        # P_{k+1} = s (1 - s) P_k'  --  differentiate, then convolve with (0, 1, -1).
        deriv = [j * coeffs[j] for j in range(1, len(coeffs))]
        out = [0] * (len(deriv) + 2)
        for i, c in enumerate(deriv):
            out[i + 1] += c
            out[i + 2] -= c
        coeffs = out
    return tuple(coeffs)


@lru_cache(maxsize=_CACHE_SIZE)
def sigmoid_polynomial_coeffs(n: int) -> tuple[float, ...]:
    """Coefficients of ``P_n(s) = sigma^(n)(z)`` as a polynomial in
    ``s = sigmoid(z)``.

    Returns a tuple ``(c_0, c_1, ..., c_{n+1})`` with
    ``P_n(s) = sum_k c_k * s^k``. ``P_0(s) = s``; the recurrence is
    ``P_{n+1}(s) = s (1 - s) * P_n'(s)`` and the coefficients are
    (signed) Eulerian numbers.
    """
    _check_order(n)
    return _narrow(_sigmoid_coeffs_int(n), n, "sigmoid")


# ---------------------------------------------------------------------------
# tanh -- Legendre-style recursion
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_CACHE_SIZE)
def _tanh_coeffs_int(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of ``T_n``; see :func:`tanh_polynomial_coeffs`."""
    coeffs = [0, 1]  # T_0(t) = t
    for _ in range(n):
        # T_{k+1} = (1 - t^2) T_k'  --  differentiate, then convolve with (1, 0, -1).
        deriv = [j * coeffs[j] for j in range(1, len(coeffs))]
        out = [0] * (len(deriv) + 2)
        for i, c in enumerate(deriv):
            out[i] += c
            out[i + 2] -= c
        coeffs = out
    return tuple(coeffs)


@lru_cache(maxsize=_CACHE_SIZE)
def tanh_polynomial_coeffs(n: int) -> tuple[float, ...]:
    """Coefficients of ``T_n(t) = tanh^(n)(z)`` as a polynomial in
    ``t = tanh(z)``.

    Returns a tuple ``(c_0, ..., c_{n+1})`` with
    ``T_n(t) = sum_k c_k * t^k``. ``T_0(t) = t``; recurrence
    ``T_{n+1}(t) = (1 - t^2) * T_n'(t)``.
    """
    _check_order(n)
    return _narrow(_tanh_coeffs_int(n), n, "tanh")


# ---------------------------------------------------------------------------
# sech -- tanh/sech Riccati recursion (secant / Euler-number tower)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_CACHE_SIZE)
def _sech_coeffs_int(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of ``Q_n``; see :func:`sech_polynomial_coeffs`."""
    coeffs = [1]  # Q_0 = 1
    for _ in range(n):
        # Q_{k+1} = (1 - t^2) Q_k' - t Q_k.
        deriv = [j * coeffs[j] for j in range(1, len(coeffs))]
        out = [0] * (len(coeffs) + 1)
        for i, c in enumerate(deriv):
            out[i] += c
            out[i + 2] -= c
        for i, c in enumerate(coeffs):
            out[i + 1] -= c
        coeffs = out
    return tuple(coeffs)


@lru_cache(maxsize=_CACHE_SIZE)
def sech_polynomial_coeffs(n: int) -> tuple[float, ...]:
    r"""Coefficients of ``Q_n(t)`` with ``sech^(n)(z) = Q_n(t) * sech(z)``,
    ``t = tanh(z)``.

    Returns a tuple ``(c_0, ..., c_n)`` with ``Q_n(t) = sum_k c_k * t^k``.
    ``sech`` closes its whole derivative tower on ``poly(t) * sech`` via the two
    Riccati rules ``d/dz sech = -t sech`` and ``d/dz t = 1 - t^2``, giving the
    recurrence ``Q_0 = 1``, ``Q_{n+1}(t) = (1 - t^2) Q_n'(t) - t Q_n(t)``.

    The constant term ``Q_n(0) = sech^(n)(0)`` is the ``n``-th **Euler (secant)
    number** ``E_n`` (``E_0 = 1``, ``E_2 = -1``, ``E_4 = 5``, ``E_6 = -61``, ...);
    :mod:`omnibias.difference` reads them straight off this tower. This is a
    coefficient generator (and its verified twin in
    :mod:`omnibias.core.verified`); no per-backend ``sech`` activation fastpath is
    provided.
    """
    _check_order(n)
    return _narrow(_sech_coeffs_int(n), n, "sech")


# ---------------------------------------------------------------------------
# Gaussian -- probabilist's Hermite polynomials
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_CACHE_SIZE)
def _hermite_coeffs_int(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of ``He_n``; see :func:`hermite_coeffs`."""
    if n == 0:
        return (1,)
    prev2: tuple[int, ...] = (1,)
    prev1: tuple[int, ...] = (0, 1)
    for m in range(2, n + 1):
        out = [0] * (m + 1)
        for k, c in enumerate(prev1):
            out[k + 1] += c
        for k, c in enumerate(prev2):
            out[k] -= (m - 1) * c
        prev2, prev1 = prev1, tuple(out)
    return prev1


@lru_cache(maxsize=_CACHE_SIZE)
def hermite_coeffs(n: int) -> tuple[float, ...]:
    """Coefficients of the probabilist's Hermite polynomial ``He_n``.

    Returns ``(c_0, ..., c_n)`` with ``He_n(z) = sum_k c_k * z^k``.
    Recurrence: ``He_0 = 1, He_1 = z, He_{n+1} = z He_n - n He_{n-1}``.
    """
    _check_order(n)
    return _narrow(_hermite_coeffs_int(n), n, "Hermite")


# ---------------------------------------------------------------------------
# Mish inner factor g(z) = tanh(softplus(z)) -- two-variable (t, s) recursion
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_CACHE_SIZE)
def mish_inner_coeffs(n: int) -> tuple[tuple[int, int, float], ...]:
    r"""Coefficients of the ``n``-th derivative of ``g(z) = tanh(softplus(z))``.

    ``g`` is the inner factor of Mish (``mish(z) = z * g(z)``). With
    ``t = tanh(softplus(z))`` and ``s = sigmoid(z) = softplus'(z)`` the two
    Riccati rules

        dt/dz = (1 - t^2) s,      ds/dz = s - s^2

    close the whole derivative tower on polynomials in ``(t, s)``:

        g^(n)(z) = sum_{i,j} c_{ij} t^i s^j.

    Returns the nonzero ``(i, j, c_{ij})`` triples, sorted. ``g^(0) = t`` so
    ``mish_inner_coeffs(0) == ((1, 0, 1.0),)``. Backends evaluate the tower with
    one ``softplus`` + one ``tanh`` + one ``sigmoid`` call, giving Mish a
    closed-form all-orders fast path that is bit-identical by construction.
    """
    _check_order(n)
    poly: dict[tuple[int, int], int] = {(1, 0): 1}
    for _ in range(n):
        nxt: dict[tuple[int, int], int] = {}
        for (i, j), c in poly.items():
            if i >= 1:  # d/dt * (1 - t^2) s
                nxt[(i - 1, j + 1)] = nxt.get((i - 1, j + 1), 0) + i * c
                nxt[(i + 1, j + 1)] = nxt.get((i + 1, j + 1), 0) - i * c
            if j >= 1:  # d/ds * (s - s^2)
                nxt[(i, j)] = nxt.get((i, j), 0) + j * c
                nxt[(i, j + 1)] = nxt.get((i, j + 1), 0) - j * c
        poly = nxt
    return tuple(sorted((i, j, float(c)) for (i, j), c in poly.items() if c != 0))


__all__ = [
    "MAX_ORDER",
    "hermite_coeffs",
    "mish_inner_coeffs",
    "sech_polynomial_coeffs",
    "sigmoid_polynomial_coeffs",
    "tanh_polynomial_coeffs",
]

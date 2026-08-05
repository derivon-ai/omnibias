# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared high-precision reference oracles + soundness assertions for tests.

A rigorous enclosure produced by ``omnibias.core.verified`` is only trustworthy
if an *independent* high-precision computation agrees that the true value lies
inside it.  This helper wraps :mod:`mpmath` (always available as a dev/oracle
dependency) and, when installed, ``python-flint``'s ball arithmetic, and exposes
two soundness assertions used across the suite:

* :func:`assert_encloses` -- the enclosure brackets a known value;
* :func:`assert_superset` -- the enclosure is a superset of a tight oracle
  enclosure (hence provably contains the same true real).

The mpmath oracles return a 1-ulp ``Interval`` that is *guaranteed* to bracket
the true mathematical value: they evaluate at ~200 bits (far beyond ``double``)
and then round the result outward to the two nearest doubles.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import mpmath
from omnibias.core.verified.interval import Interval

try:  # optional, rigorous ball arithmetic
    import flint  # type: ignore[import-not-found]

    HAVE_FLINT = True
except ImportError:  # pragma: no cover - exercised only when flint is absent
    HAVE_FLINT = False

HAVE_MPMATH = True

_DEFAULT_PREC = 200  # bits, >> 53


def mp_value(
    func: Callable[[mpmath.mpf], mpmath.mpf],
    x: float,
    *,
    prec: int = _DEFAULT_PREC,
) -> mpmath.mpf:
    """The ~``prec``-bit value ``func(x)`` (treated as ground truth in tests)."""
    with mpmath.workprec(prec):
        return func(mpmath.mpf(x))


def mp_enclosure(
    func: Callable[[mpmath.mpf], mpmath.mpf],
    x: float,
    *,
    prec: int = _DEFAULT_PREC,
) -> Interval:
    """Rigorous ``double`` enclosure of the true value ``func(x)``.

    ``func`` is evaluated at the *exact* double ``x`` in ~``prec`` bits, then the
    result is rounded outward to the two nearest doubles, so the returned
    interval provably contains the true mathematical value.
    """
    with mpmath.workprec(prec):
        val = func(mpmath.mpf(x))
        fd = float(val)
        mfd = mpmath.mpf(fd)
        lo = fd if mfd <= val else math.nextafter(fd, -math.inf)
        hi = fd if mfd >= val else math.nextafter(fd, math.inf)
    return Interval(lo, hi)


def oracle_exp(x: float) -> Interval:
    return mp_enclosure(mpmath.exp, x)


def oracle_tanh(x: float) -> Interval:
    return mp_enclosure(mpmath.tanh, x)


def oracle_sigmoid(x: float) -> Interval:
    return mp_enclosure(lambda z: 1 / (1 + mpmath.e ** (-z)), x)


def oracle_log(x: float) -> Interval:
    return mp_enclosure(mpmath.log, x)


def oracle_sqrt(x: float) -> Interval:
    return mp_enclosure(mpmath.sqrt, x)


def flint_enclosure(op: str, x: float) -> Interval | None:
    """Rigorous enclosure via ``python-flint`` ball arithmetic, or ``None``.

    Best-effort: returns ``None`` when flint is unavailable or its API for the
    requested op does not expose interval endpoints.
    """
    if not HAVE_FLINT:  # pragma: no cover - depends on optional dependency
        return None
    try:  # pragma: no cover - only runs where flint is installed
        ball = flint.arb(x)
        method = getattr(ball, op)
        result = method()
        lo = float(result.lower())
        hi = float(result.upper())
        return Interval(min(lo, hi), max(lo, hi))
    except Exception:  # pragma: no cover - defensive against API drift
        return None


def assert_encloses(
    enclosure: Interval,
    value: float | Fraction | mpmath.mpf,
    *,
    name: str = "",
) -> None:
    """Assert ``value`` lies within ``enclosure`` (rigorous containment)."""
    label = f" ({name})" if name else ""
    if isinstance(value, Fraction):
        assert Fraction(enclosure.lo) <= value <= Fraction(enclosure.hi), (
            f"{value} not in [{enclosure.lo}, {enclosure.hi}]{label}"
        )
    elif isinstance(value, mpmath.mpf):
        # mpmath compares an mpf to a float exactly (the float is promoted).
        assert enclosure.lo <= value <= enclosure.hi, (
            f"{value} not in [{enclosure.lo!r}, {enclosure.hi!r}]{label}"
        )
    else:
        v = float(value)
        assert enclosure.lo <= v <= enclosure.hi, (
            f"{v!r} not in [{enclosure.lo!r}, {enclosure.hi!r}]{label}"
        )


def assert_superset(enclosure: Interval, oracle: Interval, *, name: str = "") -> None:
    """Assert ``enclosure`` is a superset of the tight ``oracle`` enclosure.

    Since the oracle provably contains the true value, a superset relation
    proves ``enclosure`` does too -- the standard differential soundness check.
    """
    label = f" ({name})" if name else ""
    assert enclosure.lo <= oracle.lo and oracle.hi <= enclosure.hi, (
        f"[{enclosure.lo!r}, {enclosure.hi!r}] does not contain oracle "
        f"[{oracle.lo!r}, {oracle.hi!r}]{label}"
    )


__all__ = [
    "HAVE_FLINT",
    "HAVE_MPMATH",
    "assert_encloses",
    "assert_superset",
    "flint_enclosure",
    "mp_enclosure",
    "mp_value",
    "oracle_exp",
    "oracle_log",
    "oracle_sigmoid",
    "oracle_sqrt",
    "oracle_tanh",
]

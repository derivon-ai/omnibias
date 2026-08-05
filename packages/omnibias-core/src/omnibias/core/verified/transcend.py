# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous interval enclosures of the transcendental base functions.

The interval *algebra* in :mod:`omnibias.core.verified.interval` is exact up to
outward directed rounding, but it cannot by itself bound a transcendental such
as :math:`\tanh`.  This module supplies guaranteed enclosures of ``exp``,
``tanh`` and ``sigmoid`` at interval arguments.

Two backends, selected automatically:

* **mpmath (preferred).**  When ``mpmath`` is importable the base point is
  evaluated at high working precision and rounded *outward* to a double bracket.
  Because the high-precision error (~1e-60) is astronomically smaller than a
  double ulp, the resulting bracket rigorously contains the true value.  This is
  the ``mpfr``-class backend named in the certified-evidence contract.
* **stdlib fallback.**  Without mpmath the value is taken from the platform
  ``math`` libm and inflated by :data:`FALLBACK_ULPS` representable steps in each
  direction.  This is rigorous *provided* the libm error is below
  :data:`FALLBACK_ULPS` ulp -- true for every mainstream correctly-rounded-ish
  libm (glibc bounds ``tanh``/``exp`` well under 2 ulp).  :func:`backend_name`
  reports ``"libm_fallback"`` on this path so a certificate can record it
  honestly.  Certificate sealing that asserts unconditional soundness (or runs
  under :func:`certificate_mode` / :func:`set_strict_backend`) *refuses* this
  path rather than silently accepting a conditional enclosure.

``mpmath`` is intentionally an *optional* dependency: it is imported lazily via
:mod:`importlib` so :mod:`omnibias.core` keeps its zero-dependency contract and
the strict type gate never tries to resolve an un-stubbed module.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from omnibias.core.verified.interval import Interval, _pred, _succ

#: ulp inflation applied to each endpoint in the stdlib fallback.
FALLBACK_ULPS: int = 4

#: high-precision working decimal digits for the mpmath backend.
MPMATH_DPS: int = 60

#: Stamp value recorded when the high-precision backend is active.
BACKEND_MPMATH: str = "mpmath"
#: Stamp value recorded when the conditionally-rigorous stdlib path is active.
#: Named ``libm_fallback`` (not bare ``libm``) so a sealed certificate cannot be
#: misread as claiming a fully verified libm.
BACKEND_LIBM_FALLBACK: str = "libm_fallback"

_MPMATH: Any | None = None
_MPMATH_RESOLVED: bool = False

#: When ``True`` the libm fallback in :func:`_enclose_point` is *refused* (it
#: raises) instead of returning the :data:`FALLBACK_ULPS`-inflated bracket, whose
#: rigour is only *conditional* on the platform libm error budget.  Off by default
#: so the optional-mpmath contract is preserved; turn it on (or call
#: :func:`require_rigorous_backend` / :func:`certificate_mode`) before sealing a
#: certificate whose validity must be *unconditionally* sound.
_STRICT_BACKEND: bool = False

#: Sticky flag set the first time the libm fallback path in :func:`_enclose_point`
#: actually runs.  Certificate sealing reads it so a payload built under the
#: conditional backend cannot be sealed as if it were unconditionally rigorous.
_LIBM_FALLBACK_USED: bool = False


def _mpmath() -> Any | None:
    """Return the ``mpmath`` module if installed, else ``None`` (cached)."""
    global _MPMATH, _MPMATH_RESOLVED
    if not _MPMATH_RESOLVED:
        try:
            _MPMATH = importlib.import_module("mpmath")
        except ImportError:  # pragma: no cover - environment dependent
            _MPMATH = None
        _MPMATH_RESOLVED = True
    return _MPMATH


def backend_name() -> str:
    """``"mpmath"`` when the high-precision backend is active, else ``"libm_fallback"``."""
    return BACKEND_MPMATH if _mpmath() is not None else BACKEND_LIBM_FALLBACK


def set_strict_backend(enabled: bool) -> bool:
    """Enable/disable strict (mpmath-only) mode; returns the previous setting.

    In strict mode any transcendental evaluation that would fall back to the
    platform ``libm`` raises :class:`RuntimeError` instead of returning a
    *conditionally*-rigorous bracket.  Use it to guarantee a produced enclosure is
    unconditionally sound (e.g. before sealing a spectral-gap certificate).
    """
    global _STRICT_BACKEND
    previous = _STRICT_BACKEND
    _STRICT_BACKEND = bool(enabled)
    return previous


def strict_backend() -> bool:
    """``True`` when strict (mpmath-only) transcendental mode is active."""
    return _STRICT_BACKEND


def libm_fallback_used() -> bool:
    """``True`` if the libm fallback has produced at least one enclosure this process."""
    return _LIBM_FALLBACK_USED


def clear_libm_fallback_used() -> None:
    """Reset the :func:`libm_fallback_used` sticky flag (tests / fresh certificate scopes)."""
    global _LIBM_FALLBACK_USED
    _LIBM_FALLBACK_USED = False


def require_rigorous_backend() -> None:
    """Raise :class:`RuntimeError` unless the rigorous (mpmath) backend is active.

    The stdlib ``libm`` fallback is only *conditionally* rigorous -- it assumes the
    platform ``math`` error is below :data:`FALLBACK_ULPS` ulp, which the language
    does **not** guarantee.  Call this before asserting that a certificate whose
    proof chain touches a transcendental (spectral gaps via :func:`ln_iv`, Hilbert
    kernels, ...) is unconditionally sound.  :func:`make_certificate` also calls
    this automatically under :func:`certificate_mode` / strict mode / an
    ``unconditional_transcendentals`` honesty claim.
    """
    if backend_name() != BACKEND_MPMATH:
        raise RuntimeError(
            "rigorous transcendental backend (mpmath) required but unavailable: "
            f"the libm fallback is only conditionally rigorous (assumes < "
            f"{FALLBACK_ULPS} ulp libm error). Install mpmath to seal an "
            "unconditionally rigorous certificate."
        )


@contextmanager
def certificate_mode() -> Iterator[None]:
    """Context that makes the rigorous (mpmath) backend mandatory.

    Enter this (or call :func:`set_strict_backend` ``True``) before computing any
    transcendental enclosure that will feed a sealed certificate.  Inside the
    context the libm fallback raises, and :func:`make_certificate` refuses to
    seal unless :func:`backend_name` is :data:`BACKEND_MPMATH`.
    """
    require_rigorous_backend()
    previous = set_strict_backend(True)
    clear_libm_fallback_used()
    try:
        yield
    finally:
        set_strict_backend(previous)


def _inflate(value: float, ulps: int) -> tuple[float, float]:
    lo = value
    hi = value
    for _ in range(ulps):
        lo = _pred(lo)
        hi = _succ(hi)
    return lo, hi


def _bracket_mpf(mp: Any, y: Any) -> tuple[float, float]:
    """Outward double bracket of a high-precision value ``y`` (+1 ulp safety)."""
    f = float(y)
    yf = mp.mpf(f)
    lo = f
    hi = f
    if yf > y:
        lo = _pred(f)
    elif yf < y:
        hi = _succ(f)
    # one extra safety ulp absorbs the (negligible) mpmath truncation error.
    return _pred(lo), _succ(hi)


def _enclose_point(name: str, x: float) -> tuple[float, float]:
    """Rigorous double bracket of ``fn(x)`` for a scalar double ``x``."""
    mp = _mpmath()
    if mp is not None:
        with mp.workdps(MPMATH_DPS):
            arg = mp.mpf(x)
            if name == "exp":
                y = mp.e**arg
            elif name == "tanh":
                y = mp.tanh(arg)
            elif name == "sigmoid":
                y = mp.mpf(1) / (mp.mpf(1) + mp.e ** (-arg))
            elif name == "cos":
                y = mp.cos(arg)
            elif name == "sin":
                y = mp.sin(arg)
            elif name == "atan":
                y = mp.atan(arg)
            elif name == "log":
                y = mp.log(arg)
            elif name == "erf":
                y = mp.erf(arg)
            elif name == "gauss_cdf":
                y = (mp.mpf(1) + mp.erf(arg / mp.sqrt(2))) / mp.mpf(2)
            else:  # pragma: no cover - guarded by callers
                raise ValueError(f"unknown transcendental {name!r}")
        return _bracket_mpf(mp, y)
    if _STRICT_BACKEND:
        raise RuntimeError(
            "transcend strict mode is on but mpmath is unavailable: refusing the "
            f"conditionally-rigorous libm fallback for {name!r}. Install mpmath or "
            "call set_strict_backend(False)."
        )
    global _LIBM_FALLBACK_USED
    _LIBM_FALLBACK_USED = True
    if name == "exp":
        v = math.exp(x)
    elif name == "tanh":
        v = math.tanh(x)
    elif name == "sigmoid":
        v = 1.0 / (1.0 + math.exp(-x))
    elif name == "cos":
        v = math.cos(x)
    elif name == "sin":
        v = math.sin(x)
    elif name == "atan":
        v = math.atan(x)
    elif name == "log":
        v = math.log(x)
    elif name == "erf":
        v = math.erf(x)
    elif name == "gauss_cdf":
        v = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown transcendental {name!r}")
    return _inflate(v, FALLBACK_ULPS)


def _monotone_increasing(name: str, x: Interval) -> tuple[float, float]:
    lo, _ = _enclose_point(name, x.lo)
    _, hi = _enclose_point(name, x.hi)
    return lo, hi


def exp_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of ``exp(x)`` over the interval ``x``."""
    lo, hi = _monotone_increasing("exp", x)
    return Interval(lo, hi)


def tanh_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of ``tanh(x)`` (clamped to the true range ``[-1, 1]``)."""
    lo, hi = _monotone_increasing("tanh", x)
    return Interval(max(lo, -1.0), min(hi, 1.0))


def sigmoid_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of ``sigmoid(x)`` (clamped to the true range ``[0, 1]``)."""
    lo, hi = _monotone_increasing("sigmoid", x)
    return Interval(max(lo, 0.0), min(hi, 1.0))


def cosh_iv(x: Interval) -> Interval:
    r"""Guaranteed enclosure of ``cosh(x) = (e^x + e^{-x}) / 2`` over ``x``.

    ``cosh`` is **not monotone** (it is even, with its minimum ``1`` at ``0``), so
    an endpoint-only bound would be unsound.  Composing the two monotone
    :func:`exp_iv` enclosures instead is rigorous for *any* interval -- including
    one that straddles zero -- because the sum of two positive enclosures encloses
    ``e^t + e^{-t}`` for every ``t`` in ``x``.  The result stays strictly positive
    (``cosh >= 1``), so :func:`sech_iv` may take its reciprocal safely.
    """
    return (exp_iv(x) + exp_iv(-x)) * Interval.point(0.5)


def sech_iv(x: Interval) -> Interval:
    r"""Guaranteed enclosure of ``sech(x) = 1 / cosh(x)`` over ``x`` (clamped to ``(0, 1]``)."""
    out = cosh_iv(x).reciprocal()
    return Interval(max(out.lo, 0.0), min(out.hi, 1.0))


def cos_point(x: float) -> Interval:
    """Guaranteed enclosure of ``cos(x)`` at a scalar point ``x``."""
    lo, hi = _enclose_point("cos", x)
    return Interval(max(lo, -1.0), min(hi, 1.0))


def sin_point(x: float) -> Interval:
    """Guaranteed enclosure of ``sin(x)`` at a scalar point ``x``."""
    lo, hi = _enclose_point("sin", x)
    return Interval(max(lo, -1.0), min(hi, 1.0))


#: A rigorous double bracket of ``pi``.  ``math.pi`` is the nearest double to the
#: true ``pi`` (within half a ulp), so widening it one representable step in each
#: direction provably straddles ``pi`` with margin.  Kept as plain stdlib (no
#: mpmath dependency) because soundness needs only a *valid* enclosure -- a wider
#: bracket merely makes the extremum test below conservatively report an extremum
#: slightly more often, which only ever widens the result to the exact +/-1.  This
#: is the public constant Euler-Maclaurin / zeta / asymptotic enclosures compose
#: with; ``_PI_IV`` is retained as a private alias for the trig extremum tests.
PI_IV = Interval(_pred(math.pi), _succ(math.pi))
_PI_IV = PI_IV


def _has_integer_with_residue(lo: float, hi: float, residue: int, modulus: int) -> bool:
    """Whether some integer ``m`` with ``m % modulus == residue`` lies in ``[lo, hi]``.

    Callers pass an *outward* enclosure ``[lo, hi]`` of the true set of extremum
    indices, so this test has **no false negatives**: whenever a real extremum sits
    inside the argument interval it is reported.  A (conservative) false positive
    only forces the corresponding bound to the exact value ``+/-1``, which is always
    a sound over-estimate of ``sin``/``cos``.
    """
    if lo > hi:
        return False
    m = math.ceil(lo)
    m += (residue - m) % modulus
    return m <= hi


def cos_iv(x: Interval) -> Interval:
    r"""Guaranteed enclosure of ``cos`` over the (possibly wide) interval ``x``.

    ``cos`` is **not monotone**, so endpoint values alone do not bound it.  Its
    maxima (``cos = +1``) sit at *even* multiples of ``pi`` and its minima
    (``cos = -1``) at *odd* multiples; away from those it is monotone.  The range
    is therefore the hull of the two endpoint brackets, widened to ``+1`` / ``-1``
    whenever an even / odd multiple of ``pi`` lies in ``x``.  Membership is tested
    on an outward enclosure of ``x / pi`` (via :data:`_PI_IV`), so an interior
    extremum is never missed and the enclosure stays rigorous.
    """
    ca = cos_point(x.lo)
    cb = cos_point(x.hi)
    lo = min(ca.lo, cb.lo)
    hi = max(ca.hi, cb.hi)
    lo_over = (Interval.point(x.lo) / _PI_IV).lo  # lower bound on x.lo / pi
    hi_over = (Interval.point(x.hi) / _PI_IV).hi  # upper bound on x.hi / pi
    if _has_integer_with_residue(lo_over, hi_over, 0, 2):  # even multiple -> max +1
        hi = 1.0
    if _has_integer_with_residue(lo_over, hi_over, 1, 2):  # odd multiple  -> min -1
        lo = -1.0
    return Interval(max(lo, -1.0), min(hi, 1.0))


def sin_iv(x: Interval) -> Interval:
    r"""Guaranteed enclosure of ``sin`` over the (possibly wide) interval ``x``.

    The twin of :func:`cos_iv`: ``sin`` extrema sit at *odd* multiples of
    ``pi/2`` -- writing such a point as ``m * pi/2`` the maxima (``sin = +1``) are
    the ``m == 1 (mod 4)`` cases and the minima (``sin = -1``) the
    ``m == 3 (mod 4)`` cases.  Membership is tested on an outward enclosure of
    ``2 x / pi``.
    """
    sa = sin_point(x.lo)
    sb = sin_point(x.hi)
    lo = min(sa.lo, sb.lo)
    hi = max(sa.hi, sb.hi)
    lo_over = (Interval.point(x.lo) * 2 / _PI_IV).lo  # lower bound on 2 x.lo / pi
    hi_over = (Interval.point(x.hi) * 2 / _PI_IV).hi  # upper bound on 2 x.hi / pi
    if _has_integer_with_residue(lo_over, hi_over, 1, 4):  # -> max +1
        hi = 1.0
    if _has_integer_with_residue(lo_over, hi_over, 3, 4):  # -> min -1
        lo = -1.0
    return Interval(max(lo, -1.0), min(hi, 1.0))


_HALF_PI_HI = 1.5707963267948966  # nextafter-safe upper bound on pi/2


def atan_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of ``atan(x)`` (monotone; clamped to ``(-pi/2, pi/2)``)."""
    lo, hi = _monotone_increasing("atan", x)
    return Interval(max(lo, -_HALF_PI_HI), min(hi, _HALF_PI_HI))


def atan_point(x: float) -> Interval:
    """Guaranteed enclosure of ``atan(x)`` at a scalar point ``x``."""
    lo, hi = _enclose_point("atan", x)
    return Interval(max(lo, -_HALF_PI_HI), min(hi, _HALF_PI_HI))


#: A rigorous double bracket of ``1 / sqrt(2 pi)`` (the standard-normal density
#: normaliser) used by the gelu derivative tower.  The nearest double to the true
#: value is within half a ulp, so widening two representable steps in each
#: direction provably straddles it; any widening only makes downstream enclosures
#: conservatively wider, never unsound.
_INV_SQRT_2PI = Interval(
    _pred(_pred(0.3989422804014327)), _succ(_succ(0.3989422804014327))
)


def erf_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of ``erf(x)`` over ``x`` (monotone; clamped to ``[-1, 1]``)."""
    lo, hi = _monotone_increasing("erf", x)
    return Interval(max(lo, -1.0), min(hi, 1.0))


def erf_point(x: float) -> Interval:
    """Guaranteed enclosure of ``erf(x)`` at a scalar point ``x``."""
    lo, hi = _enclose_point("erf", x)
    return Interval(max(lo, -1.0), min(hi, 1.0))


def gauss_cdf_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of the standard-normal CDF ``Phi`` (monotone; clamped to ``[0, 1]``).

    ``Phi(x) = 1/2 (1 + erf(x / sqrt 2))`` is strictly increasing, so the enclosure
    is built from the lower endpoint at ``x.lo`` and the upper endpoint at ``x.hi``.
    """
    lo, hi = _monotone_increasing("gauss_cdf", x)
    return Interval(max(lo, 0.0), min(hi, 1.0))


def gauss_cdf_point(x: float) -> Interval:
    """Guaranteed enclosure of the standard-normal CDF ``Phi(x)`` at a scalar point ``x``."""
    lo, hi = _enclose_point("gauss_cdf", x)
    return Interval(max(lo, 0.0), min(hi, 1.0))


def ln_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of ``ln(x)`` over ``x`` (monotone; requires ``x.lo > 0``)."""
    if x.lo <= 0.0:
        raise ValueError("ln_iv requires a strictly positive interval")
    lo, hi = _monotone_increasing("log", x)
    return Interval(lo, hi)


def ln_point(x: float) -> Interval:
    """Guaranteed enclosure of ``ln(x)`` at a scalar point ``x > 0``."""
    if x <= 0.0:
        raise ValueError("ln_point requires x > 0")
    lo, hi = _enclose_point("log", x)
    return Interval(lo, hi)


def _softplus_point(x: float) -> Interval:
    """Guaranteed enclosure of ``softplus(x) = ln(1 + exp(x))`` at a scalar ``x``.

    Evaluated in the numerically stable branch ``softplus(x) = max(x, 0) +
    ln(1 + exp(-|x|))`` so the inner ``exp`` argument is never positive and never
    overflows for any finite double ``x``.
    """
    one = Interval.point(1.0)
    if x >= 0.0:
        e = exp_iv(Interval.point(-x))  # exp(-x) in (0, 1]
        return Interval.point(x) + ln_iv(one + e)
    e = exp_iv(Interval.point(x))  # exp(x) in (0, 1)
    return ln_iv(one + e)


def softplus_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of ``softplus(x) = ln(1 + exp(x))`` over ``x``.

    ``softplus`` is strictly increasing and positive, so the enclosure is built
    from the lower endpoint at ``x.lo`` and the upper endpoint at ``x.hi`` and
    clamped to the true range ``(0, inf)``.
    """
    lo = _softplus_point(x.lo).lo
    hi = _softplus_point(x.hi).hi
    return Interval(max(lo, 0.0), hi)


#: A rigorous enclosure of Euler's number ``e = exp(1)`` via :func:`exp_iv`.  It
#: uses the active transcendental backend (mpmath when present, else the
#: ulp-inflated libm fallback), so it is exactly as rigorous as any other
#: ``exp_iv`` call -- the public constant that Euler-Maclaurin / asymptotic
#: enclosures compose with (alongside :data:`PI_IV`).
E_IV = exp_iv(Interval.point(1.0))
# Module-level constants must not poison :func:`libm_fallback_used` for later
# certificate scopes: the stamp on a sealed certificate records the live backend.
clear_libm_fallback_used()


__all__ = [
    "BACKEND_LIBM_FALLBACK",
    "BACKEND_MPMATH",
    "E_IV",
    "FALLBACK_ULPS",
    "MPMATH_DPS",
    "PI_IV",
    "atan_iv",
    "atan_point",
    "backend_name",
    "certificate_mode",
    "clear_libm_fallback_used",
    "cos_iv",
    "cos_point",
    "cosh_iv",
    "erf_iv",
    "erf_point",
    "exp_iv",
    "gauss_cdf_iv",
    "gauss_cdf_point",
    "libm_fallback_used",
    "ln_iv",
    "ln_point",
    "require_rigorous_backend",
    "sech_iv",
    "set_strict_backend",
    "sigmoid_iv",
    "sin_iv",
    "sin_point",
    "softplus_iv",
    "strict_backend",
    "tanh_iv",
]

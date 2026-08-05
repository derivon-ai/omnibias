# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The :class:`TimeScale` object: a closed subset of ``R`` with jump operators.

A *time scale* ``T`` is a nonempty closed subset of the reals. omnibias supports the four
that matter in practice:

* ``reals()`` -- the continuum ``R`` (everywhere right/left **dense**, graininess ``0``);
* ``h_integers(h)`` -- the uniform mesh ``hZ`` (everywhere **scattered**, graininess ``h``);
* ``quantum(q)`` -- the quantum scale ``{q^k : k in Z} cup {0}`` with ``q > 1`` (scattered
  away from ``0``, graininess ``mu(t) = (q-1)t``);
* ``finite(points)`` -- an explicit finite set.

The forward jump ``sigma(t) = inf{s in T : s > t}``, backward jump
``rho(t) = sup{s in T : s < t}``, and graininess ``mu(t) = sigma(t) - t`` all have closed
forms per scale. As ``mu -> 0`` the scale is ``R`` and the time-scale derivative becomes
the ordinary derivative -- the founding ``delta -> 0`` collapse of a difference into a
derivative, generalized to a variable mesh (see :mod:`omnibias.timescale._core.derivative`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_KINDS = ("reals", "h_integers", "quantum", "finite")


@dataclass(frozen=True)
class TimeScale:
    """A time scale ``T`` and its closed-form jump operators.

    Construct via the factories :func:`reals`, :func:`h_integers`, :func:`quantum`, or
    :func:`finite` rather than the raw constructor.
    """

    kind: str
    h: float = 1.0
    q: float = 2.0
    points: tuple[float, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        if self.kind == "h_integers" and not self.h > 0:
            raise ValueError(f"h_integers needs h > 0, got {self.h}")
        if self.kind == "quantum" and not self.q > 1:
            raise ValueError(f"quantum scale needs q > 1, got {self.q}")
        if self.kind == "finite" and len(self.points) < 2:
            raise ValueError("finite time scale needs at least two points")

    # -- forward jump / graininess ------------------------------------------------
    def sigma(self, t: float) -> float:
        """Forward jump ``sigma(t) = inf{s in T : s > t}`` (``t`` if right-maximal)."""
        if self.kind == "reals":
            return t
        if self.kind == "h_integers":
            return t + self.h
        if self.kind == "quantum":
            return 0.0 if t == 0.0 else self.q * t
        nxt = [p for p in self.points if p > t + _tol(t)]
        return min(nxt) if nxt else t

    def rho(self, t: float) -> float:
        """Backward jump ``rho(t) = sup{s in T : s < t}`` (``t`` if left-minimal)."""
        if self.kind == "reals":
            return t
        if self.kind == "h_integers":
            return t - self.h
        if self.kind == "quantum":
            return 0.0 if t == 0.0 else t / self.q
        prev = [p for p in self.points if p < t - _tol(t)]
        return max(prev) if prev else t

    def mu(self, t: float) -> float:
        """Forward graininess ``mu(t) = sigma(t) - t`` (``0`` on ``R``)."""
        return self.sigma(t) - t

    def nu(self, t: float) -> float:
        """Backward graininess ``nu(t) = t - rho(t)``."""
        return t - self.rho(t)

    # -- point classification -----------------------------------------------------
    def is_right_scattered(self, t: float) -> bool:
        """``sigma(t) > t``: a forward-isolated point (the difference-quotient case)."""
        return self.sigma(t) > t + _tol(t)

    def is_right_dense(self, t: float) -> bool:
        """``sigma(t) == t``: a forward-limit point (the derivative case)."""
        return not self.is_right_scattered(t)

    def is_left_scattered(self, t: float) -> bool:
        """``rho(t) < t``: a backward-isolated point."""
        return self.rho(t) < t - _tol(t)

    def is_left_dense(self, t: float) -> bool:
        """``rho(t) == t``: a backward-limit point."""
        return not self.is_left_scattered(t)

    def contains(self, t: float) -> bool:
        """Whether ``t in T`` (up to a small tolerance for the mesh scales)."""
        if self.kind == "reals":
            return True
        if self.kind == "h_integers":
            return abs(t / self.h - round(t / self.h)) < 1e-9
        if self.kind == "quantum":
            if t == 0.0:
                return True
            if t < 0.0:
                return False
            k = math.log(t) / math.log(self.q)
            return abs(k - round(k)) < 1e-9
        return any(abs(t - p) < _tol(t) for p in self.points)

    def grid(self, a: float, b: float) -> tuple[float, ...]:
        """The scale points in ``[a, b]`` (ascending). Undefined for the continuum."""
        if self.kind == "reals":
            raise ValueError("the continuum R has no discrete grid; use a quadrature")
        if self.kind == "h_integers":
            n0 = math.ceil(a / self.h - 1e-9)
            n1 = math.floor(b / self.h + 1e-9)
            return tuple(k * self.h for k in range(n0, n1 + 1))
        if self.kind == "quantum":
            pts: list[float] = []
            if a <= 0.0 <= b:
                pts.append(0.0)
            if b > 0.0:
                lo = max(a, 0.0)
                # smallest k with q^k >= max(lo, tiny); largest with q^k <= b
                kmax = math.floor(math.log(b) / math.log(self.q) + 1e-9)
                kmin = math.ceil(math.log(lo) / math.log(self.q) - 1e-9) if lo > 0 else kmax - 60
                pts.extend(self.q**k for k in range(kmin, kmax + 1) if a - _tol(a) <= self.q**k <= b + _tol(b))
            return tuple(sorted(pts))
        return tuple(p for p in self.points if a - _tol(a) <= p <= b + _tol(b))


def _tol(t: float) -> float:
    return 1e-12 * max(1.0, abs(t))


def reals() -> TimeScale:
    """The continuum ``R`` (dense everywhere, graininess ``0``)."""
    return TimeScale("reals")


def h_integers(h: float = 1.0) -> TimeScale:
    """The uniform mesh ``hZ`` (scattered everywhere, graininess ``h``)."""
    return TimeScale("h_integers", h=h)


def quantum(q: float = 2.0) -> TimeScale:
    """The quantum scale ``{q^k} cup {0}`` with ``q > 1`` (graininess ``(q-1)t``)."""
    return TimeScale("quantum", q=q)


def finite(points: tuple[float, ...]) -> TimeScale:
    """An explicit finite time scale from a set of points (sorted, de-duplicated)."""
    uniq = tuple(sorted(set(points)))
    return TimeScale("finite", points=uniq)


__all__ = [
    "TimeScale",
    "finite",
    "h_integers",
    "quantum",
    "reals",
]

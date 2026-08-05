# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Newton-Kantorovich / radii-polynomial existence certificates.

These are the standard workhorses of computer-assisted analysis: from an
*approximate* zero ``x_bar`` of ``F`` and an approximate inverse ``A`` of the
linearisation ``DF(x_bar)``, they prove a **true** zero exists in an explicit
ball around ``x_bar`` -- and is unique there.

Two complementary, fully rigorous routes are provided:

* :func:`radii_polynomial_certificate` -- the scalar *radii polynomial*
  ``p(r) = Z2 r^2 - (1 - Z0 - Z1) r + Y0``.  If some ``r`` gives ``p(r) < 0`` and
  the contraction factor ``Z0 + Z1 + 2 Z2 r < 1`` then the Newton-like operator is
  a self-map and contraction on ``B(x_bar, r)`` -- a unique zero exists there.
  The bounds ``Y0, Z0, Z1, Z2`` are rigorous *upper* bounds; the certificate
  re-verifies the chosen radius in interval arithmetic.
* :func:`krawczyk_certificate` -- the finite-dimensional **Krawczyk** test, which
  evaluates ``DF`` over the whole box and needs no separate ``Z2``: if
  ``K(x_bar, X) subset int(X)`` and ``||I - A DF(X)|| < 1`` then ``F`` has a unique
  zero in ``X = B(x_bar, r)``.

:func:`newton_kantorovich_bounds` computes ``(Y0, Z0, Z1, Z2)`` from ``F``, its
Jacobian, ``A`` and a caller-supplied Lipschitz bound on ``DF`` (explicit for
polynomial maps), feeding the radii polynomial.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.linalg import (
    identity_matrix,
    inf_norm_matrix,
    inf_norm_vector,
    mat_sub,
    matmul,
    matvec,
    to_interval_matrix,
)

if TYPE_CHECKING:
    # Imported lazily at call time to avoid a proof <-> verified import cycle.
    from omnibias.core.proof.certificate import Cert

#: ``F`` / Jacobian act on a list of interval coordinates.
IntervalMap = Callable[[list[Interval]], Sequence[IntervalLike]]
IntervalJac = Callable[[list[Interval]], Sequence[Sequence[IntervalLike]]]


@dataclass(frozen=True)
class RadiiCertificate:
    """A verified existence radius from the radii polynomial."""

    radius: float
    kappa: float
    p_value: float
    y0: float
    z0: float
    z1: float
    z2: float
    r_interval: tuple[float, float]
    certificate: Cert


@dataclass(frozen=True)
class KrawczykCertificate:
    """A verified unique-zero box from the Krawczyk operator."""

    radius: float
    kappa: float
    center: tuple[float, ...]
    enclosure: tuple[tuple[float, float], ...]
    certificate: Cert


@dataclass(frozen=True)
class NKBounds:
    """The Newton-Kantorovich bounds feeding the radii polynomial."""

    y0: float
    z0: float
    z1: float
    z2: float


def radii_polynomial_certificate(
    y0: float,
    z0: float,
    z1: float,
    z2: float,
    *,
    r_max: float = math.inf,
    claim: str = "unique zero in closed ball B(x_bar, r)",
) -> RadiiCertificate | None:
    r"""Verify the radii polynomial and return an existence certificate, or ``None``.

    ``y0, z0, z1, z2`` must be rigorous **upper** bounds.  Returns ``None`` when no
    admissible radius exists (no contraction, or the discriminant is non-positive).
    """
    y0, z0, z1, z2 = float(y0), float(z0), float(z1), float(z2)
    if min(y0, z0, z1, z2) < 0.0:
        raise ValueError("radii-polynomial bounds must be non-negative")
    slack = 1.0 - z0 - z1
    if slack <= 0.0:
        return None

    if z2 == 0.0:
        r_lo = y0 / slack
        r_hi = r_max
        candidates = [r_lo * (1.0 + t) for t in (1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0)]
    else:
        disc = slack * slack - 4.0 * z2 * y0
        if disc <= 0.0:
            return None
        s = math.sqrt(disc)
        r_lo = (slack - s) / (2.0 * z2)
        r_hi = (slack + s) / (2.0 * z2)
        vertex = slack / (2.0 * z2)
        candidates = [r_lo + frac * (vertex - r_lo) for frac in (0.25, 0.5, 0.1, 0.75, 0.9)]

    from omnibias.core.proof.certificate import make_certificate

    yi, z0i, z1i, z2i = (Interval.point(v) for v in (y0, z0, z1, z2))
    two = Interval.point(2.0)
    for r0 in candidates:
        if not (r0 > 0.0 and r0 <= r_max):
            continue
        ri = Interval.point(r0)
        p = yi + (z0i + z1i) * ri + z2i * ri * ri - ri
        kappa = z0i + z1i + two * z2i * ri
        if p.hi < 0.0 and kappa.hi < 1.0:
            cert = make_certificate(
                claim=claim,
                payload={
                    "type": "radii_polynomial",
                    "radius": r0,
                    "kappa": kappa.hi,
                    "p_value": p.hi,
                    "Y0": y0,
                    "Z0": z0,
                    "Z1": z1,
                    "Z2": z2,
                },
            )
            return RadiiCertificate(r0, kappa.hi, p.hi, y0, z0, z1, z2, (r_lo, r_hi), cert)
    return None


def newton_kantorovich_bounds(
    func: IntervalMap,
    jacobian: IntervalJac,
    x_bar: Sequence[float],
    a_inv: Sequence[Sequence[float]],
    *,
    lipschitz_df: float,
) -> NKBounds:
    r"""Compute ``(Y0, Z0, Z1, Z2)`` for the radii polynomial.

    ``lipschitz_df`` is a rigorous bound on ``||DF(x) - DF(y)||_inf`` per unit
    ``||x - y||_inf`` over the region of interest (explicit for polynomial maps).
    ``Z2 = ||A||_inf * lipschitz_df``; ``Z1 = 0`` (the linearisation we invert is
    ``DF(x_bar)`` itself, whose enclosure folds into ``Z0``).
    """
    n = len(x_bar)
    a_iv = to_interval_matrix(a_inv)
    center = [Interval.point(v) for v in x_bar]
    fx = [Interval.from_value(v) for v in func(center)]
    y0 = inf_norm_vector(matvec(a_iv, fx))
    jc = to_interval_matrix([[Interval.from_value(v) for v in row] for row in jacobian(center)])
    z0 = inf_norm_matrix(mat_sub(identity_matrix(n), matmul(a_iv, jc)))
    z2 = Interval.point(inf_norm_matrix(a_iv)) * Interval.point(float(lipschitz_df))
    return NKBounds(y0=y0, z0=z0, z1=0.0, z2=z2.hi)


def certify_zero_radii(
    func: IntervalMap,
    jacobian: IntervalJac,
    x_bar: Sequence[float],
    a_inv: Sequence[Sequence[float]],
    *,
    lipschitz_df: float,
    r_max: float = math.inf,
) -> RadiiCertificate | None:
    """End-to-end radii-polynomial existence proof for ``F(x) = 0``."""
    bounds = newton_kantorovich_bounds(
        func, jacobian, x_bar, a_inv, lipschitz_df=lipschitz_df
    )
    return radii_polynomial_certificate(
        bounds.y0, bounds.z0, bounds.z1, bounds.z2, r_max=r_max
    )


def krawczyk_certificate(
    func: IntervalMap,
    jacobian: IntervalJac,
    x_bar: Sequence[float],
    a_inv: Sequence[Sequence[float]],
    r: float,
) -> KrawczykCertificate | None:
    r"""Krawczyk unique-zero test on the box ``B(x_bar, r)`` (rigorous).

    Returns a certificate iff ``K(x_bar, X) subset int(X)`` and
    ``||I - A DF(X)||_inf < 1`` -- a unique zero of ``F`` then lies in ``X``.
    """
    n = len(x_bar)
    if r <= 0.0:
        raise ValueError("box radius r must be positive")
    box = [Interval(x_bar[i] - r, x_bar[i] + r) for i in range(n)]
    a_iv = to_interval_matrix(a_inv)
    fx = [Interval.from_value(v) for v in func([Interval.point(v) for v in x_bar])]
    a_fx = matvec(a_iv, fx)
    j_box = to_interval_matrix([[Interval.from_value(v) for v in row] for row in jacobian(box)])
    m = mat_sub(identity_matrix(n), matmul(a_iv, j_box))  # I - A DF(X)
    diff = [box[i] - Interval.point(x_bar[i]) for i in range(n)]
    m_diff = matvec(m, diff)
    k = [Interval.point(x_bar[i]) - a_fx[i] + m_diff[i] for i in range(n)]
    kappa = inf_norm_matrix(m)
    inside = all(box[i].lo < k[i].lo and k[i].hi < box[i].hi for i in range(n))
    if kappa < 1.0 and inside:
        from omnibias.core.proof.certificate import make_certificate

        enclosure = tuple((k[i].lo, k[i].hi) for i in range(n))
        cert = make_certificate(
            claim="unique zero of F in the Krawczyk box",
            payload={
                "type": "krawczyk",
                "radius": r,
                "kappa": kappa,
                "center": list(x_bar),
                "enclosure": [[lo, hi] for lo, hi in enclosure],
            },
        )
        return KrawczykCertificate(r, kappa, tuple(x_bar), enclosure, cert)
    return None


def krawczyk_search(
    func: IntervalMap,
    jacobian: IntervalJac,
    x_bar: Sequence[float],
    a_inv: Sequence[Sequence[float]],
    *,
    radii: Sequence[float],
) -> KrawczykCertificate | None:
    """Try :func:`krawczyk_certificate` over ``radii`` (ascending), return the first hit."""
    for r in radii:
        cert = krawczyk_certificate(func, jacobian, x_bar, a_inv, r)
        if cert is not None:
            return cert
    return None


__all__ = [
    "KrawczykCertificate",
    "NKBounds",
    "RadiiCertificate",
    "certify_zero_radii",
    "krawczyk_certificate",
    "krawczyk_search",
    "newton_kantorovich_bounds",
    "radii_polynomial_certificate",
]

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified scan-peak localization (theory 03-08).

A scan template comes from the founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``,
feasibility) does not appear. Do not conflate the two.

Krawczyk on ``r'`` proves a unique stationary point in a
``local_box``. Flat or inflected peaks return
:class:`Inconclusive` rather than a certificate. The sealed
payload is a sound enclosure, not ``theorem_prover_verified``.
A deterministic enclosure of noisy data is conditional on the
data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import krawczyk_certificate
from omnibias.core.verified.sigma import sigma_tower_interval

RouteName = Literal["krawczyk", "branch_and_bound"]
SignName = Literal["negative", "positive", "indeterminate"]


def honesty_payload() -> dict[str, object]:
    return {
        "scope": "local_box",
        "sound_not_complete": True,
        "conditional_on_data": True,
        "theorem_prover_verified": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
    }


@dataclass(frozen=True)
class ScanResponse:
    """Closed-form scan ``r(tau) = sum c * tanh^{(n)}(alpha (tau - t))``.

    The order-1 (bump) template is the inner product of an interface
    with ``sigma'``. Differentiating in ``tau`` raises the tower order.
    """

    terms: tuple[tuple[float, float, float], ...]
    direction: float = 1.0
    bias: float = 0.0
    template_order: int = 1
    polynomial: tuple[float, ...] | None = None

    @classmethod
    def sech2_peak(cls, tau_star: float, *, alpha: float = 5.0, direction: float = 1.0, bias: float = 0.0) -> ScanResponse:
        return cls(terms=((float(tau_star), float(alpha), 1.0),), direction=float(direction), bias=float(bias))

    @classmethod
    def two_peaks(cls, tau_a: float, tau_b: float, *, alpha: float = 5.0) -> ScanResponse:
        return cls(terms=((float(tau_a), float(alpha), 1.0), (float(tau_b), float(alpha), 1.0)))

    @classmethod
    def constant(cls, value: float = 1.0) -> ScanResponse:
        return cls(terms=(), polynomial=(float(value),))

    @classmethod
    def flat_max(cls) -> ScanResponse:
        """``r = -tau^4``: a maximum at 0 with ``r''(0) = 0``."""
        return cls(terms=(), polynomial=(0.0, 0.0, 0.0, 0.0, -1.0))

    def deriv(self, tau: Interval, order: int) -> Interval:
        k = int(order)
        if k < 0:
            raise ValueError("order must be >= 0")
        acc = Interval.point(0.0)
        if self.polynomial is not None:
            acc = acc + _poly_deriv(self.polynomial, tau, k)
        n = int(self.template_order)
        for center, alpha, coeff in self.terms:
            z = Interval.point(alpha) * (tau - Interval.point(center))
            tower = sigma_tower_interval("tanh", z, n + k)
            acc = acc + Interval.point(coeff) * (Interval.point(alpha) ** k) * tower[n + k]
        return acc

    def physical_from_offset(self, offset: Interval) -> Interval:
        w = abs(float(self.direction))
        if w == 0.0:
            raise ValueError("direction must be nonzero")
        return -(offset + Interval.point(self.bias)) / Interval.point(w)


def _poly_deriv(coeffs: Sequence[float], tau: Interval, k: int) -> Interval:
    acc = Interval.point(0.0)
    for i, c in enumerate(coeffs):
        if i < k or c == 0.0:
            continue
        factor = 1.0
        for j in range(k):
            factor *= float(i - j)
        acc = acc + Interval.point(factor * float(c)) * (tau ** (i - k) if i > k else Interval.point(1.0))
    return acc


@dataclass(frozen=True)
class Inconclusive:
    """First-class refusal. Not a wide interval and not a certificate."""

    reason: str
    box: Interval


@dataclass(frozen=True)
class LocalizationCertificate:
    offset_enclosure: Interval
    physical_enclosure: Interval
    unique_in: Interval
    second_derivative_sign: SignName
    route: RouteName
    scope: str = "local_box"
    widths: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.scope != "local_box":
            raise ValueError("scope must be 'local_box'")


def tight_deriv(response: ScanResponse, box: Interval, order: int, *, max_width: float = 0.03) -> Interval:
    """Sound hull of ``r^{(k)}`` on a partition. Tightens wrapping; does not fork Krawczyk."""
    if box.width <= float(max_width):
        return response.deriv(box, int(order))
    mid = box.mid
    left = tight_deriv(response, Interval(box.lo, mid), order, max_width=max_width)
    right = tight_deriv(response, Interval(mid, box.hi), order, max_width=max_width)
    return Interval.hull(left, right)


def _sign_of_second(rpp: Interval) -> SignName:
    if rpp.hi < 0.0:
        return "negative"
    if rpp.lo > 0.0:
        return "positive"
    return "indeterminate"


def _krawczyk_step(response: ScanResponse, box: Interval) -> Interval | None:
    m = box.mid
    radius = 0.5 * (box.hi - box.lo)
    if radius <= 0.0:
        return None
    rpp_m = response.deriv(Interval.point(m), 2)
    if rpp_m.contains_zero():
        return None
    a_inv = [[1.0 / rpp_m.mid]]

    def func(xs: list[Interval]) -> list[Interval]:
        return [response.deriv(xs[0], 1)]

    def jac(xs: list[Interval]) -> list[list[Interval]]:
        return [[tight_deriv(response, xs[0], 2)]]

    hit = krawczyk_certificate(func, jac, [m], a_inv, radius)
    if hit is None:
        return None
    lo, hi = hit.enclosure[0]
    return Interval(float(lo), float(hi))


def branch_and_bound_peak(response: ScanResponse, box: Interval, *, n_iters: int) -> Interval:
    """Route-1 max enclosure: keep cells whose ``r`` upper bound meets the global lower bound."""
    cells = [box]
    for _ in range(int(n_iters)):
        nxt: list[Interval] = []
        for cell in cells:
            mid = cell.mid
            nxt.append(Interval(cell.lo, mid))
            nxt.append(Interval(mid, cell.hi))
        cells = nxt
    vals = [response.deriv(cell, 0) for cell in cells]
    floor = max(v.lo for v in vals)
    keep = [cell for cell, val in zip(cells, vals, strict=True) if val.hi >= floor]
    return Interval(min(c.lo for c in keep), max(c.hi for c in keep))


def certify_peak(
    response: ScanResponse,
    *,
    box: Interval,
    max_iter: int = 20,
) -> LocalizationCertificate | Inconclusive:
    """Krawczyk on ``r'``. Falls back to B&B only when ``r'' < 0`` but wrapping blocks K."""
    if box.width <= 0.0:
        return Inconclusive("empty_box", box)
    rpp = tight_deriv(response, box, 2)
    sign = _sign_of_second(rpp)
    if sign != "negative":
        return Inconclusive(
            "second_derivative_indeterminate" if sign == "indeterminate" else "not_a_maximum",
            box,
        )
    rp_lo = response.deriv(Interval.point(box.lo), 1)
    rp_hi = response.deriv(Interval.point(box.hi), 1)
    if rp_lo.hi < 0.0 and rp_hi.hi < 0.0:
        return Inconclusive("no_stationary_point", box)
    if rp_lo.lo > 0.0 and rp_hi.lo > 0.0:
        return Inconclusive("no_stationary_point", box)
    current = box
    widths: list[float] = [current.width]
    route: RouteName = "krawczyk"
    for _ in range(int(max_iter)):
        nxt = _krawczyk_step(response, current)
        if nxt is None:
            if len(widths) == 1:
                bab = branch_and_bound_peak(response, box, n_iters=max(3, int(max_iter)))
                if bab.width >= box.width * 0.999:
                    return Inconclusive("krawczyk_wrapping", box)
                route = "branch_and_bound"
                current = bab
                widths.append(current.width)
                break
            break
        if nxt.lo <= current.lo or nxt.hi >= current.hi:
            break
        current = nxt
        widths.append(current.width)
        if current.width <= 1e-15:
            break
    if current.width >= box.width:
        return Inconclusive("failed_to_contract", box)
    return LocalizationCertificate(
        offset_enclosure=current,
        physical_enclosure=response.physical_from_offset(current),
        unique_in=box,
        second_derivative_sign=sign,
        route=route,
        widths=tuple(widths),
    )


def certify_multiple_peaks(
    response: ScanResponse,
    *,
    box: Interval,
    max_peaks: int,
    min_width: float = 1e-3,
) -> tuple[tuple[LocalizationCertificate, ...], bool]:
    """Subdivide until each surviving cell certifies or is smaller than ``min_width``."""
    found: list[LocalizationCertificate] = []
    queue = [box]
    exhaustive = True
    while queue:
        cell = queue.pop()
        result = certify_peak(response, box=cell, max_iter=12)
        if isinstance(result, LocalizationCertificate):
            found.append(result)
            if len(found) >= int(max_peaks):
                exhaustive = False
                break
            continue
        if cell.width <= float(min_width):
            exhaustive = False
            continue
        mid = cell.mid
        queue.append(Interval(cell.lo, mid))
        queue.append(Interval(mid, cell.hi))
    return tuple(found), exhaustive


def seal(cert: LocalizationCertificate) -> Cert:
    """Hash-sealed v1 certificate. Does not assert ``theorem_prover_verified``."""
    return make_certificate(
        claim="unique scan peak in a local box",
        payload={
            "type": "scan_localization",
            "offset": [cert.offset_enclosure.lo, cert.offset_enclosure.hi],
            "physical": [cert.physical_enclosure.lo, cert.physical_enclosure.hi],
            "unique_in": [cert.unique_in.lo, cert.unique_in.hi],
            "second_derivative_sign": cert.second_derivative_sign,
            "route": cert.route,
            "scope": cert.scope,
            "formal_tier": "sound_enclosure",
        },
        honesty={
            "unproven_claim": False,
            "conditional_on_data": True,
        },
        meta={"scope": "local_box"},
    )


def sealed_digest_ok(cert: LocalizationCertificate) -> bool:
    return verify_certificate_digest(seal(cert))


__all__ = [
    "Inconclusive",
    "LocalizationCertificate",
    "ScanResponse",
    "branch_and_bound_peak",
    "certify_multiple_peaks",
    "certify_peak",
    "honesty_payload",
    "tight_deriv",
    "seal",
    "sealed_digest_ok",
    "tight_deriv",
]

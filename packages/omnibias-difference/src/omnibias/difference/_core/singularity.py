# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Jet-Padé singularity location (theory 03-10).

A high-order jet is a truncated Taylor series. Domb-Sykes and
Padé poles estimate the nearest singularity; a coefficient tail
bound turns ``|x_s|`` into a sound annulus. This is a diagnostic
and an estimate, not a proof of blow-up.

Jets come from the founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. Do not conflate the two.

Domb-Sykes assumes a single dominant algebraic singularity.
Essential singularities and comparable-distance pairs must
report failure rather than a confident wrong answer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import inf, nextafter
from typing import Literal

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sequence_space import geometric_tail_bound
from omnibias.difference._core.generating import _polynomial_roots  # type: ignore[import-not-found]
from omnibias.difference._core.pade import (  # type: ignore[import-not-found]
    pade_approximant,
    pade_certified_remainder,
)

DISCLAIMER = "diagnostic and estimate; not a proof of blow-up"
# Froissart doublets are pole-zero pairs at machine scale (Weinstein-Saff).
# 1e-8 is ~45 ulps at unit scale, independent of the recovery suite.
FROISSART_REL = 1e-8


def honesty_payload() -> dict[str, object]:
    return {
        "blowup_proof": False,
        "diagnostic": True,
        "disclaimer": DISCLAIMER,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "ns_regularity": False,
        "order_ceiling_float64": 20,
    }


@dataclass(frozen=True)
class SingularityEstimate:
    location: complex | None
    exponent: float | None
    method: Literal["domb_sykes", "pade", "both", "failed"]
    residual: float
    enclosure: Interval | None
    failed: bool = False
    reason: str = ""


@dataclass(frozen=True)
class BlowupFit:
    t_c: float
    gamma: float
    prefactor: float
    t_c_lo: float
    t_c_hi: float


@dataclass(frozen=True)
class SingularityTrack:
    times: tuple[float, ...]
    locations: tuple[complex, ...]
    exponents: tuple[float, ...]
    blowup_fit: BlowupFit | None
    disclaimer: str = DISCLAIMER

    def to_payload(self) -> dict[str, object]:
        return {
            "times": list(self.times),
            "locations": [[z.real, z.imag] for z in self.locations],
            "exponents": list(self.exponents),
            "blowup_fit": None
            if self.blowup_fit is None
            else {
                "t_c": self.blowup_fit.t_c,
                "gamma": self.blowup_fit.gamma,
                "prefactor": self.blowup_fit.prefactor,
                "t_c_lo": self.blowup_fit.t_c_lo,
                "t_c_hi": self.blowup_fit.t_c_hi,
            },
            "disclaimer": self.disclaimer,
        }


def _as_float(coeffs: Sequence[complex | float]) -> list[complex]:
    return [complex(c) for c in coeffs]


def _linfit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two points")
    mx = sum(xs) / n
    my = sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0.0:
        raise ValueError("degenerate Domb-Sykes abscissa")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var
    intercept = my - slope * mx
    resid = math.sqrt(sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True)) / n)
    return slope, intercept, resid


def domb_sykes(coeffs: Sequence[complex | float], *, drop_first: int = 1) -> SingularityEstimate:
    """Linear fit of ``a_k / a_{k-1}`` against ``1/k``. Failure is first-class."""
    c = _as_float(coeffs)
    xs: list[float] = []
    ys: list[complex] = []
    start = max(int(drop_first), 1)
    for k in range(start, len(c)):
        prev = c[k - 1]
        if abs(prev) <= 1e-30:
            return SingularityEstimate(None, None, "failed", math.inf, None, True, "vanishing_coefficient")
        ratio = c[k] / prev
        xs.append(1.0 / float(k))
        ys.append(ratio)
    if len(xs) < 3:
        return SingularityEstimate(None, None, "failed", math.inf, None, True, "too_few_ratios")
    slope_r, intercept_r, resid_r = _linfit(xs, [z.real for z in ys])
    slope_i, intercept_i, resid_i = _linfit(xs, [z.imag for z in ys])
    intercept = complex(intercept_r, intercept_i)
    resid = math.hypot(resid_r, resid_i)
    scale = max(abs(intercept), 1.0)
    if abs(intercept) < 1e-8:
        return SingularityEstimate(None, None, "failed", resid, None, True, "essential_or_infinite_radius")
    if resid / scale > 0.05:
        return SingularityEstimate(None, None, "failed", resid, None, True, "not_single_algebraic")
    xs_loc = 1.0 / intercept
    exponent = 1.0 + complex(slope_r, slope_i) * xs_loc
    return SingularityEstimate(xs_loc, float(exponent.real), "domb_sykes", float(resid), None, False, "")


def pade_singularities(
    coeffs: Sequence[complex | float],
    *,
    numer_deg: int,
    denom_deg: int,
) -> tuple[complex, ...]:
    """Denominator roots of ``[L/M]`` after a justified Froissart filter."""
    fracs: list[Fraction] = []
    for c in coeffs:
        z = complex(c)
        if abs(z.imag) > 1e-14 * max(1.0, abs(z.real)):
            raise ValueError("pade_singularities expects real Taylor coefficients")
        fracs.append(Fraction(z.real).limit_denominator(10**12))
    numer, denom = pade_approximant(fracs, int(numer_deg), int(denom_deg))
    poles = _polynomial_roots([complex(q) for q in denom])
    zeros = _polynomial_roots([complex(p) for p in numer])
    kept: list[complex] = []
    for pole in poles:
        doublet = False
        for zero in zeros:
            if abs(pole - zero) <= FROISSART_REL * max(1.0, abs(pole)):
                doublet = True
                break
        if not doublet:
            kept.append(pole)
    return tuple(kept)


def pade_estimate(coeffs: Sequence[complex | float], *, numer_deg: int, denom_deg: int) -> SingularityEstimate:
    poles = pade_singularities(coeffs, numer_deg=numer_deg, denom_deg=denom_deg)
    if not poles:
        return SingularityEstimate(None, None, "failed", math.inf, None, True, "no_stable_pole")
    loc = min(poles, key=abs)
    return SingularityEstimate(loc, None, "pade", 0.0, None, False, "")


def agreement(a: SingularityEstimate, b: SingularityEstimate) -> float:
    """Relative location disagreement. ``inf`` if either method failed."""
    if a.failed or b.failed or a.location is None or b.location is None:
        return math.inf
    denom = max(abs(a.location), abs(b.location), 1e-16)
    return abs(a.location - b.location) / denom


def both_estimates(
    coeffs: Sequence[complex | float], *, numer_deg: int = 0, denom_deg: int = 1
) -> SingularityEstimate:
    ds = domb_sykes(coeffs)
    pd = pade_estimate(coeffs, numer_deg=numer_deg, denom_deg=denom_deg)
    if ds.failed and pd.failed:
        return SingularityEstimate(None, None, "failed", math.inf, None, True, "both_failed")
    if ds.failed:
        return pd
    if pd.failed:
        return ds
    gap = agreement(ds, pd)
    if gap > 0.05:
        return SingularityEstimate(None, None, "failed", gap, None, True, "methods_disagree")
    loc = 0.5 * (ds.location + pd.location)  # type: ignore[operator]
    exp = ds.exponent
    return SingularityEstimate(loc, exp, "both", gap, None, False, "")


def _abs_lo(iv: Interval) -> float:
    if iv.lo <= 0.0 <= iv.hi:
        return 0.0
    return min(abs(iv.lo), abs(iv.hi))


def certified_singularity_annulus(
    coeffs: Sequence[Interval | float],
    *,
    tail_bound: Interval,
    tail_ratio: float,
) -> Interval:
    """Sound enclosure of ``|x_s|`` from Cauchy-Hadamard plus a geometric tail."""
    if float(tail_ratio) <= 0.0:
        raise ValueError("tail_ratio must be > 0")
    ivs = [c if isinstance(c, Interval) else Interval.from_value(c) for c in coeffs]
    if len(ivs) < 2:
        raise ValueError("need at least two coefficients")
    l_hi = nextafter(float(tail_ratio) * max(1.0, tail_bound.hi) ** (1.0 / float(len(ivs))), inf)
    l_lo = 0.0
    for k in range(1, len(ivs)):
        hi = nextafter(ivs[k].mag ** (1.0 / float(k)), inf)
        raw_lo = _abs_lo(ivs[k])
        lo = nextafter(raw_lo ** (1.0 / float(k)), 0.0) if raw_lo > 0.0 else 0.0
        l_hi = max(l_hi, hi)
        l_lo = max(l_lo, lo)
    # Weighted tail in sequence space must be finite for the majorant to exist.
    _ = geometric_tail_bound(float(tail_bound.hi), float(tail_ratio), nu=0.5 / float(tail_ratio), n_trunc=len(ivs) - 1)
    if l_lo <= 0.0 or l_hi <= 0.0:
        raise ValueError("cannot enclose |x_s|: limsup lower bound is not positive")
    # |x_s| = 1 / limsup |a_k|^{1/k}, outward-rounded.
    return Interval(nextafter(1.0 / l_hi, 0.0), nextafter(1.0 / l_lo, inf))


def remainder_on_safe_disc(
    coeffs: Sequence[float],
    *,
    numer_deg: int,
    denom_deg: int,
    radius: float,
    tail_bound: float,
    tail_ratio: float,
) -> Interval:
    """Reuse ``pade_certified_remainder`` on a disc inside the Taylor radius."""
    fracs = [Fraction(float(c)).limit_denominator(10**12) for c in coeffs]
    numer, denom = pade_approximant(fracs, int(numer_deg), int(denom_deg))
    ivs = [Interval.from_value(float(c)) for c in coeffs]
    return pade_certified_remainder(numer, denom, ivs, float(radius), tail_bound=float(tail_bound), tail_ratio=float(tail_ratio))


def fit_blowup(times: Sequence[float], locations: Sequence[complex]) -> BlowupFit | None:
    """Linear fit of ``|Im x_s(t)| ~ C (t_c - t)``. Uncertainty is the residual band."""
    ts = [float(t) for t in times]
    ims = [abs(complex(z).imag) for z in locations]
    if len(ts) < 3 or min(ims) <= 0.0:
        return None
    slope, intercept, resid = _linfit(ts, ims)
    if abs(slope) < 1e-14:
        return None
    t_c = -intercept / slope
    # Im = slope * t + intercept = (-C) t + C t_c with C = -slope when approaching 0.
    prefactor = -slope if slope < 0.0 else abs(slope)
    width = max(3.0 * resid / max(prefactor, 1e-16), 1e-6)
    return BlowupFit(float(t_c), 1.0, float(prefactor), float(t_c - width), float(t_c + width))


def track_singularity(
    coeff_rows: Sequence[Sequence[complex | float]],
    times: Sequence[float],
    *,
    method: Literal["domb_sykes", "pade", "both"] = "both",
) -> SingularityTrack:
    """Track ``x_s(t)`` from jets at sampled times. Not a blow-up proof."""
    if len(coeff_rows) != len(times):
        raise ValueError("coeff_rows and times must have the same length")
    locs: list[complex] = []
    exps: list[float] = []
    for row in coeff_rows:
        if method == "domb_sykes":
            est = domb_sykes(row)
        elif method == "pade":
            est = pade_estimate(row, numer_deg=0, denom_deg=1)
        else:
            est = both_estimates(row, numer_deg=0, denom_deg=1)
        if est.failed or est.location is None:
            locs.append(complex("nan"))
            exps.append(float("nan"))
        else:
            locs.append(est.location)
            exps.append(float("nan") if est.exponent is None else float(est.exponent))
    finite = [(t, z) for t, z in zip(times, locs, strict=True) if z == z]
    fit = fit_blowup([p[0] for p in finite], [p[1] for p in finite]) if len(finite) >= 3 else None
    return SingularityTrack(tuple(float(t) for t in times), tuple(locs), tuple(exps), fit, DISCLAIMER)


__all__ = [
    "BlowupFit",
    "DISCLAIMER",
    "FROISSART_REL",
    "SingularityEstimate",
    "SingularityTrack",
    "agreement",
    "both_estimates",
    "certified_singularity_annulus",
    "domb_sykes",
    "fit_blowup",
    "honesty_payload",
    "pade_estimate",
    "pade_singularities",
    "remainder_on_safe_disc",
    "track_singularity",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Uncertainty slabs (theory 04-02).

A slab ``[lo, hi]`` is the shared *shape* of three different
guarantees. They are not interchangeable.

The band role uses a **finite** gap. That is the opposite of
founding bias collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. do not conflate
the two.

Conformal coverage is marginal and assumes exchangeability. It is
not a sound enclosure and must not be sealed as one.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

DISCLAIMER = (
    "three guarantee kinds stay apart: sound enclosure, conformal, "
    "and model-based; conformal is not sealable"
)

WORKED_RESIDUALS: tuple[float, ...] = (
    0.02,
    0.05,
    0.07,
    0.09,
    0.11,
    0.14,
    0.16,
    0.19,
    0.22,
    0.25,
    0.28,
    0.33,
    0.38,
    0.44,
    0.51,
    0.60,
    0.72,
    0.88,
    1.15,
)


class GuaranteeKind(str, Enum):
    SOUND_ENCLOSURE = "sound_enclosure"
    CONFORMAL = "conformal"
    MODEL_BASED = "model_based"


def honesty_payload() -> dict[str, bool]:
    return {
        "registers_blended": False,
        "conformal_sealed": False,
        "conditional_coverage_guaranteed": False,
        "exchangeability_assumed": True,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


def conformal_index(n: int, alpha: float) -> int:
    """1-based index ``ceil((n + 1)(1 - alpha))``. May equal ``n + 1``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return int(math.ceil((n + 1) * (1.0 - float(alpha))))


def split_conformal(residuals: Sequence[float], *, alpha: float) -> float:
    """Finite-sample corrected quantile of ``|residual|``."""
    scores = sorted(abs(float(v)) for v in residuals)
    n = len(scores)
    k = conformal_index(n, alpha)
    if k > n:
        return math.inf
    return scores[k - 1]


def adaptive_conformal(
    residuals: Sequence[float],
    widths: Sequence[float],
    *,
    alpha: float,
) -> float:
    """Quantile of the normalized score ``|y - f| / w(x)``."""
    if len(residuals) != len(widths):
        raise ValueError("residuals and widths must have the same length")
    norms: list[float] = []
    for res, width in zip(residuals, widths, strict=True):
        w = float(width)
        if w <= 0.0:
            raise ValueError("widths must be positive")
        norms.append(abs(float(res)) / w)
    return split_conformal(norms, alpha=alpha)


def softplus_width(z: float) -> float:
    """Positive OMBU width. Derivative is ``sigmoid(z)`` (closed form)."""
    x = float(z)
    if x > 40.0:
        return x
    if x < -40.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def softplus_width_prime(z: float) -> float:
    x = float(z)
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class UncertaintyInterval:
    lo: float
    hi: float
    kind: GuaranteeKind
    level: float | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if float(self.lo) > float(self.hi):
            raise ValueError("lo must be <= hi")
        if self.kind is GuaranteeKind.SOUND_ENCLOSURE:
            if self.level is not None:
                raise ValueError("sound enclosure has level None")
        elif self.level is None:
            raise ValueError(f"{self.kind.value} requires a probability level")

    def __add__(self, other: object) -> UncertaintyInterval:
        raise TypeError(
            "Intervals of different guarantee kinds do not combine by "
            "arithmetic. Use combine_enclosure_with_conformal."
        )

    def __radd__(self, other: object) -> UncertaintyInterval:
        return self.__add__(other)


@dataclass(frozen=True)
class CombinedStatement:
    """Enclosure and conformal residual, recorded separately."""

    enclosure: UncertaintyInterval
    conformal_q: float
    alpha: float
    combined_lo: float
    combined_hi: float

    def __post_init__(self) -> None:
        if self.enclosure.kind is not GuaranteeKind.SOUND_ENCLOSURE:
            raise TypeError("combine needs a sound enclosure")
        if self.enclosure.level is not None:
            raise TypeError("enclosure level must stay None")


@dataclass(frozen=True)
class CalibrationReport:
    marginal_coverage: float
    conditional_coverage: Mapping[str, float]
    average_width: float
    width_coverage_curve: tuple[tuple[float, float], ...]
    exchangeability_ok: bool = True
    shift_detected: bool = False


def combine_enclosure_with_conformal(
    enclosure: UncertaintyInterval,
    q: float,
    *,
    alpha: float,
) -> CombinedStatement:
    if enclosure.kind is not GuaranteeKind.SOUND_ENCLOSURE:
        raise TypeError("combine_enclosure_with_conformal needs a sound enclosure")
    half = abs(float(q))
    return CombinedStatement(
        enclosure=enclosure,
        conformal_q=half,
        alpha=float(alpha),
        combined_lo=enclosure.lo - half,
        combined_hi=enclosure.hi + half,
    )


def refuse_conformal_seal(interval: UncertaintyInterval) -> None:
    """Certificate v1 seals enclosures, not statistical intervals."""
    if interval.kind is GuaranteeKind.CONFORMAL:
        raise ValueError("conformal intervals cannot be sealed into certificate v1")
    if interval.kind is GuaranteeKind.MODEL_BASED:
        raise ValueError("model-based intervals cannot be sealed into certificate v1")


def _cover_rate(scores: Sequence[float], q: float) -> float:
    if not scores:
        raise ValueError("scores must be non-empty")
    hits = sum(1 for s in scores if abs(float(s)) <= q)
    return hits / float(len(scores))


def calibration_report(
    residuals: Sequence[float],
    *,
    q: float,
    strata: Mapping[str, Sequence[float]] | None = None,
    alphas: Sequence[float] = (0.2, 0.1, 0.05),
    exchangeability_ok: bool = True,
    shift_detected: bool = False,
) -> CalibrationReport:
    half = 2.0 * abs(float(q))
    cond: dict[str, float] = {}
    if strata is not None:
        cond = {name: _cover_rate(vals, q) for name, vals in strata.items()}
    curve = tuple(
        (float(a), _cover_rate(residuals, split_conformal(residuals, alpha=float(a))))
        for a in alphas
    )
    return CalibrationReport(
        marginal_coverage=_cover_rate(residuals, q),
        conditional_coverage=cond,
        average_width=half,
        width_coverage_curve=curve,
        exchangeability_ok=exchangeability_ok,
        shift_detected=shift_detected,
    )


def worked_example() -> dict[str, float]:
    q = split_conformal(WORKED_RESIDUALS, alpha=0.1)
    return {"q": q, "alpha": 0.1, "n": float(len(WORKED_RESIDUALS))}


def theoretical_coverage(n: int, alpha: float) -> float:
    k = conformal_index(n, alpha)
    if k > n:
        return 1.0
    return k / float(n + 1)


def resample_coverage(
    n: int,
    alpha: float,
    *,
    trials: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        data = [abs(rng.gauss(0.0, 1.0)) for _ in range(n + 1)]
        q = split_conformal(data[:n], alpha=alpha)
        if data[n] <= q:
            hits += 1
    return hits / float(trials)


def shift_diagnostic(
    *,
    n: int = 99,
    seed: int = 0,
    alpha: float = 0.1,
) -> CalibrationReport:
    """G4: calibration N(0,1), test N(0,3). Coverage drops; the report says so."""
    rng = random.Random(seed)
    cal = [abs(rng.gauss(0.0, 1.0)) for _ in range(n)]
    test = [abs(rng.gauss(0.0, 3.0)) for _ in range(n)]
    q = split_conformal(cal, alpha=alpha)
    cover = _cover_rate(test, q)
    shifted = cover < (1.0 - alpha) - 0.15
    return calibration_report(
        test,
        q=q,
        exchangeability_ok=not shifted,
        shift_detected=shifted,
    )


def _abs_normal_cdf(q: float, sigma: float) -> float:
    z = abs(float(q)) / float(sigma)
    return math.erf(z / math.sqrt(2.0))


def adaptive_vs_fixed(
    *,
    seeds: int = 5,
    n_cal: int = 4000,
    n_test: int = 5000,
    alpha: float = 0.1,
) -> dict[str, object]:
    """G3: heteroscedastic mixture; adaptive conditional coverage wins."""
    fixed_devs: list[float] = []
    adapt_devs: list[float] = []
    fixed_widths: list[float] = []
    adapt_widths: list[float] = []
    for seed in range(seeds):
        rng = random.Random(1000 + seed)
        cal_r: list[float] = []
        cal_w: list[float] = []
        for _ in range(n_cal):
            low = rng.random() < 0.5
            scale = 0.1 if low else 1.0
            cal_r.append(rng.gauss(0.0, scale))
            cal_w.append(scale)
        q_fix = split_conformal(cal_r, alpha=alpha)
        q_ad = adaptive_conformal(cal_r, cal_w, alpha=alpha)
        low_fix: list[float] = []
        high_fix: list[float] = []
        low_ad: list[float] = []
        high_ad: list[float] = []
        w_fix = 0.0
        w_ad = 0.0
        for _ in range(n_test):
            low = rng.random() < 0.5
            scale = 0.1 if low else 1.0
            res = rng.gauss(0.0, scale)
            w_fix += 2.0 * q_fix
            w_ad += 2.0 * q_ad * scale
            if low:
                low_fix.append(res)
                low_ad.append(res / scale)
            else:
                high_fix.append(res)
                high_ad.append(res / scale)
        fix_low = _cover_rate(low_fix, q_fix)
        fix_high = _cover_rate(high_fix, q_fix)
        ad_low = _cover_rate(low_ad, q_ad)
        ad_high = _cover_rate(high_ad, q_ad)
        nom = 1.0 - alpha
        fixed_devs.append(max(abs(fix_low - nom), abs(fix_high - nom)))
        adapt_devs.append(max(abs(ad_low - nom), abs(ad_high - nom)))
        fixed_widths.append(w_fix / float(n_test))
        adapt_widths.append(w_ad / float(n_test))
    return {
        "fixed_max_dev": max(fixed_devs),
        "adaptive_max_dev": max(adapt_devs),
        "fixed_mean_dev": sum(fixed_devs) / float(seeds),
        "adaptive_mean_dev": sum(adapt_devs) / float(seeds),
        "fixed_width": sum(fixed_widths) / float(seeds),
        "adaptive_width": sum(adapt_widths) / float(seeds),
        "g3_earned": (
            (sum(adapt_devs) / float(seeds)) <= 0.03
            and max(adapt_devs) <= 0.045
            and max(fixed_devs) >= 0.08
            and (sum(adapt_widths) / float(seeds)) <= (sum(fixed_widths) / float(seeds))
        ),
        "population_fixed_q": 1.2815515655446004,
        "population_adaptive_z": 1.6448536269514722,
        "low_noise_fixed_cover": _abs_normal_cdf(1.2815515655446004, 0.1),
        "high_noise_fixed_cover": _abs_normal_cdf(1.2815515655446004, 1.0),
    }


__all__ = [
    "CalibrationReport",
    "CombinedStatement",
    "DISCLAIMER",
    "GuaranteeKind",
    "UncertaintyInterval",
    "WORKED_RESIDUALS",
    "adaptive_conformal",
    "adaptive_vs_fixed",
    "calibration_report",
    "combine_enclosure_with_conformal",
    "conformal_index",
    "honesty_payload",
    "refuse_conformal_seal",
    "resample_coverage",
    "shift_diagnostic",
    "softplus_width",
    "softplus_width_prime",
    "split_conformal",
    "theoretical_coverage",
    "worked_example",
]

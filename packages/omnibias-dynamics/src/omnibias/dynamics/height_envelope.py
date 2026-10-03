# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Alpha-0 T-h envelope after the shrinking-root x-corridor.

The first-root height continuation uses ``T_h = q/h + k`` and the
comparison

    T_h = 1 + C eps T/h + C eps^3 / h.

At ``C = 0`` this is ``T_h = 1``, so ``T - h`` is conserved and equals
the exit gap ``T_e - h_e = eps^2 x_*^2/2 - eps^3 y0``. For
``eps y0 < x_*^2/2`` the gap is positive (orbit starts above the
parabola ``T = h``). The AM-GM identity

    eps (eps^2 + w^2) - 2 eps^2 w = eps (eps - w)^2

is the pointwise ingredient of ``|q| <= C eps (eps^2 + T)``. At the
matching section the exact leading ratio

    |q| / (eps (eps^2 + T)) = (x - r1)(r2 - x) / (1 + x^2/2)

is independent of ``eps`` and tends to ``x (r2 - x) / (1 + x^2/2)`` as
``r1 -> 0``, so a uniform ``C`` exists.

The ``C eps`` perturbation of the comparison is the already-majorized
factor ``(h/h_e)^{C eps} -> 1``. This is not first-hit of the large
height section: the actual field is not the comparison ODE, and the
transversal event remains unsealed. Not G1 or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.post_corridor import integrating_factor_exponent_majorant

__all__ = [
    "HeightEnvelopeReport",
    "identity_verdicts",
    "q_envelope_ratio",
    "report",
    "residual_amgm_qbound",
    "residual_envelope_exit",
    "residual_q_envelope",
    "residual_th_alpha0",
]

_SAMPLE_R1 = Fraction(1, 5)
_SAMPLE_R2 = Fraction(4)
_SAMPLE_XSTAR = Fraction(1)
_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_Y0 = Fraction(6)
_SAMPLE_HMAX = Fraction(1)
_SAMPLE_MARGIN = Fraction(1, 8)


def _honesty(*, height_envelope_alpha0: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        height_envelope_alpha0=height_envelope_alpha0,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_amgm_qbound(eps: Fraction, w: Fraction) -> Fraction:
    """``eps(eps^2 + w^2) - 2 eps^2 w - eps(eps - w)^2``."""
    return eps * (eps**2 + w**2) - 2 * eps**2 * w - eps * (eps - w) ** 2


def residual_envelope_exit(
    kinetic: Fraction, eps: Fraction, height: Fraction, xstar: Fraction, y0: Fraction
) -> Fraction:
    """``T (1 + eps y0) - (eps^2 + h) x_*^2 / 2`` at the matching exit."""
    return kinetic * (1 + eps * y0) - (eps**2 + height) * xstar**2 / 2


def residual_th_alpha0(
    kinetic: Fraction, height: Fraction, t_exit: Fraction, h_exit: Fraction
) -> Fraction:
    """``(T - h) - (T_e - h_e)``; conserved on the ``C = 0`` comparison."""
    return (kinetic - height) - (t_exit - h_exit)


def residual_q_envelope(
    q_abs: Fraction,
    eps: Fraction,
    kinetic: Fraction,
    xstar: Fraction,
    r1: Fraction,
    r2: Fraction,
) -> Fraction:
    """``|q| (1 + x^2/2) - eps (eps^2 + T) (x - r1)(r2 - x)`` at matching."""
    return q_abs * (1 + xstar**2 / 2) - eps * (eps**2 + kinetic) * (xstar - r1) * (
        r2 - xstar
    )


def q_envelope_ratio(xstar: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """``(x - r1)(r2 - x) / (1 + x^2/2)``; independent of ``eps``."""
    return (xstar - r1) * (r2 - xstar) / (1 + xstar**2 / 2)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    r1, r2, xstar = _SAMPLE_R1, _SAMPLE_R2, _SAMPLE_XSTAR
    eps, y0, hmax = _SAMPLE_EPS, _SAMPLE_Y0, _SAMPLE_HMAX
    t_exit = (eps * xstar) ** 2 / 2
    h_exit = eps**3 * y0
    q_abs = eps**3 * (xstar - r1) * (r2 - xstar)
    t_alpha0 = t_exit + hmax - h_exit
    w = 2 * eps
    return {
        "amgm_qbound": _verdict(residual_amgm_qbound(eps, w)),
        "envelope_exit": _verdict(residual_envelope_exit(t_exit, eps, h_exit, xstar, y0)),
        "th_alpha0": _verdict(residual_th_alpha0(t_alpha0, hmax, t_exit, h_exit)),
        "q_envelope": _verdict(residual_q_envelope(q_abs, eps, t_exit, xstar, r1, r2)),
    }


def _ratio_uniform() -> dict[str, float | bool]:
    """``(x-r1)(r2-x)/(1+x^2/2)`` increases toward a finite limit as ``r1 -> 0``."""
    xstar, r2 = _SAMPLE_XSTAR, _SAMPLE_R2
    samples = []
    for n in (5, 10, 25, 100):
        r1 = Fraction(1, n)
        samples.append(float(q_envelope_ratio(xstar, r1, r2)))
    increasing = all(samples[i] < samples[i + 1] for i in range(len(samples) - 1))
    bounded = samples[-1] < 2.1
    return {
        "ratio_hi": samples[-1],
        "increasing": increasing,
        "uniform_C": increasing and bounded,
    }


def _perturbation_majorant() -> dict[str, float | bool]:
    """Reuse the post-corridor exponent majorant on ``eps = 1/n``, ``n >= 25``."""
    samples = []
    for n in (25, 50, 100):
        eps = Interval.from_rational(Fraction(1, n))
        bound = integrating_factor_exponent_majorant(eps)
        samples.append(bound.hi)
    decreasing = all(samples[i] > samples[i + 1] for i in range(len(samples) - 1))
    small = samples[-1] < 0.6
    below_one = samples[0] < 1.0
    return {
        "majorant_hi": samples[-1],
        "decreasing": decreasing,
        "vanishes": decreasing and small and below_one,
    }


@dataclass(frozen=True)
class HeightEnvelopeReport:
    """Alpha-0 T-h envelope. Not height-section first-hit or G1."""

    identities: Mapping[str, str]
    te_above_he: bool
    gap_below_margin: bool
    ratio: Mapping[str, float | bool]
    perturbation: Mapping[str, float | bool]
    height_envelope_alpha0: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-height-envelope-v1",
            "identities": dict(self.identities),
            "te_above_he": self.te_above_he,
            "gap_below_margin": self.gap_below_margin,
            "ratio": dict(self.ratio),
            "perturbation": dict(self.perturbation),
            "height_envelope_alpha0": self.height_envelope_alpha0,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "C=0 comparison conserves T-h = T_e-h_e; |q|/(eps(eps^2+T)) "
                "at matching is independent of eps and bounded as r1->0. "
                "Not height-section first-hit, actual-field T-h=O(eps), G1, "
                "or Hilbert XVI."
            ),
        }


def report() -> HeightEnvelopeReport:
    """Replay alpha-0 envelope identities and the uniform |q| ratio."""
    identities = identity_verdicts()
    xstar, eps, y0 = _SAMPLE_XSTAR, _SAMPLE_EPS, _SAMPLE_Y0
    t_exit = (eps * xstar) ** 2 / 2
    h_exit = eps**3 * y0
    te_above_he = t_exit > h_exit
    gap = t_exit - h_exit
    gap_below_margin = 0 < gap < _SAMPLE_MARGIN
    try:
        ratio = _ratio_uniform()
        perturbation = _perturbation_majorant()
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and te_above_he
            and gap_below_margin
            and bool(ratio["uniform_C"])
            and bool(perturbation["vanishes"])
        )
    except (ValueError, ZeroDivisionError):
        ratio = {"ratio_hi": float("inf"), "increasing": False, "uniform_C": False}
        perturbation = {
            "majorant_hi": float("inf"),
            "decreasing": False,
            "vanishes": False,
        }
        sealed = False
    return HeightEnvelopeReport(
        identities=identities,
        te_above_he=te_above_he,
        gap_below_margin=gap_below_margin,
        ratio=ratio,
        perturbation=perturbation,
        height_envelope_alpha0=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(height_envelope_alpha0=sealed),
    )

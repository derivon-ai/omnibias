# SPDX-License-Identifier: Apache-2.0
"""Poisson-image small-time SU(2) heat kernels, including both radial endpoints.

Haar mass is one, C_fund=3/4 and Delta_G=(d_uu+2*cot(u)*d_u)/4.
The full image remainder is proved uniformly on 0<t<=5/64 in the cited
API proof. The evaluator keeps exp(-u^2/t) in the logarithm and never
requires the ordinary, possibly underflowed heat kernel. No autodiff.
"""

from __future__ import annotations

from fractions import Fraction as Q
from math import factorial, isfinite
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.verified.interval import Interval as IV
from omnibias.core.verified.transcend import PI_IV, certificate_mode, cos_iv, exp_iv, ln_iv
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

_MAX_TIME = Q(5, 64)
_IMAGE_ERROR = Q(1, 2**99)


def _time(time: int | Q) -> Q:
    t = _rational(time, "time")
    if not 0 < t <= _MAX_TIME:
        raise ValueError("proved small-time domain requires 0<time<=5/64")
    return t


def _bounds(value: IV) -> list[str]:
    if not isfinite(value.lo) or not isfinite(value.hi):
        raise ValueError("finite interval range exhausted at this time or derivative order")
    return [str(Q(value.lo)), str(Q(value.hi))]


def _pad(value: IV, radius: Q) -> IV:
    upper = IV.from_value(radius).hi
    return value + IV(-upper, upper)


def _scope() -> dict[str, bool]:
    return {
        "actual_ks_vacuum_verified": False,
        "wilson_conditional_law_identified": False,
        "lattice_to_continuum_dynamics_verified": False,
        "infinite_volume_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }


def _seal(
    kind: str,
    witness: dict[str, Any],
    earned: dict[str, bool],
    *,
    rational: bool = False,
) -> dict[str, Any]:
    certificate = make_certificate(
        claim="small-time compact SU(2) heat-kernel bounds with analytic all-image remainder",
        payload={"type": kind, "status": "PASS", "witness": witness},
        honesty={**_scope(), **earned, "unconditional_transcendentals": not rational},
        meta={
            "analytic_implication": "docs/api/gauge-stochastic-small-time.md",
            **({"transcend_backend": "not_used"} if rational else {}),
        },
    )
    return {
        "status": "PASS",
        "witness": witness,
        "certificate": certificate,
        **_scope(),
        **earned,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def su2_small_time_score_bounds(time: int | Q) -> dict[str, Any]:
    """Record the proved full-angle time-score theorem at an exact small time.

    u^2/t^2-3/t <= d_t log K_t(u) <= u^2/t^2-3/(2t)+1/2.
    This follows from the analytic image theorem, not from sampled values.
    Its rational certificate checks the theorem's constants/domain; Lean
    has not verified the compact heat-kernel analytic implication.
    """
    t = _time(time)
    algebra = {
        "identity_min_z_lower": str(36 / t),
        "cut_min_z_lower": str(81 / t),
        "identity_y_derivative_polynomial_slack": str(400**5 - 4**10 * factorial(10)),
        "time_derivative_polynomial_slack": str(400**3 - 10 * 4**6 * factorial(6)),
        "cut_y_derivative_polynomial_slack": str(1000**7 - 4**14 * factorial(14)),
        "relative_image_remainder_upper": str(_IMAGE_ERROR),
        "relative_image_y_derivative_upper": str(_IMAGE_ERROR),
        "relative_image_y_second_derivative_upper": str(_IMAGE_ERROR),
        "relative_image_time_derivative_upper": str(_IMAGE_ERROR),
        "log_image_time_derivative_upper": str(2 * _IMAGE_ERROR),
        "cut_principal_score_correction_upper": "3/2",
        "score_lower_angle_squared_coefficient": str(1 / t**2),
        "score_lower_constant": str(-3 / t),
        "score_upper_angle_squared_coefficient": str(1 / t**2),
        "score_upper_constant": str(-Q(3, 2) / t + Q(1, 2)),
        "valid_time_upper": str(_MAX_TIME),
    }
    passed = (
        36 / t >= 400
        and 81 / t >= 1000
        and all(int(algebra[key]) > 0 for key in (
            "identity_y_derivative_polynomial_slack",
            "time_derivative_polynomial_slack",
            "cut_y_derivative_polynomial_slack",
        ))
    )
    if not passed:
        raise ValueError("analytic proof arithmetic did not close")
    return _seal(
        "su2_small_time_score_bounds_v1",
        {
            "inputs": {"time": str(t)},
            "normalization": "Haar mass1; exp(t*Delta_G); C_fund=3/4; Tr(U)=2*cos(u)",
            "angle_quantifier": "every real u in [0,pi], including removable endpoints",
            "uniform_time_score_inequality": "u^2/t^2-3/t <= d_t log K_t(u) <= u^2/t^2-3/(2t)+1/2",
            "proof_register": "written analytic Poisson-image proof plus finite exact arithmetic; not Lean",
            "arithmetic": algebra,
        },
        {"finite_gate_verified": True, "all_image_tail_verified": True,
         "uniform_time_score_verified": True},
        rational=True,
    )


def _even_series(
    argument: IV, derivative: int, offset: int, terms: int, *, alternating: bool = False
) -> tuple[IV, Q]:
    """Enclose derivatives of sum (+/-w)^m/(2m+offset)! including its tail."""
    w = IV(max(0.0, argument.lo), max(0.0, argument.hi))
    total = IV.from_value(0)
    for m in range(terms, derivative - 1, -1):
        coefficient = Q(factorial(m), factorial(m - derivative) * factorial(2 * m + offset))
        if alternating and m % 2:
            coefficient = -coefficient
        total = total * w + IV.from_value(coefficient)
    first = terms + 1
    ratio = IV.from_value(
        Q(first + 1, first + 1 - derivative)
        / ((2 * first + offset + 1) * (2 * first + offset + 2))
    ) * w
    if ratio.hi >= 1:
        raise ValueError("chosen series_terms do not certify the analytic-series tail")
    initial = (
        IV.from_value(Q(factorial(first), factorial(first - derivative) * factorial(2 * first + offset)))
        * w.pow_int(first - derivative)
    )
    tail = initial / (IV.from_value(1) - ratio)
    bound = Q(tail.hi)
    return _pad(total, bound), bound


def _sinc_pack(v: IV, terms: int) -> tuple[IV, IV, IV, list[str]]:
    y = v.pow_int(2)
    pack = [_even_series(y, r, 1, terms, alternating=True) for r in range(3)]
    values = [entry[0] for entry in pack]
    if values[0].lo <= 0:
        raise ValueError("sinc denominator not separated; increase series_terms")
    return values[0], values[1], values[2], [str(entry[1]) for entry in pack]


def su2_small_time_heat_kernel(
    time: int | Q, angle_pi: int | Q, *, series_terms: int = 32
) -> dict[str, Any]:
    """Enclose log K, rescaled K, radial log score and group log-Laplacian.

    The exact input specifies u/pi in [0,1]; u itself is not treated as a
    rational radian value. The two endpoints are handled by their analytic
    limits. Large action u^2/t remains in log K, avoiding kernel underflow.
    The image theorem covers 0<t<=5/64. Extremely small times may exhaust
    binary64 derivative range and are explicitly refused.
    """
    t = _time(time)
    a = _rational(angle_pi, "angle_pi")
    terms = _integer(series_terms, "series_terms")
    if not 0 <= a <= 1 or not 16 <= terms <= 128:
        raise ValueError("angle_pi must lie in [0,1] and series_terms in 16..128")
    source = su2_small_time_score_bounds(t)
    if not replay_small_time_certificate(source["certificate"]):
        raise ValueError("canonical all-image source failed replay")
    with certificate_mode():
        tiv = IV.from_value(t)
        u = PI_IV * IV.from_value(a)
        cut_distance = PI_IV * IV.from_value(1 - a)
        prefactor_log = ln_iv(IV.from_value(2)) + ln_iv(PI_IV) / 2 + tiv / 4 - ln_iv(tiv) * Q(3, 2)
        evidence: dict[str, Any] = {}
        if a <= Q(1, 2):
            chart = "identity hemisphere; one principal image, paired infinite tail"
            v = u
            y = v.pow_int(2)
            sinc, sinc_y, sinc_yy, tails = _sinc_pack(v, terms)
            ratio = sinc_y / sinc
            ly = -1 / tiv - ratio
            lyy = -sinc_yy / sinc + ratio.pow_int(2)
            log_scaled = -ln_iv(sinc)
            principal_log = prefactor_log - u.pow_int(2) / tiv + log_scaled
            principal_score = 2 * v * ly
            principal_lap = (Q(1, 2) + cos_iv(v) / sinc) * ly + y * lyy
            principal_time = Q(1, 4) - Q(3, 2) / tiv + u.pow_int(2) / tiv.pow_int(2)
            evidence["sinc_series_tails"] = tails
        elif 1 - a <= t / 4:
            chart = "cut-locus layer; analytic even hyperbolic pair"
            v = cut_distance
            y = v.pow_int(2)
            sinc, sinc_y, sinc_yy, tails = _sinc_pack(v, terms)
            c = (2 * PI_IV / tiv).pow_int(2)
            w = c * y
            epsilon = tiv / (2 * PI_IV.pow_int(2))
            hyperbolic_s = [_even_series(w, r, 1, terms) for r in range(3)]
            hyperbolic_c = [_even_series(w, r, 0, terms) for r in range(3)]
            f = [hyperbolic_s[r][0] - epsilon * hyperbolic_c[r][0] for r in range(3)]
            if f[0].lo <= 0:
                raise ValueError("principal cut pair not separated; increase series_terms")
            ratio = f[1] / f[0]
            sinc_ratio = sinc_y / sinc
            ly = -1 / tiv - sinc_ratio + c * ratio
            lyy = -sinc_yy / sinc + sinc_ratio.pow_int(2) + c.pow_int(2) * (f[2] / f[0] - ratio.pow_int(2))
            log_scaled = ln_iv(4 * PI_IV.pow_int(2) / tiv) + ln_iv(f[0]) - ln_iv(sinc) - 2 * PI_IV * v / tiv
            principal_log = prefactor_log - u.pow_int(2) / tiv + log_scaled
            principal_score = -2 * v * ly
            principal_lap = (Q(1, 2) + cos_iv(v) / sinc) * ly + y * lyy
            principal_time = (
                Q(1, 4) - Q(5, 2) / tiv + (PI_IV.pow_int(2) + y) / tiv.pow_int(2)
                - (2 * w * f[1] + epsilon * hyperbolic_c[0][0]) / (tiv * f[0])
            )
            evidence["sinc_series_tails"] = tails
            evidence["sinhc_series_tails"] = [str(x[1]) for x in hyperbolic_s]
            evidence["cosh_series_tails"] = [str(x[1]) for x in hyperbolic_c]
        else:
            chart = "cut hemisphere away from endpoint; exponentially factored pair"
            v = cut_distance
            sinc, _, _, tails = _sinc_pack(v, terms)
            sine = v * sinc
            cot_u = -cos_iv(v) / sine
            rate = 4 * PI_IV / tiv
            image_ratio = exp_iv(-rate * v)
            d = 2 * PI_IV - u
            numerator = u - d * image_ratio
            numerator_u = 1 + image_ratio - rate * d * image_ratio
            numerator_uu = (2 * rate - rate.pow_int(2) * d) * image_ratio
            if numerator.lo <= 0:
                raise ValueError("principal image numerator not separated")
            small_score = numerator_u / numerator - cot_u
            principal_score = -2 * u / tiv + small_score
            principal_lap = (
                -(Q(1, 2) + u * cot_u) / tiv
                + (numerator_uu / numerator - small_score.pow_int(2) + 1) / 4
            )
            log_scaled = ln_iv(numerator / sine)
            principal_log = prefactor_log - u.pow_int(2) / tiv + log_scaled
            principal_time = (
                Q(1, 4) - Q(3, 2) / tiv + u.pow_int(2) / tiv.pow_int(2)
                - d * 4 * PI_IV * v * image_ratio / (tiv.pow_int(2) * numerator)
            )
            evidence["sinc_series_tails"] = tails
            evidence["second_image_ratio"] = _bounds(image_ratio)
        log_value = _pad(principal_log, 2 * _IMAGE_ERROR)
        score = _pad(principal_score, 8 * _IMAGE_ERROR) if a not in (0, 1) else IV.from_value(0)
        laplacian = _pad(principal_lap, 16 * _IMAGE_ERROR)
        time_score = _pad(principal_time, 2 * _IMAGE_ERROR)
        # The multiplicative all-image correction is evaluated outward too.
        correction = IV(
            IV.from_value(1 - _IMAGE_ERROR).lo,
            IV.from_value(1 + _IMAGE_ERROR).hi,
        )
        scaled = exp_iv(log_scaled) * correction
        uniform_score = IV(
            (u.pow_int(2) / tiv.pow_int(2) - 3 / tiv).lo,
            (u.pow_int(2) / tiv.pow_int(2) - Q(3, 2) / tiv + Q(1, 2)).hi,
        )
        if time_score.hi < uniform_score.lo or uniform_score.hi < time_score.lo:
            raise ValueError("point interval contradicts the proved uniform time-score bound")
        time_score = IV(max(time_score.lo, uniform_score.lo), min(time_score.hi, uniform_score.hi))
        arithmetic = {
            "angle_radians_enclosure": _bounds(u),
            "cut_distance_over_time_enclosure": _bounds(cut_distance / tiv),
            "log_kernel_enclosure": _bounds(log_value),
            "scaled_kernel_enclosure": _bounds(scaled),
            "radial_log_derivative_enclosure": _bounds(score),
            "group_log_laplacian_enclosure": _bounds(laplacian),
            "time_log_derivative_enclosure": _bounds(time_score),
            "uniform_time_score_enclosure": _bounds(uniform_score),
            "group_squared_log_gradient_enclosure": _bounds(score.pow_int(2) / 4),
            "heat_equation_identity_enclosure": _bounds(time_score - laplacian - score.pow_int(2) / 4),
        }
        return _seal(
            "su2_small_time_heat_kernel_v1",
            {
                "inputs": {"time": str(t), "angle_pi": str(a), "series_terms": terms},
                "normalization": "Haar mass1; Tr(U)=2*cos(u); Delta_G=(d_uu+2*cot(u)*d_u)/4",
                "rescaled_kernel_definition": "t^(3/2)/(2*sqrt(pi))*exp(u^2/t-t/4)*K_t(u)",
                "chart": chart,
                "arithmetic": arithmetic,
                "finite_series": evidence,
                "uniform_source_certificate": source["certificate"],
                "proof_register": "analytic all-image remainder plus directed finite enclosures; not Lean",
            },
            {"finite_gate_verified": True, "all_image_tail_verified": True,
             "log_kernel_enclosure_verified": True, "radial_log_derivative_enclosure_verified": True,
             "group_log_laplacian_enclosure_verified": True, "uniform_time_score_verified": True},
        )


def replay_small_time_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild canonical exact inputs, source, finite series and every scope flag."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        kind = certificate["payload"]["type"]
        inputs = certificate["payload"]["witness"]["inputs"]
        if kind == "su2_small_time_score_bounds_v1":
            report = su2_small_time_score_bounds(Q(inputs["time"]))
        elif kind == "su2_small_time_heat_kernel_v1":
            report = su2_small_time_heat_kernel(
                Q(inputs["time"]), Q(inputs["angle_pi"]), series_terms=inputs["series_terms"]
            )
        else:
            return False
        return bool(report["certificate"] == certificate)
    except (KeyError, ValueError, TypeError, OverflowError, IndexError, RuntimeError, ZeroDivisionError):
        return False


__all__ = [
    "replay_small_time_certificate",
    "su2_small_time_heat_kernel",
    "su2_small_time_score_bounds",
]

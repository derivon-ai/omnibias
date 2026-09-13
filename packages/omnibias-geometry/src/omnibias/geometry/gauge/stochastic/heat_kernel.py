# SPDX-License-Identifier: Apache-2.0
"""All-representation compact-group heat enclosures and a normalized SU(2) bridge.

Haar mass is one and K_t is the kernel of exp(-t C), with fundamental
Casimir3/4 for SU(2),4/3 for SU(3). These kernels are not lattice Wilson
densities or Kogut-Susskind vacuum densities. Finite character sums always
carry an analytic infinite tail. Insufficient tails or denominator
separation return INCONCLUSIVE. Replay accepts canonical failed reports
as reports, without upgrading their gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from math import isfinite
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.verified.complex_interval import ComplexInterval as CI
from omnibias.core.verified.interval import Interval as IV
from omnibias.core.verified.transcend import besseli_iv, certificate_mode, cos_iv, exp_iv, sin_iv
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def _bounds(x: IV) -> list[str]:
    if not isfinite(x.lo) or not isfinite(x.hi):
        raise ValueError("finite interval arithmetic exhausted; increase time or reduce cutoff")
    return [str(Q(x.lo)), str(Q(x.hi))]


def _time_cutoff(time: int | Q, cutoff: int) -> tuple[Q, int]:
    t = _rational(time, "time")
    n = _integer(cutoff, "cutoff")
    if t <= 0 or not 0 <= n <= 256:
        raise ValueError("time must be positive and cutoff must lie in0..256")
    return t, n


def _scope() -> dict[str, bool]:
    return {
        "wilson_to_heat_kernel_error_verified": False,
        "lattice_to_continuum_dynamics_verified": False,
        "actual_ks_vacuum_verified": False,
        "infinite_volume_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }


def _seal(
    kind: str, witness: dict[str, Any], earned: dict[str, bool], *, rational: bool = False
) -> dict[str, Any]:
    status = "PASS" if earned["finite_gate_verified"] else "INCONCLUSIVE"
    meta = {"analytic_implication": "docs/api/gauge-stochastic-heat-kernel.md"}
    if rational:
        meta["transcend_backend"] = "not_used"
    certificate = make_certificate(
        claim="compact heat-kernel arithmetic and explicitly conditional probability comparison",
        payload={"type": kind, "status": status, "witness": witness},
        honesty={**_scope(), **earned, "unconditional_transcendentals": not rational},
        meta=meta,
    )
    return {
        "status": status,
        "witness": witness,
        "certificate": certificate,
        **_scope(),
        **earned,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def _tail(
    time: Q, cutoff: int, *, group: str, derivative: int = 0
) -> tuple[IV | None, dict[str, Any]]:
    first = cutoff + 1
    if group == "SU2":
        power, shift = 2 * derivative + 2, 1
        divisor = 1
        for j in range(1, derivative + 1):
            divisor *= 2 * j + 1
        energy = Q(first * (first + 2), 4)
        step = Q(2 * first + 3, 4)
    else:
        power, shift, divisor = 7, 2, 64
        energy = Q(first * first, 4) + first
        step = Q(2 * first + 5, 4)
    ratio = IV.from_value(Q(first + shift + 1, first + shift) ** power) * exp_iv(
        IV.from_value(-time * step)
    )
    first_bound = IV.from_value(Q((first + shift) ** power, divisor)) * exp_iv(
        IV.from_value(-time * energy)
    )
    bound = first_bound / (IV.from_value(1) - ratio) if ratio.hi < 1 else None
    return bound, {
        "first_omitted_label": first,
        "majorant_power": power,
        "majorant_shift": shift,
        "majorant_divisor": divisor,
        "first_term_upper": _bounds(first_bound)[1],
        "decreasing_ratio_enclosure": _bounds(ratio),
        "tail_upper": _bounds(bound)[1] if bound is not None else None,
        "all_representation_tail_verified": bound is not None,
    }


def _character_max(n: int, derivative: int) -> Q:
    if n < derivative:
        return Q(0)
    value = Q(n + 1)
    for j in range(1, derivative + 1):
        value *= Q((n + 1) ** 2 - j * j, 2 * j + 1)
    return value


def _su2_sum(time: Q, x: Q, cutoff: int, derivative: int) -> IV:
    total = IV.from_value(0)
    previous = [IV.from_value(0) for _ in range(derivative + 1)]
    current = [IV.from_value(1), *[IV.from_value(0) for _ in range(derivative)]]
    xiv = IV.from_value(x)
    for n in range(cutoff + 1):
        if abs(x) == 1:
            character = IV.from_value(
                _character_max(n, derivative) * ((-1) ** (n + derivative) if x == -1 else 1)
            )
        else:
            cap = IV.from_value(_character_max(n, derivative)).hi
            character = IV(max(-cap, current[derivative].lo), min(cap, current[derivative].hi))
        total += IV.from_value(n + 1) * exp_iv(IV.from_value(-time * Q(n * (n + 2), 4))) * character
        following = [
            2 * xiv * current[r] - previous[r] + (2 * r * current[r - 1] if r else 0)
            for r in range(derivative + 1)
        ]
        previous, current = current, following
    return total


def _su2_value(time: Q, x: Q, cutoff: int, derivative: int) -> tuple[IV | None, dict[str, Any]]:
    finite = _su2_sum(time, x, cutoff, derivative)
    tail, evidence = _tail(time, cutoff, group="SU2", derivative=derivative)
    value = finite + IV(-tail.hi, tail.hi) if tail is not None else None
    if derivative == 0 and value is not None:
        value = IV(max(0.0, value.lo), value.hi)
    evidence.update(
        {
            "finite_sum_enclosure": _bounds(finite),
            "value_enclosure": _bounds(value) if value is not None else None,
        }
    )
    return value, evidence


def su2_heat_kernel_enclosure(
    time: int | Q, half_trace: int | Q, *, cutoff: int = 32, derivative_order: int = 0
) -> dict[str, Any]:
    """Enclose K_t and its first two derivatives in x=Tr(U)/2 at an exact x.

    The derivative is a class-coordinate derivative, not a Lie-group gradient.
    PASS means a finite enclosure including every omitted spin; it need not
    separate the strictly positive kernel from zero numerically.
    """
    t, n = _time_cutoff(time, cutoff)
    x = _rational(half_trace, "half_trace")
    r = _integer(derivative_order, "derivative_order")
    if not -1 <= x <= 1 or r not in (0, 1, 2):
        raise ValueError("half_trace must lie in[-1,1] and derivative_order in0..2")
    with certificate_mode():
        value, evidence = _su2_value(t, x, n, r)
        return _seal(
            "su2_heat_kernel_enclosure_v1",
            {
                "inputs": {
                    "time": str(t),
                    "half_trace": str(x),
                    "cutoff": n,
                    "derivative_order": r,
                },
                "normalization": "Haar mass1; exp(-t*C); C_n=n(n+2)/4; K_t=sum(n+1)exp(-t*C_n)U_n(x)",
                "arithmetic": evidence,
            },
            {
                "finite_gate_verified": value is not None,
                "all_spin_enclosure_verified": value is not None,
            },
        )


def _quaternion(values: Sequence[int | Q], name: str) -> tuple[Q, Q, Q, Q]:
    if len(values) != 4:
        raise ValueError(f"{name} must contain four rational quaternion components")
    q = tuple(_rational(v, name) for v in values)
    if sum(v * v for v in q) != 1:
        raise ValueError(f"{name} must have exact unit norm")
    return q[0], q[1], q[2], q[3]


def su2_heat_kernel_bridge(
    time: int | Q, x: Sequence[int | Q], boundary: Sequence[int | Q], *, cutoff: int = 32
) -> dict[str, Any]:
    """Enclose a density of the first of five heat-kernel increments given B.

    Exactly normalized law: K_t(X) K_(4t)(X^-1 B) / K_(5t)(B) dX.
    Exact rational unit quaternions avoid an unverified group-membership
    premise. The bridge is conjugation covariant even at B=-I.
    """
    t, n = _time_cutoff(time, cutoff)
    a, b = _quaternion(x, "x"), _quaternion(boundary, "boundary")
    relative = sum((a[i] * b[i] for i in range(4)), Q(0))
    with certificate_mode():
        left, e_left = _su2_value(t, a[0], n, 0)
        right, e_right = _su2_value(4 * t, relative, n, 0)
        denominator, e_den = _su2_value(5 * t, b[0], n, 0)
        passed = (
            left is not None
            and right is not None
            and denominator is not None
            and denominator.lo > 0
        )
        value = (
            left * right / denominator
            if passed and left is not None and right is not None and denominator is not None
            else None
        )
        if value is not None:
            value = IV(max(0.0, value.lo), value.hi)
        return _seal(
            "su2_heat_kernel_bridge_v1",
            {
                "inputs": {
                    "time": str(t),
                    "x": [str(v) for v in a],
                    "boundary": [str(v) for v in b],
                    "cutoff": n,
                },
                "normalization": "K_t(X)*K_(4t)(X^-1*B)/K_(5t)(B) relative to normalized Haar dX",
                "normalization_proof": "character orthogonality gives K_t*K_s=K_(t+s); positive compact-group heat kernel",
                "arithmetic": {
                    "relative_half_trace": str(relative),
                    "density_enclosure": _bounds(value) if value is not None else None,
                },
                "left_kernel": e_left,
                "right_kernel": e_right,
                "normalizing_kernel": e_den,
            },
            {
                "finite_gate_verified": passed,
                "density_enclosure_verified": passed,
                "exact_bridge_normalization_verified": True,
                "conjugation_covariance_verified": True,
                "five_increment_conditional_law_verified": True,
            },
        )


def su3_heat_kernel_torus_enclosure(
    time: int | Q, angles: Sequence[int | Q], *, cutoff: int = 16
) -> dict[str, Any]:
    """Enclose the full SU(3) kernel on diag(e^ia,e^ib,e^-i(a+b)).

    Angles are exact radians. Every conjugacy class is represented on this
    torus. Retained labels satisfy p+q<=cutoff; the remainder covers every
    other irreducible representation, including multiplicities d_(p,q).
    """
    t, n = _time_cutoff(time, cutoff)
    if len(angles) != 2:
        raise ValueError("two exact angles are required")
    a, b = (_rational(v, "angle") for v in angles)
    with certificate_mode():
        z = CI.zero()
        for angle in (a, b, -a - b):
            arg = IV.from_value(angle)
            z += CI(cos_iv(arg), sin_iv(arg))
        h = [CI.one()]
        for degree in range(1, n + 2):
            h.append(
                z * h[-1]
                - (z.conj() * h[-2] if degree >= 2 else CI.zero())
                + (h[-3] if degree >= 3 else CI.zero())
            )
        total = IV.from_value(0)
        for p in range(n + 1):
            for q in range(n + 1 - p):
                dim = (p + 1) * (q + 1) * (p + q + 2) // 2
                character = h[p + q] * h[q] - (h[p + q + 1] * h[q - 1] if q else CI.zero())
                cap = IV.from_value(dim).hi
                real = IV(max(-cap, character.re.lo), min(cap, character.re.hi))
                energy = Q(p * p + q * q + p * q + 3 * p + 3 * q, 3)
                total += IV.from_value(dim) * exp_iv(IV.from_value(-t * energy)) * real
        tail, evidence = _tail(t, n, group="SU3")
        value = total + IV(-tail.hi, tail.hi) if tail is not None else None
        if value is not None:
            value = IV(max(0.0, value.lo), value.hi)
        evidence.update(
            {
                "retained_representation_count": (n + 1) * (n + 2) // 2,
                "finite_sum_enclosure": _bounds(total),
                "value_enclosure": _bounds(value) if value is not None else None,
            }
        )
        return _seal(
            "su3_heat_kernel_torus_v1",
            {
                "inputs": {"time": str(t), "angles": [str(a), str(b)], "cutoff": n},
                "normalization": "Haar mass1; C_(p,q)=(p^2+q^2+pq+3p+3q)/3; K_t=sum d_(p,q)exp(-t*C_(p,q))*chi_(p,q)",
                "character_identity": "Jacobi-Trudi chi_(p,q)=h_(p+q)*h_q-h_(p+q+1)*h_(q-1); h_k=z*h_(k-1)-conj(z)*h_(k-2)+h_(k-3)",
                "arithmetic": evidence,
            },
            {
                "finite_gate_verified": value is not None,
                "all_representation_enclosure_verified": value is not None,
            },
        )


def normalized_product_tv_bound(
    relative_factor_bounds: Sequence[Sequence[int | Q]],
) -> dict[str, Any]:
    """Prove a conditional TV implication from pointwise factor-ratio bounds.

    The user supplies 0<a_i<=f_i/g_i<=b_i on a COMMON support and base
    measure. This function verifies the implication's arithmetic, not these
    analytic premises. TV uses sup_A|mu(A)-nu(A)|=half the L1 difference.
    """
    lower, upper = Q(1), Q(1)
    factors: list[list[str]] = []
    for pair in relative_factor_bounds:
        if len(pair) != 2:
            raise ValueError("each factor bound must be a pair")
        lo, hi = (_rational(v, "factor ratio") for v in pair)
        if not 0 < lo <= hi:
            raise ValueError("factor ratios require0<lower<=upper")
        lower *= lo
        upper *= hi
        factors.append([str(lo), str(hi)])
    tv = (upper - lower) / (upper + lower)
    return _seal(
        "normalized_product_tv_v1",
        {
            "inputs": {"relative_factor_bounds": factors},
            "external_premises": [
                "common base measure and common support",
                "positive integrable products with nonzero finite normalizers",
                "each pointwise factor ratio lies in its supplied interval",
            ],
            "arithmetic": {
                "product_ratio_lower": str(lower),
                "product_ratio_upper": str(upper),
                "normalizer_ratio_enclosure": [str(lower), str(upper)],
                "normalized_density_ratio_enclosure": [str(lower / upper), str(upper / lower)],
                "total_variation_upper": str(tv),
                "bounded_observable_error_multiplier": str(2 * tv),
            },
        },
        {
            "finite_gate_verified": True,
            "conditional_tv_implication_verified": True,
            "pointwise_factor_bounds_verified": False,
            "actual_measure_distance_verified": False,
        },
        rational=True,
    )


def su2_wilson_heat_kernel_comparison(
    time: int | Q, convolution_steps: int, *, cutoff: int = 32, face_count: int = 1
) -> dict[str, Any]:
    """Certify a uniform error of actual m-fold Wilson convolution versus K_t.

    Wilson density is exp(beta*Tr(U)/2)/Z, beta=2m/t. All omitted spins
    are bounded with the positive Bessel-series ratio. When the uniform
    relative error is below one, a TV bound also applies to any finite family
    of face holonomies carrying these convolved Wilson and heat-kernel
    factors on a common product-Haar configuration space. No identification
    with a multidimensional lattice subdivision is asserted.
    """
    t, n = _time_cutoff(time, cutoff)
    m = _integer(convolution_steps, "convolution_steps")
    faces = _integer(face_count, "face_count")
    if not 1 <= m <= 1024 or not 1 <= faces <= 10000:
        raise ValueError("convolution_steps must lie in1..1024 and face_count in1..10000")
    beta = 2 * m / t
    if beta > 600:
        raise ValueError(
            "beta=2*convolution_steps/time exceeds the directed Bessel backend scope600"
        )
    with certificate_mode():
        argument = IV.from_value(beta)
        denominator = besseli_iv(1, argument)
        if denominator.lo <= 0:
            raise ValueError("Bessel normalizer could not be separated from zero")
        heat_tail, heat_evidence = _tail(t, n, group="SU2")
        finite_error = IV.from_value(0)
        heat_nonconstant = IV.from_value(0)
        coefficients: list[dict[str, Any]] = []
        for label in range(1, n + 2):
            a = besseli_iv(label + 1, argument) / denominator
            a = IV(max(0.0, a.lo), min(1.0, a.hi))
            convolved = a.pow_int(m)
            if label <= n:
                heat = exp_iv(IV.from_value(-t * Q(label * (label + 2), 4)))
                error = IV.from_value((label + 1) ** 2) * IV.from_value(Q((convolved - heat).mag))
                finite_error += error
                heat_nonconstant += IV.from_value((label + 1) ** 2) * heat
                coefficients.append(
                    {
                        "doubled_spin": label,
                        "wilson_multiplier": _bounds(a),
                        "convolved_multiplier": _bounds(convolved),
                        "heat_multiplier": _bounds(heat),
                        "weighted_error_upper": _bounds(error)[1],
                    }
                )
            else:
                first = IV.from_value((label + 1) ** 2) * convolved
        first_label = n + 1
        ratio = Q(first_label + 2, first_label + 1) ** 2 * (beta / (2 * (first_label + 2))) ** m
        wilson_tail = first / IV.from_value(1 - ratio) if ratio < 1 else None
        error_iv = (
            finite_error + heat_tail + wilson_tail
            if heat_tail is not None and wilson_tail is not None
            else None
        )
        lower = (
            (IV.from_value(1) - heat_nonconstant - heat_tail).lo if heat_tail is not None else None
        )
        relative = (
            (error_iv / IV.from_value(Q(lower))).hi
            if error_iv is not None and lower is not None and lower > 0
            else None
        )
        product_report = None
        if relative is not None and relative < 1:
            e = Q(relative)
            product_report = normalized_product_tv_bound([[1 - e, 1 + e]] * faces)
        passed = error_iv is not None
        return _seal(
            "su2_wilson_heat_kernel_comparison_v1",
            {
                "inputs": {
                    "time": str(t),
                    "convolution_steps": m,
                    "cutoff": n,
                    "face_count": faces,
                },
                "normalization": "normalized Haar; w_beta=exp(beta*Tr/2)/Z; beta=2m/t; a_n=I_(n+1)(beta)/I_1(beta)",
                "measure_scope": "products of m-fold CONVOLVED Wilson face factors compared with K_t factors; arbitrary common face holonomy maps; not a 3D/4D refinement identity",
                "arithmetic": {
                    "beta": str(beta),
                    "uniform_density_error_upper": _bounds(error_iv)[1]
                    if error_iv is not None
                    else None,
                    "single_group_total_variation_upper": str(min(Q(1), Q(error_iv.hi) / 2))
                    if error_iv is not None
                    else None,
                    "uniform_heat_kernel_lower": str(Q(lower)) if lower is not None else None,
                    "uniform_relative_error_upper": str(Q(relative))
                    if relative is not None
                    else None,
                    "finite_face_total_variation_upper": product_report["witness"]["arithmetic"][
                        "total_variation_upper"
                    ]
                    if product_report is not None
                    else None,
                },
                "retained_coefficients": coefficients,
                "heat_tail": heat_evidence,
                "wilson_tail": {
                    "first_omitted_label": first_label,
                    "first_term_upper": _bounds(first)[1],
                    "decreasing_ratio_upper": str(ratio),
                    "tail_upper": _bounds(wilson_tail)[1] if wilson_tail is not None else None,
                    "proof": "positive Bessel series gives I_(j+1)(beta)/I_j(beta)<=beta/(2(j+1)) for every integer j>=1",
                },
                "product_implication_certificate": product_report["certificate"]
                if product_report is not None
                else None,
            },
            {
                "finite_gate_verified": passed,
                "wilson_to_heat_kernel_error_verified": passed,
                "actual_single_group_distance_verified": passed,
                "finite_coarse_product_comparison_verified": product_report is not None,
                "uniform_in_refinement_steps_rate_verified": False,
                "multidimensional_wilson_refinement_verified": False,
            },
        )


def replay_heat_kernel_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild all inputs, series, tails and gates, including INCONCLUSIVE reports."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        kind = certificate["payload"]["type"]
        inputs = certificate["payload"]["witness"]["inputs"]
        if kind == "su2_heat_kernel_enclosure_v1":
            report = su2_heat_kernel_enclosure(
                Q(inputs["time"]),
                Q(inputs["half_trace"]),
                cutoff=inputs["cutoff"],
                derivative_order=inputs["derivative_order"],
            )
        elif kind == "su2_heat_kernel_bridge_v1":
            report = su2_heat_kernel_bridge(
                Q(inputs["time"]),
                [Q(x) for x in inputs["x"]],
                [Q(x) for x in inputs["boundary"]],
                cutoff=inputs["cutoff"],
            )
        elif kind == "su3_heat_kernel_torus_v1":
            report = su3_heat_kernel_torus_enclosure(
                Q(inputs["time"]), [Q(x) for x in inputs["angles"]], cutoff=inputs["cutoff"]
            )
        elif kind == "normalized_product_tv_v1":
            report = normalized_product_tv_bound(
                [[Q(x) for x in pair] for pair in inputs["relative_factor_bounds"]]
            )
        elif kind == "su2_wilson_heat_kernel_comparison_v1":
            report = su2_wilson_heat_kernel_comparison(
                Q(inputs["time"]),
                inputs["convolution_steps"],
                cutoff=inputs["cutoff"],
                face_count=inputs["face_count"],
            )
        else:
            return False
        return bool(report["certificate"] == certificate)
    except (
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        IndexError,
        RuntimeError,
    ):
        return False


__all__ = [
    "normalized_product_tv_bound",
    "replay_heat_kernel_certificate",
    "su2_heat_kernel_bridge",
    "su2_heat_kernel_enclosure",
    "su2_wilson_heat_kernel_comparison",
    "su3_heat_kernel_torus_enclosure",
]

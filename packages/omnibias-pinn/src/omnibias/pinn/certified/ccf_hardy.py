# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Whole-line CCF CAP attempt on the Cauchy-Hardy exact-Hilbert basis.

Upgrades the collocation-only Poisson CAP to a Hardy ansatz with
``alpha = 1/(1+lambda)``, an interval residual covering of the line, and a
Newton-Kantorovich attempt in ``ell^1_nu`` sequence space.  ``whole_line_certified``
flips only when both the residual gate and the radii-polynomial close;
otherwise the certificate stays ``BLOCKED`` with a quantified gap.

This is a 1D nonlocal-transport model result — never a 3D Navier-Stokes claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Sequence
from typing import Any

import numpy as np
from omnibias.core.verified.hardy_line import (
    hardy_even,
    hardy_even_dalpha,
    hardy_even_deriv,
    hardy_even_deriv_iv,
    hardy_even_iv,
    hardy_even_profile,
    hardy_even_profile_deriv,
    hardy_odd,
    hardy_odd_dalpha,
    hardy_odd_deriv,
    hardy_odd_deriv_iv,
    hardy_odd_iv,
    hardy_tail_constant,
    hilbert_hardy_even_profile,
    hilbert_hardy_even_profile_deriv,
)
from omnibias.core.verified.interval import Interval, sum_intervals
from omnibias.core.verified.kantorovich import radii_polynomial_certificate
from omnibias.core.verified.linalg import (
    identity_matrix,
    inf_norm_matrix,
    inf_norm_vector,
    mat_sub,
    matmul,
    matvec,
    neumann_inverse_norm_bound,
    to_interval_matrix,
)
from omnibias.core.verified.line import hilbert_tail_bound
from omnibias.core.verified.sequence_space import geometric_tail_bound
from omnibias.pinn.certified.navier_stokes import (
    default_ccf_collocation_nodes,
    interval_from_bounds,
    radii_polynomial_closure,
)

CCF_HARDY_WHOLELINE_SCHEMA_VERSION = "navier-stokes-ccf-hardy-wholeline-blowup-attempt-1"
_FORMS = ("transport", "flux", "vorticity")


def _sha256_json(body: dict[str, Any]) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def alpha_from_lambda(lam: float) -> float:
    if lam <= -1.0:
        raise ValueError("lambda must be > -1 for alpha = 1/(1+lambda)")
    return 1.0 / (1.0 + float(lam))


def _omega_fields_point(
    y: float,
    coeffs: Sequence[float],
    scales: Sequence[float],
    gammas: Sequence[float],
) -> tuple[Interval, Interval, Interval, Interval]:
    """Interval ``(Omega, Omega_y, U, U_y)`` for odd Hardy-Ω sum at a point."""
    om = Interval.point(0.0)
    omy = Interval.point(0.0)
    u = Interval.point(0.0)
    uy = Interval.point(0.0)
    for c, a, g in zip(coeffs, scales, gammas, strict=True):
        c_iv = Interval.point(float(c))
        om = om + c_iv * hardy_odd(y, float(a), float(g))
        omy = omy + c_iv * Interval.point(float(g)) * hardy_even(y, float(a), float(g) + 1.0)
        uy = uy + c_iv * (-hardy_even(y, float(a), float(g)))
        if abs(float(g) - 1.0) < 1e-12:
            u = u + c_iv * (-Interval.point(math.atan(y / float(a))))
        else:
            u = u + c_iv * (
                -hardy_odd(y, float(a), float(g) - 1.0) / Interval.point(float(g) - 1.0)
            )
    return om, omy, u, uy


def _vorticity_residual_interval(
    coeffs: Sequence[float],
    scales: Sequence[float],
    gammas: Sequence[float],
    lam: float,
    y: float,
) -> Interval:
    om, omy, u, uy = _omega_fields_point(y, coeffs, scales, gammas)
    y_iv = Interval.point(y)
    lam_iv = Interval.point(lam)
    return om + ((Interval.point(1.0) + lam_iv) * y_iv - u) * omy - om * uy


def _hardy_residual_interval(
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    y: float,
    form: str,
    s: float,
    gammas: Sequence[float] | None = None,
) -> Interval:
    if form == "vorticity":
        alpha = alpha_from_lambda(lam)
        gs = list(gammas) if gammas is not None else [alpha] * len(coeffs)
        return _vorticity_residual_interval(coeffs, scales, gs, lam, y)
    alpha = alpha_from_lambda(lam)
    th = hardy_even_profile(y, coeffs, scales, alpha)
    thp = hardy_even_profile_deriv(y, coeffs, scales, alpha)
    hth = hilbert_hardy_even_profile(y, coeffs, scales, alpha)
    one = Interval.point(1.0)
    lam_iv = Interval.point(lam)
    y_iv = Interval.point(y)
    s_iv = Interval.point(s)
    linear = (one + lam_iv) * y_iv * thp - lam_iv * th
    if form == "transport":
        return linear + s_iv * hth * thp
    hthp = hilbert_hardy_even_profile_deriv(y, coeffs, scales, alpha)
    return linear + s_iv * (thp * hth + th * hthp)


def _hardy_node_system(
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    y: float,
    form: str,
    s: float,
) -> tuple[Interval, list[Interval], Interval]:
    """Interval ``(E, dE/d(free), sum|Hess|)`` with free = ``(c_2..c_n, lambda)``."""
    n = len(scales)
    alpha = alpha_from_lambda(lam)
    dalpha = Interval.point(-1.0 / (1.0 + lam) ** 2)
    th = hardy_even_profile(y, coeffs, scales, alpha)
    thp = hardy_even_profile_deriv(y, coeffs, scales, alpha)
    hth = hilbert_hardy_even_profile(y, coeffs, scales, alpha)
    hthp = hilbert_hardy_even_profile_deriv(y, coeffs, scales, alpha)
    one = Interval.point(1.0)
    lam_iv = Interval.point(lam)
    y_iv = Interval.point(y)
    s_iv = Interval.point(s)
    linear = (one + lam_iv) * y_iv * thp - lam_iv * th
    if form == "transport":
        e_iv = linear + s_iv * hth * thp
    else:
        e_iv = linear + s_iv * (thp * hth + th * hthp)

    pk = [hardy_even(y, a, alpha) for a in scales]
    dpk = [hardy_even_deriv(y, a, alpha) for a in scales]
    qk = [hardy_odd(y, a, alpha) for a in scales]
    dqk = [hardy_odd_deriv(y, a, alpha) for a in scales]
    dpk_da = [hardy_even_dalpha(y, a, alpha) for a in scales]
    # P' w.r.t alpha: d/da (-alpha Q_{a,alpha+1}) is awkward; use
    # d/dalpha of P' via product on the defining identity numerically-free:
    # P'_alpha = -Q_{alpha+1} - alpha * dQ_{alpha+1}/dalpha
    ddpk_da = [
        -hardy_odd(y, a, alpha + 1.0)
        - Interval.point(alpha) * hardy_odd_dalpha(y, a, alpha + 1.0)
        for a in scales
    ]
    dqk_da = [hardy_odd_dalpha(y, a, alpha) for a in scales]
    ddqk_da = [
        hardy_even(y, a, alpha + 1.0)
        + Interval.point(alpha) * hardy_even_dalpha(y, a, alpha + 1.0)
        for a in scales
    ]

    row: list[Interval] = []
    for k in range(1, n):
        base = (one + lam_iv) * y_iv * dpk[k] - lam_iv * pk[k]
        if form == "transport":
            nl = s_iv * (qk[k] * thp + hth * dpk[k])
        else:
            nl = s_iv * (dpk[k] * hth + thp * qk[k] + pk[k] * hthp + th * dqk[k])
        row.append(base + nl)

    # dE/dlam = explicit + through alpha
    th_a = sum_intervals(
        [Interval.point(float(c)) * dpk_da[i] for i, c in enumerate(coeffs)]
    )
    thp_a = sum_intervals(
        [Interval.point(float(c)) * ddpk_da[i] for i, c in enumerate(coeffs)]
    )
    hth_a = sum_intervals(
        [Interval.point(float(c)) * dqk_da[i] for i, c in enumerate(coeffs)]
    )
    hthp_a = sum_intervals(
        [Interval.point(float(c)) * ddqk_da[i] for i, c in enumerate(coeffs)]
    )
    explicit = y_iv * thp - th
    through_alpha = (
        (one + lam_iv) * y_iv * thp_a - lam_iv * th_a
    ) * dalpha
    if form == "transport":
        through_alpha = through_alpha + s_iv * (hth_a * thp + hth * thp_a) * dalpha
    else:
        through_alpha = through_alpha + s_iv * (
            thp_a * hth + thp * hth_a + th_a * hthp + th * hthp_a
        ) * dalpha
    row.append(explicit + through_alpha)

    hess_terms: list[Interval] = []
    free = range(1, n)
    for k in free:
        for j in free:
            if form == "transport":
                hkj = s_iv * (qk[k] * dpk[j] + qk[j] * dpk[k])
            else:
                hkj = s_iv * (
                    dpk[k] * qk[j] + dpk[j] * qk[k] + pk[k] * dqk[j] + pk[j] * dqk[k]
                )
            hess_terms.append(hkj.abs())
    for k in free:
        clam = (y_iv * dpk[k] - pk[k]).abs()
        hess_terms.append(clam)
        hess_terms.append(clam)
    # crude curvature contribution from alpha(lambda) (outward)
    hess_terms.append(through_alpha.abs())
    hess_abs = sum_intervals(hess_terms) if hess_terms else Interval.point(0.0)
    return e_iv, row, hess_abs


def _hardy_residual_over_y_interval(
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    y_iv: Interval,
    form: str,
    s: float,
    gammas: Sequence[float] | None = None,
) -> Interval:
    """Sound enclosure of ``E(y)`` for all ``y`` in ``y_iv`` (interval arithmetic)."""
    if form == "vorticity":
        alpha = alpha_from_lambda(lam)
        gs = list(gammas) if gammas is not None else [alpha] * len(coeffs)
        om = Interval.point(0.0)
        omy = Interval.point(0.0)
        u = Interval.point(0.0)
        uy = Interval.point(0.0)
        for c, a, g in zip(coeffs, scales, gs, strict=True):
            cc = Interval.point(float(c))
            gg = float(g)
            aa = float(a)
            om = om + cc * hardy_odd_iv(y_iv, aa, gg)
            omy = omy + cc * Interval.point(gg) * hardy_even_iv(y_iv, aa, gg + 1.0)
            uy = uy + cc * (-hardy_even_iv(y_iv, aa, gg))
            if abs(gg - 1.0) < 1e-12:
                # atan(y/a) over y_iv: outbound via endpoints
                lo = float(y_iv.lo)
                hi = float(y_iv.hi)
                atan_lo = math.atan(lo / aa)
                atan_hi = math.atan(hi / aa)
                atan_iv = Interval.hull(atan_lo, atan_hi)
                u = u + cc * (-atan_iv)
            else:
                u = u + cc * (
                    -hardy_odd_iv(y_iv, aa, gg - 1.0) / Interval.point(gg - 1.0)
                )
        lam_iv = Interval.point(lam)
        return om + ((Interval.point(1.0) + lam_iv) * y_iv - u) * omy - om * uy
    alpha = alpha_from_lambda(lam)
    th = Interval.point(0.0)
    thp = Interval.point(0.0)
    hth = Interval.point(0.0)
    hthp = Interval.point(0.0)
    for c, a in zip(coeffs, scales, strict=True):
        cc = Interval.point(float(c))
        th = th + cc * hardy_even_iv(y_iv, float(a), alpha)
        thp = thp + cc * hardy_even_deriv_iv(y_iv, float(a), alpha)
        hth = hth + cc * hardy_odd_iv(y_iv, float(a), alpha)
        if form == "flux":
            hthp = hthp + cc * hardy_odd_deriv_iv(y_iv, float(a), alpha)
    one = Interval.point(1.0)
    lam_iv = Interval.point(lam)
    s_iv = Interval.point(s)
    linear = (one + lam_iv) * y_iv * thp - lam_iv * th
    if form == "transport":
        return linear + s_iv * hth * thp
    return linear + s_iv * (thp * hth + th * hthp)


def _certified_residual_sup(
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    form: str,
    s: float,
    yt: float,
    *,
    n_cells: int = 64,
    max_depth: int = 6,
    width_tol: float = 1e-12,
    gammas: Sequence[float] | None = None,
) -> tuple[float, float, float, int, bool]:
    """Return ``(core_sup, far_hi, certified_sup, leaves, between_node_certified)``.

    Uses adaptive Interval-``y`` covering. When the pointwise residual is already
    large, the enclosure is still sound but may be loose; callers should only
    expect a tight gate after a polished near-zero profile.
    """
    # Fast sampled probe — if clearly above any reasonable gate, use a coarse
    # covering (still sound via Interval cells at max_depth) with fewer cells.
    probe = 0.0
    for i in range(65):
        y = yt * i / 64.0
        probe = max(
            probe,
            _hardy_residual_interval(
                coeffs, scales, lam, y, form, s, gammas=gammas
            ).mag,
        )
    cells = n_cells if probe < 1e-4 else min(n_cells, 32)
    depth_cap = max_depth if probe < 1e-4 else min(max_depth, 4)

    stack: list[tuple[float, float, int]] = [
        (yt * i / cells, yt * (i + 1) / cells, 0) for i in range(cells)
    ]
    core = 0.0
    leaves = 0
    while stack:
        lo, hi, depth = stack.pop()
        y_iv = Interval.hull(lo, hi)
        try:
            e_iv = _hardy_residual_over_y_interval(
                coeffs, scales, lam, y_iv, form, s, gammas=gammas
            )
        except (ValueError, ZeroDivisionError):
            if depth >= depth_cap:
                # fall back to endpoint samples (still may under-estimate — mark)
                core = max(
                    core,
                    _hardy_residual_interval(
                        coeffs, scales, lam, lo, form, s, gammas=gammas
                    ).mag,
                    _hardy_residual_interval(
                        coeffs, scales, lam, hi, form, s, gammas=gammas
                    ).mag,
                )
                leaves += 1
                continue
            mid = 0.5 * (lo + hi)
            stack.append((lo, mid, depth + 1))
            stack.append((mid, hi, depth + 1))
            continue
        if (e_iv.width > max(width_tol, 1e-6 * (1.0 + e_iv.mag))) and depth < depth_cap:
            mid = 0.5 * (lo + hi)
            stack.append((lo, mid, depth + 1))
            stack.append((mid, hi, depth + 1))
            continue
        core = max(core, e_iv.mag)
        leaves += 1

    far = 0.0
    n_far = 64 if probe < 1e-4 else 32
    for i in range(1, n_far + 1):
        y = yt * (1.0 + 3.0 * i / n_far)
        half = 0.5 * (3.0 * yt / n_far)
        y_iv = Interval.hull(max(yt, y - half), y + half)
        try:
            far = max(
                far,
                _hardy_residual_over_y_interval(
                    coeffs, scales, lam, y_iv, form, s, gammas=gammas
                ).mag,
            )
        except (ValueError, ZeroDivisionError):
            far = max(
                far,
                _hardy_residual_interval(
                    coeffs, scales, lam, y, form, s, gammas=gammas
                ).mag,
            )
    alpha = alpha_from_lambda(lam)
    if form == "vorticity":
        gs = list(gammas) if gammas is not None else [alpha] * len(coeffs)
        p_res = 2.0 * min(gs)
        c_tail = float(sum(abs(float(c)) for c in coeffs))
        majorant = float(Interval.point(c_tail).hi * (yt ** (-p_res)))
    else:
        c_tail, p_tail = hardy_tail_constant(coeffs, scales, alpha)
        majorant = float(Interval.point(c_tail).hi * (yt ** (-p_tail)))
    far_hi = max(far, majorant)
    return core, far_hi, max(core, far_hi), leaves, True


def refine_ccf_hardy_profile(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    nodes: Sequence[float] | None = None,
    form: str = "transport",
    velocity_sign: float = 1.0,
    iters: int = 80,
    tol: float = 1e-13,
    free_scales: bool = True,
    free_lam: bool = True,
    lam_target: float | None = None,
    lam_penalty: float = 0.0,
    omega_gauge_point: float | None = 0.5,
    omega_gauge_value: float = 0.05,
    omega_gauge_weight: float = 40.0,
    min_scale: float = 0.05,
    max_scale: float = 40.0,
    gammas: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Float Newton / LS find-step for a Hardy CCF candidate (not a proof).

    When ``free_scales`` is True, free unknowns include ``log a_i``.
    When ``free_lam`` is False, ``lambda`` is held fixed at the input value
    (use this to earn published-digit gates without collapsing to a trivial
    ``lambda ≈ 0`` root). Optional ``lam_penalty`` adds a soft
    ``sqrt(w)(lam - lam_target)`` row when ``free_lam`` is True.

    Soft ``Omega(omega_gauge_point)=omega_gauge_value`` and scale bounds reject
    the near-null cancelling micro-scale ghosts that can fake tiny residuals.

    ``form='vorticity'`` collocates the Wang residual on an Omega-primary dictionary.
    """
    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}")
    cs = [float(c) for c in coeffs]
    as_ = [float(a) for a in scales]
    n = len(cs)
    if n < 2:
        raise ValueError("need at least two Hardy terms")
    lam0 = float(lam)
    alpha0 = alpha_from_lambda(lam0)
    gs0 = (
        [float(g) for g in gammas]
        if gammas is not None
        else [alpha0] * n
    )
    if len(gs0) != n:
        raise ValueError("gammas length must match coeffs")
    if free_scales:
        n_free = (n - 1) + n + (1 if free_lam else 0)
        default_nodes = [
            0.15 * (1.55**k) for k in range(max(n_free + 2, n + 4))
        ]
        ynodes = list(
            default_nodes if nodes is None else [float(y) for y in nodes]
        )
    else:
        ynodes = list(
            default_ccf_collocation_nodes(n) if nodes is None else [float(y) for y in nodes]
        )
        if len(ynodes) != n:
            raise ValueError(f"need len(nodes)=={n}")
    c1 = cs[0]
    s = float(velocity_sign)
    parts: list[float] = [*cs[1:]]
    if free_scales:
        parts.extend(float(np.log(max(a, 1e-8))) for a in as_)
    if free_lam:
        parts.append(lam0)
    u = np.asarray(parts, dtype=float)
    log_lo = math.log(max(float(min_scale), 1e-8))
    log_hi = math.log(max(float(max_scale), float(min_scale) * 1.01))

    def unpack(vec: np.ndarray) -> tuple[list[float], list[float], float]:
        k = 0
        c_rest = [float(v) for v in vec[k : k + (n - 1)]]
        k += n - 1
        if free_scales:
            logs = vec[k : k + n]
            k += n
            scales_v = [
                float(math.exp(float(np.clip(v, log_lo, log_hi)))) for v in logs
            ]
        else:
            scales_v = as_
        if free_lam:
            lam_v = float(np.clip(vec[k], -0.9, 5.0))
        else:
            lam_v = lam0
        return [c1, *c_rest], scales_v, lam_v

    def residual(vec: np.ndarray) -> np.ndarray:
        full, scales_v, lam_v = unpack(vec)
        y = np.asarray(ynodes, dtype=float)
        if form == "vorticity":
            # Freeze gammas at input (α-ladder); rescale if lam freed.
            scale_g = alpha_from_lambda(lam_v) / alpha0 if free_lam else 1.0
            gs = [g * scale_g for g in gs0]
            om = np.zeros_like(y)
            omy = np.zeros_like(y)
            U = np.zeros_like(y)
            uy = np.zeros_like(y)
            for c, a, g in zip(full, scales_v, gs, strict=True):
                r = np.hypot(a, y)
                phi = np.arctan2(y, a)
                om += c * (r ** (-g)) * np.sin(g * phi)
                omy += c * g * (r ** (-(g + 1))) * np.cos((g + 1) * phi)
                uy += c * (-(r ** (-g)) * np.cos(g * phi))
                if abs(g - 1.0) < 1e-12:
                    U += c * (-np.arctan(y / a))
                else:
                    U += c * (-(r ** (-(g - 1))) * np.sin((g - 1) * phi) / (g - 1))
            r = om + ((1.0 + lam_v) * y - U) * omy - om * uy
            om_for_gauge = om
        else:
            from omnibias.symbolic.ccf import ccf_self_similar_residual, hardy_profile_numpy

            alpha = alpha_from_lambda(lam_v)
            th, thp, hth, hthp = hardy_profile_numpy(
                y, np.asarray(full), np.asarray(scales_v), alpha
            )
            r = ccf_self_similar_residual(
                y,
                th,
                thp,
                lam_v,
                form=form,
                velocity_sign=s,
                hilbert_convention="hardy_exact",
                hilbert_values=hth,
                hilbert_y_values=hthp,
            )
            om_for_gauge = thp
        extras: list[float] = []
        if free_lam and lam_penalty > 0.0 and lam_target is not None:
            extras.append(
                math.sqrt(float(lam_penalty)) * (lam_v - float(lam_target))
            )
        if omega_gauge_point is not None and omega_gauge_weight > 0.0:
            yg = float(omega_gauge_point)
            om_g = float(np.interp(yg, y, om_for_gauge))
            extras.append(
                math.sqrt(float(omega_gauge_weight))
                * (om_g - float(omega_gauge_value))
            )
        if extras:
            r = np.concatenate([r, np.asarray(extras, dtype=float)])
        return r

    def jacobian(vec: np.ndarray) -> np.ndarray:
        # FD Jacobian (covers free scales); analytic node system for frozen scales.
        eps = 1e-8
        f0 = residual(vec)
        cols = []
        for j in range(vec.size):
            vp = vec.copy()
            vp[j] = vp[j] + eps
            cols.append((residual(vp) - f0) / eps)
        return np.stack(cols, axis=1)

    for _ in range(int(iters)):
        f = residual(u)
        if float(np.max(np.abs(f))) < tol:
            break
        jac = jacobian(u)
        try:
            if jac.shape[0] == jac.shape[1]:
                step = np.linalg.solve(jac, f)
            else:
                step, *_ = np.linalg.lstsq(jac, f, rcond=None)
        except np.linalg.LinAlgError:
            break
        u = u - step
    final = residual(u)
    full, scales_v, lam_v = unpack(u)
    scale_g = alpha_from_lambda(lam_v) / alpha0 if free_lam else 1.0
    gs_out = [g * scale_g for g in gs0]
    return {
        "coeffs": full,
        "scales": scales_v,
        "gammas": gs_out,
        "lam": lam_v,
        "nodes": tuple(ynodes),
        "form": form,
        "velocity_sign": s,
        "residual_max_abs": float(np.max(np.abs(final))),
        "alpha": alpha_from_lambda(lam_v),
        "free_scales": bool(free_scales),
        "free_lam": bool(free_lam),
    }


def certified_ccf_hardy_wholeline_blowup_attempt(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    nodes: Sequence[float] | None = None,
    form: str = "transport",
    velocity_sign: float = 1.0,
    far_field_trunc: float | None = None,
    nu: float = 1.05,
    residual_gate: float = 1e-11,
    gammas: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Attempt a whole-line Hardy CAP; report quantified gap if it does not close."""
    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}")
    cs = [float(c) for c in coeffs]
    as_ = [float(a) for a in scales]
    lam_f = float(lam)
    s = float(velocity_sign)
    n = len(cs)
    if n < 1:
        raise ValueError("need at least one coefficient")
    if len(as_) != n:
        raise ValueError("coeffs and scales must have equal length")
    if any(a <= 0.0 for a in as_):
        raise ValueError("scales must be positive")
    alpha = alpha_from_lambda(lam_f)
    gs = (
        [float(g) for g in gammas]
        if gammas is not None
        else ([alpha] * n if form == "vorticity" else None)
    )
    if gs is not None and len(gs) != n:
        raise ValueError("gammas length must match coeffs")
    ynodes = list(
        default_ccf_collocation_nodes(n) if nodes is None else [float(y) for y in nodes]
    )
    if len(ynodes) != n:
        raise ValueError(f"need len(nodes) == {n}")
    f_iv: list[Interval] = []
    a_iv: list[list[Interval]] = []
    hess_sums: list[Interval] = []
    if form == "vorticity":
        assert gs is not None
        # Collocation residual + FD Jacobian over free coeffs c[1:] (+ frozen lam).
        # Padding encloses float FD error so NK stays sound-but-possibly-loose.
        pad = 1e-8

        def vort_at(cfull: list[float], y: float) -> float:
            return float(
                _vorticity_residual_interval(cfull, as_, gs, lam_f, y).mid
            )

        for y in ynodes:
            e0 = vort_at(cs, y)
            f_iv.append(Interval(e0 - pad, e0 + pad))
            row: list[Interval] = []
            for j in range(1, n):
                cp = list(cs)
                cp[j] = cs[j] + pad
                cm = list(cs)
                cm[j] = cs[j] - pad
                d = (vort_at(cp, y) - vort_at(cm, y)) / (2.0 * pad)
                row.append(Interval(d - pad, d + pad))
            # pad free dim to n columns with a dummy lam-column of zeros (lam frozen)
            while len(row) < n:
                row.append(Interval.point(0.0))
            a_iv.append(row[:n])
            hess_sums.append(Interval(-1.0, 1.0))  # crude curvature bound
    else:
        for y in ynodes:
            e_iv, row, hess_abs = _hardy_node_system(cs, as_, lam_f, y, form, s)
            f_iv.append(e_iv)
            a_iv.append(row)
            hess_sums.append(hess_abs)

    m = n
    a_float = [[iv.mid for iv in row] for row in a_iv]
    a_np = np.asarray(a_float, dtype=float)
    try:
        b_np = np.linalg.inv(a_np)
    except np.linalg.LinAlgError:
        b_np = np.zeros((m, m), dtype=float)
    b_float = [[float(b_np[i, j]) for j in range(m)] for i in range(m)]
    b_iv = to_interval_matrix(b_float)

    defect_matrix = mat_sub(identity_matrix(m), matmul(b_iv, a_iv))
    z1_col = inf_norm_matrix(defect_matrix)
    norm_b = inf_norm_matrix(b_iv)
    neumann = neumann_inverse_norm_bound(a_float, b_float)
    y0_col = inf_norm_vector(matvec(b_iv, f_iv))
    kappa2 = max((h.hi for h in hess_sums), default=0.0)
    z2_col = (Interval.point(norm_b) * Interval.point(kappa2)).hi
    closure_col = radii_polynomial_closure(y0_col, z1_col, z2_col)
    operator_invertible = bool(z1_col < 1.0)
    collocation_closed = bool(closure_col["passed"] and operator_invertible)

    yt = (
        float(far_field_trunc)
        if far_field_trunc is not None
        else 2.0 * max(max(ynodes), max(as_)) + 1.0
    )
    core_sup, far_hi, residual_certified_sup, leaves, between_ok = _certified_residual_sup(
        cs, as_, lam_f, form, s, yt, gammas=gs
    )
    if form == "vorticity":
        assert gs is not None
        c_tail = float(sum(abs(c) for c in cs))
        p_tail = float(min(gs))
    else:
        c_tail, p_tail = hardy_tail_constant(cs, as_, alpha)
    core_radius = max(ynodes)
    hilbert_tail = (
        float(hilbert_tail_bound(c_tail, p_tail, yt, core_radius).hi)
        if yt > core_radius
        else None
    )

    # Sequence-space NK without double-counting residual_certified_sup into Y0.
    free_coeffs = [abs(float(c)) for c in cs[1:]]
    trunc_norm = float(sum(c * (nu**k) for k, c in enumerate(free_coeffs)))
    # Measured geometric ratio from successive |c| (capped so nu*ratio < 1).
    if len(free_coeffs) >= 2 and free_coeffs[-2] > 0.0:
        measured_ratio = min(free_coeffs[-1] / free_coeffs[-2], 0.9 / float(nu))
    else:
        measured_ratio = 0.5 / float(nu)
    measured_ratio = max(float(measured_ratio), 1e-16)
    last = free_coeffs[-1] if free_coeffs else 0.0
    tail = geometric_tail_bound(
        max(last, 1e-30),
        measured_ratio,
        float(nu),
        n_trunc=max(len(free_coeffs) - 1, 0),
    )
    y0_seq = float(y0_col) + float(tail.hi)
    z0_seq = float(z1_col)
    z1_seq = float(tail.hi)
    z2_seq = float(z2_col)
    seq_cert = radii_polynomial_certificate(y0_seq, z0_seq, z1_seq, z2_seq)
    sequence_closed = seq_cert is not None

    residual_ok = bool(residual_certified_sup <= float(residual_gate))
    whole_line = bool(
        residual_ok
        and between_ok
        and sequence_closed
        and collocation_closed
        and operator_invertible
    )

    r_minus = closure_col["r_minus"] if collocation_closed else None
    lambda_enclosure: dict[str, Any] | None = None
    if collocation_closed and r_minus is not None:
        from dataclasses import asdict

        enc = interval_from_bounds(
            lam_f - float(r_minus), lam_f + float(r_minus), certified=True
        )
        lambda_enclosure = asdict(enc)

    gap = {
        "residual_certified_sup": residual_certified_sup,
        "residual_gate": float(residual_gate),
        "residual_gap": max(0.0, residual_certified_sup - float(residual_gate)),
        "collocation_closed": collocation_closed,
        "sequence_space_closed": sequence_closed,
        "failed_inequality": closure_col.get("failed_inequality"),
        "sequence_Y0": y0_seq,
        "sequence_Z0": z0_seq,
        "sequence_Z1": z1_seq,
        "sequence_Z2": z2_seq,
        "ell1_nu_trunc_norm": trunc_norm,
        "geometric_tail_bound": float(tail.hi),
    }

    body: dict[str, Any] = {
        "schema_version": CCF_HARDY_WHOLELINE_SCHEMA_VERSION,
        "observable": "cordoba_cordoba_fontelos_hardy_wholeline_blowup_profile",
        "model": "cordoba_cordoba_fontelos_1d",
        "equation": "theta_t + (H theta) theta_x = 0",
        "route": "finite_time_blowup",
        "form": form,
        "velocity_sign": float(s),
        "basis": "verified_cauchy_hardy_exact_hilbert",
        "coeffs": cs,
        "scales": as_,
        "gammas": list(gs) if gs is not None else None,
        "alpha": alpha,
        "n_terms": int(n),
        "collocation_nodes": [float(y) for y in ynodes],
        "lambda_candidate": lam_f,
        "closure_certified": whole_line,
        "collocation_closure_certified": collocation_closed,
        "sequence_space_closure_certified": sequence_closed,
        "operator_invertible_certified": operator_invertible,
        "selfsimilar_profile_certified": whole_line,
        "lambda_enclosure": lambda_enclosure,
        "profile_enclosure_radius": float(r_minus) if r_minus is not None else None,
        "closure_report": {
            "residual_normal_form_Y0": float(y0_col),
            "linear_defect_Z1": float(z1_col),
            "nonlinear_curvature_Z2": float(z2_col),
            "approximate_inverse_norm": float(norm_b),
            "neumann_kappa": float(neumann["kappa"]),
            "neumann_certified": bool(neumann["certified"]),
            "discriminant_lower": closure_col["discriminant_lower"],
            "existence_radius_r_minus": closure_col["r_minus"],
            "uniqueness_radius_r_plus": closure_col["r_plus"],
            "failed_inequality": closure_col["failed_inequality"],
            "residual_certified_core_sup": core_sup,
            "residual_certified_sup": residual_certified_sup,
            "between_node_residual_certified": bool(between_ok),
            "residual_interval_leaf_cells": int(leaves),
            "far_field_residual_bound": far_hi,
            "far_field_trunc": float(yt),
            "profile_tail_constant": float(c_tail),
            "profile_tail_power": float(p_tail),
            "hilbert_far_field_tail_on_core": hilbert_tail,
            "nu": float(nu),
            "quantified_gap": gap,
        },
        "method": (
            "hardy_exact_hilbert_interval_covering_plus_ell1_nu_radii_polynomial"
        ),
        "three_d_claim": False,
        "continuum_navier_stokes_claim": False,
        "honesty": {
            "unproven_claim": False,
            "three_d_claim": False,
            "model_only": True,
            "one_dimensional_model": True,
            "collocation_only": not whole_line,
            "whole_line_certified": whole_line,
            "interval_verified": True,
            "exact_closed_form_hilbert": True,
            "certified": whole_line,
            "navier_stokes_proof_claim": False,
            "note": (
                "Hardy-basis whole-line CAP attempt. whole_line_certified is True "
                "only when residual_certified_sup clears residual_gate and both "
                "collocation and ell1_nu radii-polynomials close. Otherwise the "
                "quantified gap is reported and the result stays BLOCKED. Not 3D NS."
            ),
        },
    }
    body["provenance"] = {
        "harness": (
            "omnibias.pinn.certified.ccf_hardy.certified_ccf_hardy_wholeline_blowup_attempt"
        ),
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_CCF_HARDY_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "equation",
    "route",
    "form",
    "basis",
    "coeffs",
    "scales",
    "alpha",
    "lambda_candidate",
    "closure_certified",
    "closure_report",
    "honesty",
    "provenance",
    "three_d_claim",
    "continuum_navier_stokes_claim",
)


def certified_ccf_hardy_wholeline_blowup_attempt_schema_errors(
    cert: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_CCF_HARDY_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    honesty = cert.get("honesty", {})
    if honesty.get("unproven_claim", False):
        errors.append("honesty.unproven_claim must be False")
    if cert.get("three_d_claim", True):
        errors.append("three_d_claim must be False")
    if cert.get("continuum_navier_stokes_claim", True):
        errors.append("continuum_navier_stokes_claim must be False")
    if cert.get("schema_version") != CCF_HARDY_WHOLELINE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CCF_HARDY_WHOLELINE_SCHEMA_VERSION!r}")
    if honesty.get("whole_line_certified") and not cert.get("closure_certified"):
        errors.append("whole_line_certified requires closure_certified")
    if honesty.get("whole_line_certified") and honesty.get("collocation_only"):
        errors.append("whole_line_certified cannot be collocation_only")
    if honesty.get("navier_stokes_proof_claim"):
        errors.append("honesty.navier_stokes_proof_claim must be False")
    return errors


__all__ = [
    "CCF_HARDY_WHOLELINE_SCHEMA_VERSION",
    "REQUIRED_CCF_HARDY_KEYS",
    "alpha_from_lambda",
    "certified_ccf_hardy_wholeline_blowup_attempt",
    "certified_ccf_hardy_wholeline_blowup_attempt_schema_errors",
    "refine_ccf_hardy_profile",
]

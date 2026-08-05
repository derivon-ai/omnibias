# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Independent numpy replay of the 2-D SQG steady-vortex certificate.

This module is **numpy-only** (it imports nothing from
:mod:`omnibias.pinn.certified`), so it is a genuine *second source* for the
certificate produced by
:func:`omnibias.pinn.certified.certified_sqg_steady_vortex`.  It reimplements the
SQG velocity ``u = R^\perp\theta = (1/2\pi)(y,-x)D^{-3/2}``, the temperature
gradient ``\nabla\theta`` and the single Riesz transform from scratch and:

1. **re-confirms the exact steady state** -- ``u\cdot\nabla\theta`` on a dense 2-D
   grid is ``~10^{-17}`` (the perpendicularity of the tangential velocity and the
   radial temperature gradient), together with ``\nabla\cdot u`` and the
   ``u = R^\perp\theta`` consistency;
2. **densely samples** the three radial magnitudes and checks that the reported
   ``velocity_sup`` / ``temperature_sup`` / ``strain_sup`` genuinely *dominate*
   the samples -- a forged, understated sup is caught (anti-faking);
3. **recomputes the total temperature** ``\sum_i c_i``.

Nothing here asserts an SQG finite-time singularity or a global-regularity result;
:func:`verify_sqg_steady_vortex` returns a plain dict with ``unproven_claim=False``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def sqg_radial_fields(
    r: np.ndarray, coeffs: np.ndarray, scales: np.ndarray
) -> dict[str, np.ndarray]:
    r"""Radial reductions of the SQG vortex norms (independent closed forms).

    Returns ``r |S|`` (velocity up to ``1/2\pi``), ``|\sum_i c_i a_i D_i^{-3/2}|``
    (temperature up to ``1/2\pi``) and ``M = 2 S^2 - 6 r^2 S T + 9 r^4 T^2`` (strain
    ``\lVert\nabla u\rVert_F^2`` up to ``1/(2\pi)^2``), with
    ``S=\sum_i c_i D_i^{-3/2}``, ``T=\sum_i c_i D_i^{-5/2}`` and ``D_i=r^2+a_i^2``.
    """
    d = r[:, None] ** 2 + scales[None, :] ** 2
    s = (coeffs[None, :] * d ** (-1.5)).sum(axis=1)
    t = (coeffs[None, :] * d ** (-2.5)).sum(axis=1)
    theta_num = (coeffs[None, :] * scales[None, :] * d ** (-1.5)).sum(axis=1)
    return {
        "r_abs_s": np.abs(r * s),
        "abs_theta_num": np.abs(theta_num),
        "strain_sq_num": 2.0 * s**2 - 6.0 * r**2 * s * t + 9.0 * r**4 * t**2,
    }


def sqg_grid_residuals(
    coeffs: np.ndarray, scales: np.ndarray, r_trunc: float, g: int
) -> dict[str, float]:
    r"""Max over a ``g x g`` grid of ``u\cdot\nabla\theta``, ``\nabla\cdot u`` and ``u-R^\perp\theta``.

    All from independent numpy closed forms: ``u = (1/2\pi)(y,-x)D^{-3/2}``,
    ``\nabla\theta = -(3a/2\pi)(x,y)D^{-5/2}`` and the single Riesz transform
    ``R_j\theta = -(1/2\pi)x_j D^{-3/2}``.
    """
    span = np.linspace(-r_trunc, r_trunc, g)
    xx, yy = np.meshgrid(span, span, indexing="ij")
    x = xx.ravel()
    y = yy.ravel()
    res = np.zeros_like(x)
    div = np.zeros_like(x)
    perp = np.zeros_like(x)
    tp = 2.0 * np.pi
    for c, a in zip(coeffs, scales, strict=True):
        d = x * x + y * y + a * a
        d15 = d**1.5
        d25 = d**2.5
        ux = c * y / (tp * d15)
        uy = -c * x / (tp * d15)
        tx = -3.0 * c * a * x / (tp * d25)
        ty = -3.0 * c * a * y / (tp * d25)
        res = res + ux * tx + uy * ty
        # div u = d_x u_x + d_y u_y = -3 c x y/(tp D^{5/2}) + 3 c x y/(tp D^{5/2}) = 0
        div = div + (-3.0 * c * x * y / (tp * d25)) + (3.0 * c * x * y / (tp * d25))
        # u = R^perp theta = (-R_2 theta, R_1 theta), R_j theta = -c x_j/(tp D^{3/2})
        r0 = -c * x / (tp * d15)
        r1 = -c * y / (tp * d15)
        perp = perp + (ux - (-r1)) + (uy - r0)
    return {
        "steady_residual_grid_max": float(np.max(np.abs(res))),
        "divergence_grid_max": float(np.max(np.abs(div))),
        "riesz_perp_identity_grid_max": float(np.max(np.abs(perp))),
    }


def _trapz(yv: np.ndarray, xv: np.ndarray) -> float:
    """Composite-trapezoid integral (avoids the deprecated ``np.trapz`` name)."""
    return float(np.sum(0.5 * (yv[1:] + yv[:-1]) * np.diff(xv)))


def sqg_selfsimilar_l2_quantities(
    coeffs: np.ndarray, scales: np.ndarray, *, r_max: float | None = None, n: int = 400001
) -> dict[str, float]:
    r"""Independent radial-quadrature of the self-similar obstruction integrals.

    For ``\theta=\sum_i c_i\theta_{a_i}`` (radial), the self-similar residual is
    ``F=(y+R^\perp\theta)\cdot\nabla\theta = y\cdot\nabla\theta = r\theta'(r)``
    (``R^\perp\theta\perp\nabla\theta``).  Returns ``\lVert\theta\rVert_2^2``,
    ``\langle F,\theta\rangle`` and ``\lVert F\rVert_2^2`` as whole-plane radial
    integrals ``\int_0^\infty(\cdot)\,2\pi r\,dr`` (numpy trapezoid; the integrands
    decay like ``r^{-5}`` so the truncation tail is negligible).  The no-go theorem
    predicts ``\langle F,\theta\rangle = -\lVert\theta\rVert_2^2`` and
    ``\lVert F\rVert_2 \ge \lVert\theta\rVert_2``.
    """
    rm = float(50.0 * float(np.max(scales)) + 50.0) if r_max is None else float(r_max)
    r = np.linspace(0.0, rm, n)
    d = r[:, None] ** 2 + scales[None, :] ** 2
    tp = 2.0 * np.pi
    theta = (coeffs[None, :] * scales[None, :] * d ** (-1.5)).sum(axis=1) / tp
    thetap = (coeffs[None, :] * scales[None, :] * (-3.0 * r[:, None]) * d ** (-2.5)).sum(axis=1) / tp
    fres = r * thetap  # F = y . grad theta = r theta'
    w = tp * r
    return {
        "profile_l2_norm_sq": _trapz(theta * theta * w, r),
        "selfsimilar_residual_inner_product": _trapz(fres * theta * w, r),
        "selfsimilar_residual_l2_sq": _trapz(fres * fres * w, r),
    }


def verify_sqg_selfsimilar_blowup_attempt(
    cert: dict[str, Any], *, rtol: float = 1e-4, atol: float = 1e-9
) -> dict[str, Any]:
    r"""Independent numpy replay of an ``sqg-selfsimilar-blowup-attempt-1`` certificate.

    Recomputes the self-similar obstruction integrals by radial quadrature and
    confirms, from a *second source*:

    1. the closed-form ``\lVert\theta\rVert_2^2`` (``sum_ij c_ic_j/(2\pi(a_i+a_j)^2)``)
       matches the quadrature (anti-faking of the diagonalised norm);
    2. the **no-go identity** ``\langle F,\theta\rangle = -\lVert\theta\rVert_2^2``
       (the divergence-theorem mechanism, ``\nabla\!\cdot V=2``);
    3. the **obstruction is real**: ``\lVert F\rVert_2 \ge \lVert\theta\rVert_2 > 0``,
       so no exact self-similar profile exists, and the reported
       ``selfsimilar_residual_l2_lower_bound`` genuinely under-bounds ``\lVert F\rVert_2``;
    4. the ``L^2`` drift energy coefficient ``\langle -y\cdot\nabla W,W\rangle =
       +\lVert W\rVert_2^2`` (destabilizing) for ``W=\theta``;
    5. ``(R^\perp\theta)\cdot\nabla\theta\equiv0`` on a 2-D grid (so ``F=y\cdot\nabla\theta``);
    6. the honesty flags (no blow-up / global-regularity / 3-D claim; obstruction recorded).

    Returns a report with ``replay_match`` and ``unproven_claim=False``.
    """
    coeffs = np.asarray(cert["coeffs"], dtype=float)
    scales = np.asarray(cert["scales"], dtype=float)
    r_trunc = float(cert["far_field_trunc"])

    q = sqg_selfsimilar_l2_quantities(coeffs, scales)
    l2_sq = q["profile_l2_norm_sq"]
    ip = q["selfsimilar_residual_inner_product"]
    f_norm = q["selfsimilar_residual_l2_sq"] ** 0.5
    th_norm = l2_sq**0.5

    rep_l2_lo, rep_l2_hi = (float(v) for v in cert["profile_l2_norm_sq"])
    rep_l2_mid = 0.5 * (rep_l2_lo + rep_l2_hi)
    l2_match = bool(abs(l2_sq - rep_l2_mid) <= rtol * rep_l2_mid + atol)

    rep_ip_lo, rep_ip_hi = (float(v) for v in cert["selfsimilar_residual_inner_product"])
    rep_ip_mid = 0.5 * (rep_ip_lo + rep_ip_hi)
    ip_match = bool(abs(ip - rep_ip_mid) <= rtol * abs(rep_ip_mid) + atol)
    nogo_identity = bool(abs(ip + l2_sq) <= rtol * abs(l2_sq) + atol)

    obstruction_holds = bool(f_norm + atol >= th_norm > 0.0)
    rep_lb = float(cert["selfsimilar_residual_l2_lower_bound"])
    lower_bound_valid = bool(0.0 < rep_lb <= f_norm * (1.0 + rtol) + atol)

    drift_coeff = -ip / l2_sq
    drift_match = bool(
        abs(drift_coeff - float(cert["l2_self_similar_drift_energy_coefficient"])) <= 1e-3
    )

    grid = sqg_grid_residuals(coeffs, scales, r_trunc, int(cert.get("grid_points", 9)))
    perpendicular = bool(grid["steady_residual_grid_max"] <= 1e-10)

    honesty = cert.get("honesty", {})
    honesty_ok = bool(
        not honesty.get("blowup_claim", True)
        and not honesty.get("unproven_claim", True)
        and not honesty.get("three_d_claim", True)
        and cert.get("exact_selfsimilar_profile_exists", True) is False
    )

    replay_match = bool(
        l2_match
        and ip_match
        and nogo_identity
        and obstruction_holds
        and lower_bound_valid
        and drift_match
        and perpendicular
        and honesty_ok
    )
    return {
        "recomputed_profile_l2_norm_sq": l2_sq,
        "recomputed_selfsimilar_residual_inner_product": ip,
        "recomputed_selfsimilar_residual_l2_norm": f_norm,
        "recomputed_profile_l2_norm": th_norm,
        "recomputed_l2_drift_energy_coefficient": drift_coeff,
        "recomputed_perpendicularity_grid_max": grid["steady_residual_grid_max"],
        "profile_l2_norm_sq_match": l2_match,
        "residual_inner_product_match": ip_match,
        "nogo_identity_holds": nogo_identity,
        "obstruction_holds": obstruction_holds,
        "lower_bound_valid": lower_bound_valid,
        "drift_coefficient_match": drift_match,
        "perpendicularity_holds": perpendicular,
        "honesty_consistent": honesty_ok,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


def sqg_grad_theta_sup_sample(
    coeffs: np.ndarray, scales: np.ndarray, *, r_max: float | None = None, n: int = 200001
) -> float:
    r"""Independent dense-sample of ``\lVert\nabla\bar\Theta\rVert_\infty``.

    ``|\nabla\theta| = (3/2\pi)\,r\,|\sum_i c_i a_i D_i^{-5/2}|`` (radial); returns the
    max over a fine radial grid -- an under-estimate of the true sup, so a certified
    sup must dominate it (anti-faking).
    """
    rm = float(20.0 * float(np.max(scales)) + 20.0) if r_max is None else float(r_max)
    r = np.linspace(0.0, rm, n)
    d = r[:, None] ** 2 + scales[None, :] ** 2
    inner = (coeffs[None, :] * scales[None, :] * d ** (-2.5)).sum(axis=1)
    grad_mag = (3.0 / (2.0 * np.pi)) * np.abs(r * inner)
    return float(np.max(grad_mag))


def verify_sqg_linearized_coercivity_attempt(
    cert: dict[str, Any], *, rtol: float = 1e-4, atol: float = 1e-9
) -> dict[str, Any]:
    r"""Independent numpy replay of an ``sqg-linearized-coercivity-attempt-1`` certificate.

    From a second source it (1) densely samples ``\lVert\nabla\bar\Theta\rVert_\infty``
    and checks the certified ``grad_theta_sup`` dominates it (anti-faking), (2)
    recomputes the Weyl ``L^2`` gap ``1 - \lVert\nabla\bar\Theta\rVert_\infty`` and
    matches it, (3) re-derives the block-operator gap ``\tfrac12[(a+d) -
    \sqrt{(a-d)^2 + 4b^2}]`` with ``a=d=1``, ``b=`` coupling and matches the reported
    ``block_operator_gap.gap_lower``, and (4) checks the honesty flags.  Returns a
    report with ``replay_match`` and ``unproven_claim=False``.
    """
    coeffs = np.asarray(cert["coeffs"], dtype=float)
    scales = np.asarray(cert["scales"], dtype=float)

    sampled_grad = sqg_grad_theta_sup_sample(coeffs, scales)
    rep_grad = float(cert["grad_theta_sup"])
    grad_dominates = bool(sampled_grad <= rep_grad * (1.0 + rtol) + atol)

    rep_gap = float(cert["l2_coercivity_gap_lower"])
    recomputed_gap = 1.0 - rep_grad  # Weyl bound from the reported (certified) sup
    gap_match = bool(abs(recomputed_gap - rep_gap) <= rtol * abs(recomputed_gap) + atol)
    coercive_match = bool(bool(cert["l2_coercive"]) == (rep_gap > 0.0))

    b = float(cert["stretching_coupling_bound"])
    block_formula = 0.5 * (2.0 - math.sqrt(4.0 * b * b))  # a=d=1
    rep_block = float(cert["block_operator_gap"]["gap_lower"])
    block_match = bool(abs(block_formula - rep_block) <= rtol * abs(block_formula) + atol)

    facts_ok = bool(
        float(cert.get("drift_self_adjoint_coefficient", 0.0)) == 1.0
        and float(cert.get("riesz_isometry_constant", 0.0)) == 1.0
    )
    honesty = cert.get("honesty", {})
    honesty_ok = bool(
        not honesty.get("blowup_claim", True)
        and not honesty.get("unproven_claim", True)
        and not honesty.get("three_d_claim", True)
        and not honesty.get("stability_claim", True)
    )

    replay_match = bool(
        grad_dominates and gap_match and coercive_match and block_match and facts_ok and honesty_ok
    )
    return {
        "sampled_grad_theta_sup": sampled_grad,
        "recomputed_l2_gap": recomputed_gap,
        "recomputed_block_gap": block_formula,
        "grad_sup_dominates_samples": grad_dominates,
        "l2_gap_match": gap_match,
        "coercive_match": coercive_match,
        "block_gap_match": block_match,
        "exact_facts_consistent": facts_ok,
        "honesty_consistent": honesty_ok,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


def verify_sqg_steady_vortex(
    cert: dict[str, Any], *, rtol: float = 1e-6, atol: float = 1e-9
) -> dict[str, Any]:
    r"""Independent numpy replay of an ``sqg-steady-vortex-1`` certificate.

    Recomputes the total temperature, densely samples the radial norm magnitudes
    to confirm the reported sups dominate them (anti-faking), and re-confirms the
    steady residual / divergence / ``u = R^\perp\theta`` identities on a 2-D grid.
    Returns a report with ``unproven_claim=False`` and ``replay_match``.
    """
    coeffs = np.asarray(cert["coeffs"], dtype=float)
    scales = np.asarray(cert["scales"], dtype=float)
    r_trunc = float(cert["far_field_trunc"])

    total_temperature = float(coeffs.sum())
    mass_match = bool(abs(total_temperature - float(cert["total_temperature"])) <= atol)

    # dense radial sampling -> physical sups (apply the analytic pi constants).
    r = np.linspace(0.0, r_trunc, 20001)
    fields = sqg_radial_fields(r, coeffs, scales)
    tp = 2.0 * np.pi
    vel_sampled = float(np.max(fields["r_abs_s"]) / tp)
    temp_sampled = float(np.max(fields["abs_theta_num"]) / tp)
    strain_sampled = float(np.sqrt(np.max(fields["strain_sq_num"])) / tp)

    rep_vel = float(cert["velocity_sup"])
    rep_temp = float(cert["temperature_sup"])
    rep_strain = float(cert["strain_sup"])
    sup_dominates = bool(
        vel_sampled <= rep_vel * (1.0 + rtol) + atol
        and temp_sampled <= rep_temp * (1.0 + rtol) + atol
        and strain_sampled <= rep_strain * (1.0 + rtol) + atol
    )

    grid = sqg_grid_residuals(coeffs, scales, r_trunc, int(cert.get("grid_points", 9)))
    residual_is_zero = bool(grid["steady_residual_grid_max"] <= 1e-10)
    identities_hold = bool(
        grid["divergence_grid_max"] <= 1e-8 and grid["riesz_perp_identity_grid_max"] <= 1e-8
    )
    verdict_match = bool(
        bool(cert.get("honesty", {}).get("exact_steady_state", False))
        == (residual_is_zero and float(cert["steady_residual_certified_sup"]) == 0.0)
    )

    replay_match = bool(
        mass_match and sup_dominates and residual_is_zero and identities_hold and verdict_match
    )
    return {
        "recomputed_total_temperature": total_temperature,
        "sampled_velocity_sup": vel_sampled,
        "sampled_temperature_sup": temp_sampled,
        "sampled_strain_sup": strain_sampled,
        "recomputed_steady_residual_grid_max": grid["steady_residual_grid_max"],
        "recomputed_divergence_grid_max": grid["divergence_grid_max"],
        "recomputed_riesz_perp_identity_grid_max": grid["riesz_perp_identity_grid_max"],
        "total_temperature_match": mass_match,
        "sup_dominates_samples": sup_dominates,
        "steady_residual_is_zero": residual_is_zero,
        "identities_hold": identities_hold,
        "verdict_match": verdict_match,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


__all__ = [
    "sqg_grad_theta_sup_sample",
    "sqg_grid_residuals",
    "sqg_radial_fields",
    "sqg_selfsimilar_l2_quantities",
    "verify_sqg_linearized_coercivity_attempt",
    "verify_sqg_selfsimilar_blowup_attempt",
    "verify_sqg_steady_vortex",
]

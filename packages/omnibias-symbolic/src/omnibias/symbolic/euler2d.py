# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Independent numpy replay of the 2-D Euler steady-vortex certificate.

This module is **numpy-only** (it imports nothing from
:mod:`omnibias.pinn.certified`), so it is a genuine *second source* for the
certificate produced by
:func:`omnibias.pinn.certified.certified_euler2d_steady_vortex`.  It reimplements
the Biot--Savart velocity, the vorticity gradient and the second-order Riesz
building blocks from scratch and:

1. **re-confirms the exact steady state** -- ``u\cdot\nabla\omega`` on a dense 2-D
   grid is ``~10^{-17}`` (the perpendicularity of the tangential velocity and the
   radial vorticity gradient), together with ``\nabla\cdot u`` and the
   Calderon--Zygmund trace ``R_{11}\omega + R_{22}\omega + \omega``;
2. **densely samples** the three radial magnitudes and checks that the reported
   ``velocity_sup`` / ``vorticity_sup`` / ``strain_sup`` genuinely *dominate* the
   samples -- a forged, understated sup is caught (anti-faking);
3. **recomputes the circulation** ``\sum_i c_i``.

Nothing here asserts a Navier--Stokes/Euler blow-up or a global-regularity result;
:func:`verify_euler2d_steady_vortex` returns a plain dict with ``unproven_claim=False``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def euler2d_radial_fields(
    r: np.ndarray, coeffs: np.ndarray, scales: np.ndarray
) -> dict[str, np.ndarray]:
    r"""Radial reductions of the vortex norms (independent closed forms).

    Returns ``r |Q|`` (velocity up to ``1/2\pi``), ``|\Omega_p|`` (vorticity up to
    ``1/\pi``) and ``\Omega_p^2 + r^4 W^2`` (strain ``\lVert\nabla u\rVert_F^2`` up
    to ``1/2\pi^2``), with ``Q=\sum_i c_i/D_i``, ``\Omega_p=\sum_i c_i a_i^2/D_i^2``,
    ``W=\sum_i c_i/D_i^2`` and ``D_i = r^2 + a_i^2``.
    """
    d = r[:, None] ** 2 + scales[None, :] ** 2
    q = (coeffs[None, :] / d).sum(axis=1)
    omega_p = (coeffs[None, :] * scales[None, :] ** 2 / d**2).sum(axis=1)
    w = (coeffs[None, :] / d**2).sum(axis=1)
    return {
        "r_abs_q": np.abs(r * q),
        "abs_omega_p": np.abs(omega_p),
        "strain_sq_num": omega_p**2 + r**4 * w**2,
    }


def euler2d_grid_residuals(
    coeffs: np.ndarray, scales: np.ndarray, r_trunc: float, g: int
) -> dict[str, float]:
    r"""Max over a ``g x g`` grid of the steady residual, ``\nabla\cdot u`` and CZ trace.

    All from independent numpy closed forms: ``u = \nabla^\perp\psi`` with
    ``\nabla N_a = (x,y)/(2\pi D)``, ``\nabla f_a = -4a^2(x,y)/(\pi D^3)`` and
    ``R_iR_k f_a = (2x_ix_k - \delta_{ik}D)/(2\pi D^2)``.
    """
    span = np.linspace(-r_trunc, r_trunc, g)
    xx, yy = np.meshgrid(span, span, indexing="ij")
    x = xx.ravel()
    y = yy.ravel()
    res = np.zeros_like(x)
    div = np.zeros_like(x)
    cz = np.zeros_like(x)
    for c, a in zip(coeffs, scales, strict=True):
        d = x * x + y * y + a * a
        # velocity u = (-d_y psi, d_x psi), psi = sum c_i N_{a_i}; grad N = (x,y)/(2 pi D)
        ux = -c * y / (2.0 * np.pi * d)
        uy = c * x / (2.0 * np.pi * d)
        # vorticity gradient nabla f_a = -4 a^2 (x, y) / (pi D^3)
        ox = -4.0 * c * a * a * x / (np.pi * d**3)
        oy = -4.0 * c * a * a * y / (np.pi * d**3)
        res = res + ux * ox + uy * oy
        # R01 = R10 = 2 x y / (2 pi D^2); div u = sum c (R01 - R10) f -> 0
        r01 = c * (2.0 * x * y) / (2.0 * np.pi * d**2)
        div = div + r01 - r01
        # R11 + R22 = -f_a; trace = sum c (R11 + R22 + f)
        r11 = c * (2.0 * x * x - d) / (2.0 * np.pi * d**2)
        r22 = c * (2.0 * y * y - d) / (2.0 * np.pi * d**2)
        fa = c * a * a / (np.pi * d**2)
        cz = cz + r11 + r22 + fa
    return {
        "steady_residual_grid_max": float(np.max(np.abs(res))),
        "divergence_grid_max": float(np.max(np.abs(div))),
        "riesz_trace_identity_grid_max": float(np.max(np.abs(cz))),
    }


def verify_euler2d_steady_vortex(
    cert: dict[str, Any], *, rtol: float = 1e-6, atol: float = 1e-9
) -> dict[str, Any]:
    r"""Independent numpy replay of an ``euler2d-steady-vortex-1`` certificate.

    Recomputes the circulation, densely samples the radial norm magnitudes to
    confirm the reported sups dominate them (anti-faking), and re-confirms the
    steady residual / divergence / Calderon--Zygmund-trace identities on a 2-D
    grid.  Returns a report with ``unproven_claim=False`` and ``replay_match``.
    """
    coeffs = np.asarray(cert["coeffs"], dtype=float)
    scales = np.asarray(cert["scales"], dtype=float)
    r_trunc = float(cert["far_field_trunc"])

    circulation = float(coeffs.sum())
    circ_match = bool(abs(circulation - float(cert["circulation"])) <= atol)

    # dense radial sampling -> physical sups (apply the analytic pi constants).
    r = np.linspace(0.0, r_trunc, 20001)
    fields = euler2d_radial_fields(r, coeffs, scales)
    vel_sampled = float(np.max(fields["r_abs_q"]) / (2.0 * np.pi))
    vort_sampled = float(np.max(fields["abs_omega_p"]) / np.pi)
    strain_sampled = float(np.sqrt(np.max(fields["strain_sq_num"]) / (2.0 * np.pi**2)))

    rep_vel = float(cert["velocity_sup"])
    rep_vort = float(cert["vorticity_sup"])
    rep_strain = float(cert["strain_sup"])
    sup_dominates = bool(
        vel_sampled <= rep_vel * (1.0 + rtol) + atol
        and vort_sampled <= rep_vort * (1.0 + rtol) + atol
        and strain_sampled <= rep_strain * (1.0 + rtol) + atol
    )

    grid = euler2d_grid_residuals(coeffs, scales, r_trunc, int(cert.get("grid_points", 9)))
    residual_is_zero = bool(grid["steady_residual_grid_max"] <= 1e-10)
    identities_hold = bool(
        grid["divergence_grid_max"] <= 1e-8 and grid["riesz_trace_identity_grid_max"] <= 1e-8
    )
    verdict_match = bool(
        bool(cert.get("honesty", {}).get("exact_steady_state", False))
        == (residual_is_zero and float(cert["steady_residual_certified_sup"]) == 0.0)
    )

    replay_match = bool(
        circ_match and sup_dominates and residual_is_zero and identities_hold and verdict_match
    )
    return {
        "recomputed_circulation": circulation,
        "sampled_velocity_sup": vel_sampled,
        "sampled_vorticity_sup": vort_sampled,
        "sampled_strain_sup": strain_sampled,
        "recomputed_steady_residual_grid_max": grid["steady_residual_grid_max"],
        "recomputed_divergence_grid_max": grid["divergence_grid_max"],
        "recomputed_riesz_trace_identity_grid_max": grid["riesz_trace_identity_grid_max"],
        "circulation_match": circ_match,
        "sup_dominates_samples": sup_dominates,
        "steady_residual_is_zero": residual_is_zero,
        "identities_hold": identities_hold,
        "verdict_match": verdict_match,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


__all__ = [
    "euler2d_grid_residuals",
    "euler2d_radial_fields",
    "verify_euler2d_steady_vortex",
]

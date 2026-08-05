# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Symbolic / neural-jet validation of Córdoba-Córdoba-Fontelos candidates.

This module is **numpy-only** (no jax / torch / pinn imports) so it can act as an
*independent second source* for verifying CCF self-similar candidates produced by
:mod:`omnibias.pinn.jax.discovery`. It does three honest things:

1. **Independent residual recomputation** (:func:`ccf_self_similar_residual`,
   :func:`verify_cap_bundle`). The self-similar CCF residual and its periodic
   Hilbert transform are re-implemented from scratch here; agreement with the
   jax/torch operator (or a CAP bundle's stored ``residual_samples``) is an
   exact-substitution cross-check, not a re-run of the same code.

2. **Governing-law recovery** (:func:`recover_ccf_scaling_law`). Using the
   library-free sparse regressor :func:`omnibias.symbolic.fit_sparse_equation`,
   it recovers the coefficients of the CCF self-similar relation

   .. math:: \Theta = \tfrac{1+\lambda}{\lambda}\,(y\,\Theta')
             + \tfrac{1}{\lambda}\,(H\Theta)\,\Theta')  \; [-\tfrac{1}{\lambda} g]

   from sampled jets ``(y, Theta, Theta', H Theta)`` (plus an optional forcing
   ``g`` for manufactured-solution checks), and reads :math:`\lambda` back off the
   fitted coefficient. For a manufactured exact candidate this recovery is exact;
   for a genuine numerical candidate it is approximate and reported as such.

3. **No overclaiming.** Nothing here asserts an exact symbolic solution or a
   Navier-Stokes result; :func:`assess_ccf_candidate` returns a plain dict of
   measured quantities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omnibias.symbolic.discovery import fit_sparse_equation

_FORMS = ("transport", "flux")


def periodic_hilbert(values: np.ndarray, *, axis: int = -1) -> np.ndarray:
    r"""Periodic discrete Hilbert transform via FFT (numpy reference).

    Convention ``-i sgn(m)`` with mean and even-N Nyquist modes zeroed, matching
    :mod:`omnibias.pinn.jax.hilbert`. Exact for band-limited periodic data.
    """
    x = np.asarray(values)
    n = x.shape[axis]
    if n < 2:
        raise ValueError(f"periodic_hilbert needs at least 2 samples, got {n}")
    modes = np.fft.fftfreq(n) * n
    mult = -1j * np.sign(modes)
    if n % 2 == 0:
        mult[n // 2] = 0.0
    shape = [1] * x.ndim
    shape[axis] = n
    mult = mult.reshape(shape)
    out = np.fft.ifft(np.fft.fft(x, axis=axis) * mult, axis=axis)
    if np.iscomplexobj(x):
        return out
    return np.real(out)


def ccf_self_similar_residual(
    y: np.ndarray,
    theta: np.ndarray,
    theta_y: np.ndarray,
    lam: float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
) -> np.ndarray:
    """Independent numpy reimplementation of the CCF self-similar residual."""
    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}, got {form!r}")
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    theta_y = np.asarray(theta_y, dtype=float)
    h_theta = periodic_hilbert(theta)
    if form == "transport":
        nonlocal_term = h_theta * theta_y
    else:
        h_theta_y = periodic_hilbert(theta_y)
        nonlocal_term = theta_y * h_theta + theta * h_theta_y
    residual = (1.0 + lam) * y * theta_y - lam * theta + velocity_sign * nonlocal_term
    return np.asarray(residual, dtype=float)


def verify_ccf_residual(
    y: np.ndarray,
    theta: np.ndarray,
    theta_y: np.ndarray,
    lam: float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    forcing: np.ndarray | None = None,
) -> dict[str, float]:
    """Recompute the residual independently and return max / RMS norms."""
    res = ccf_self_similar_residual(
        y, theta, theta_y, lam, form=form, velocity_sign=velocity_sign
    )
    if forcing is not None:
        res = res - np.asarray(forcing, dtype=float)
    return {
        "max_abs_residual": float(np.max(np.abs(res))),
        "rms_residual": float(np.sqrt(np.mean(res * res))),
    }


def verify_cap_bundle(bundle: dict[str, Any], *, atol: float = 1e-8) -> dict[str, Any]:
    """Independently recompute a CAP bundle's residual and compare to stored.

    Reads only ``bundle['validation_inputs']`` (the network-free description) and
    ``bundle['residual_samples']``; returns agreement metrics. This is exactly
    the check an external interval-arithmetic verifier would perform first.
    """
    vin = bundle["validation_inputs"]
    y = np.asarray(vin["y"], dtype=float)
    theta = np.asarray(vin["theta"], dtype=float)
    theta_y = np.asarray(vin["theta_y"], dtype=float)
    lam = float(vin["lambda"])
    form = vin.get("form", "transport")
    velocity_sign = float(vin.get("velocity_sign", 1.0))
    recomputed = ccf_self_similar_residual(
        y, theta, theta_y, lam, form=form, velocity_sign=velocity_sign
    )
    reported = np.asarray(bundle["residual_samples"], dtype=float)
    diff = float(np.max(np.abs(recomputed - reported)))
    return {
        "recomputed_max_abs": float(np.max(np.abs(recomputed))),
        "reported_max_abs": float(np.max(np.abs(reported))),
        "agreement_max_abs_diff": diff,
        "residual_samples_match": bool(diff <= atol),
    }


def line_even_profile_jet(
    x: float, coeffs: np.ndarray, scales: np.ndarray
) -> tuple[float, float, float, float]:
    r"""Line even Poisson profile jet ``(Theta, Theta', H Theta, (H Theta)')``.

    Independent numpy reimplementation of the *exact* whole-line Hilbert pair
    ``p_a(x) = a/(x^2+a^2)`` (even) and ``H[p_a] = q_a(x) = x/(x^2+a^2)`` (odd)
    used by the verified CCF self-similar certificate. No FFT / periodic
    truncation -- this is the closed-form line transform, so it is a genuine
    second source for the certificate's interval values.
    """
    th = thp = hth = hthp = 0.0
    for c, a in zip(coeffs, scales, strict=True):
        d = x * x + a * a
        th += c * a / d
        thp += c * (-2.0 * a * x) / (d * d)
        hth += c * x / d
        hthp += c * (a * a - x * x) / (d * d)
    return float(th), float(thp), float(hth), float(hthp)


def ccf_self_similar_line_residual(
    x: float,
    coeffs: np.ndarray,
    scales: np.ndarray,
    lam: float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
) -> float:
    """Independent line (closed-form Hilbert) CCF self-similar residual at ``x``."""
    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}, got {form!r}")
    th, thp, hth, hthp = line_even_profile_jet(x, coeffs, scales)
    linear = (1.0 + lam) * x * thp - lam * th
    if form == "transport":
        nonlocal_term = velocity_sign * hth * thp
    else:
        nonlocal_term = velocity_sign * (thp * hth + th * hthp)
    return float(linear + nonlocal_term)


def _ccf_line_collocation_floats(
    coeffs: np.ndarray,
    scales: np.ndarray,
    lam: float,
    nodes: np.ndarray,
    form: str,
    s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Recompute ``(F, A, kappa2)`` of the normalized CCF collocation system.

    ``F`` is the residual at the nodes; ``A`` is the Jacobian w.r.t. the free
    unknowns ``(c_2..c_n, lambda)`` (``c_1`` fixed); ``kappa2`` is the max over
    nodes of the sum of absolute Hessian entries (the constant curvature bound).
    """
    n = len(scales)
    f = np.asarray(
        [ccf_self_similar_line_residual(y, coeffs, scales, lam, form=form, velocity_sign=s)
         for y in nodes],
        dtype=float,
    )
    a = np.zeros((len(nodes), n), dtype=float)
    kappa2 = 0.0
    for r, y in enumerate(nodes):
        th, thp, hth, hthp = line_even_profile_jet(y, coeffs, scales)
        p = scales / (y * y + scales**2)
        dp = -2.0 * scales * y / (y * y + scales**2) ** 2
        q = y / (y * y + scales**2)
        dq = (scales**2 - y * y) / (y * y + scales**2) ** 2
        col = 0
        for k in range(1, n):
            base = (1.0 + lam) * y * dp[k] - lam * p[k]
            if form == "transport":
                nl = s * (q[k] * thp + hth * dp[k])
            else:
                nl = s * (dp[k] * hth + thp * q[k] + p[k] * hthp + th * dq[k])
            a[r, col] = base + nl
            col += 1
        a[r, col] = y * thp - th
        # Hessian abs-sum at this node (cc block + 2 * c-lambda block).
        node_sum = 0.0
        for k in range(1, n):
            for j in range(1, n):
                if form == "transport":
                    hkj = s * (q[k] * dp[j] + q[j] * dp[k])
                else:
                    hkj = s * (dp[k] * q[j] + dp[j] * q[k] + p[k] * dq[j] + p[j] * dq[k])
                node_sum += abs(hkj)
        for k in range(1, n):
            node_sum += 2.0 * abs(y * dp[k] - p[k])
        kappa2 = max(kappa2, node_sum)
    return f, a, kappa2


def verify_ccf_selfsimilar_blowup_attempt(
    cert: dict[str, Any], *, rtol: float = 1e-6, atol: float = 1e-9
) -> dict[str, Any]:
    r"""Independent numpy replay of a CCF self-similar radii-polynomial certificate.

    Recomputes -- via a separate code path importing nothing from
    ``omnibias.pinn.certified`` -- the residual normal-form ``Y0``, the linear
    defect ``Z1 = ||I - B A||``, the curvature ``Z2 = ||B|| * sup|Hessian|`` and
    the radii-polynomial discriminant ``(1-Z1)^2 - 4 Z2 Y0`` from the
    certificate's own ``coeffs``/``scales``/``lambda``/``nodes``, then checks they
    agree with the reported ``closure_report`` and that the recomputed
    closed/blocked verdict matches ``closure_certified``. Returns a report with
    ``unproven_claim=False`` and an overall ``replay_match`` flag.
    """
    coeffs = np.asarray(cert["coeffs"], dtype=float)
    scales = np.asarray(cert["scales"], dtype=float)
    lam = float(cert["lambda_candidate"])
    nodes = np.asarray(cert["collocation_nodes"], dtype=float)
    form = str(cert.get("form", "transport"))
    s = float(cert.get("velocity_sign", 1.0))

    f, a, kappa2 = _ccf_line_collocation_floats(coeffs, scales, lam, nodes, form, s)
    try:
        b = np.linalg.inv(a)
        invertible = True
    except np.linalg.LinAlgError:
        b = np.zeros_like(a)
        invertible = False
    m = a.shape[0]
    y0 = float(np.max(np.abs(b @ f))) if m else 0.0
    z1 = float(np.max(np.abs(np.eye(m) - b @ a))) if m else 0.0
    norm_b = float(np.max(np.abs(b).sum(axis=1))) if m else 0.0
    z2 = norm_b * kappa2
    disc = (1.0 - z1) ** 2 - 4.0 * z2 * y0
    closed = bool(invertible and z1 < 1.0 and disc >= 0.0)

    report = cert.get("closure_report", {})
    rep_y0 = float(report.get("residual_normal_form_Y0", float("nan")))
    rep_z1 = float(report.get("linear_defect_Z1", float("nan")))
    rep_z2 = float(report.get("nonlinear_curvature_Z2", float("nan")))

    def _close(recomputed: float, reported: float) -> bool:
        return bool(abs(recomputed - reported) <= atol + rtol * abs(reported))

    y0_match = _close(y0, rep_y0)
    z1_match = bool(abs(z1 - rep_z1) <= 1e-6 + 1e-3 * abs(rep_z1) + 1e-9)
    z2_match = _close(z2, rep_z2)
    verdict_match = bool(closed == bool(cert.get("closure_certified", False)))
    residual_at_nodes_max = float(np.max(np.abs(f))) if m else 0.0

    replay_match = bool(y0_match and z1_match and z2_match and verdict_match)
    return {
        "recomputed_Y0": y0,
        "recomputed_Z1": z1,
        "recomputed_Z2": z2,
        "recomputed_discriminant": disc,
        "recomputed_closed": closed,
        "residual_at_nodes_max_abs": residual_at_nodes_max,
        "Y0_match": y0_match,
        "Z1_match": z1_match,
        "Z2_match": z2_match,
        "verdict_match": verdict_match,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


def verify_ccf_linearized_operator_bound(
    cert: dict[str, Any], *, rtol: float = 1e-6, atol: float = 1e-9
) -> dict[str, Any]:
    r"""Independent numpy replay of a CCF *continuum* linearized-operator certificate.

    Recomputes -- via a separate code path importing nothing from
    ``omnibias.pinn.certified`` -- the exact scaling resolvent
    ``kappa = 1/|1/2 + 3/2 lambda|``, the Neumann ratio
    ``rho = ||H Theta/y||_inf (1 + |lambda| kappa)/|1+lambda| + ||Theta'||_inf
    kappa`` from the certificate's *reported* sup-norms, and the inverse-norm
    bound ``kappa/(1-rho)``.  As an anti-faking guard it also **densely samples**
    the three even coefficient functions and checks the reported sup-norms
    genuinely dominate the samples (a forged, understated sup is caught).  Returns
    a report with ``unproven_claim=False`` and an overall ``replay_match`` flag.
    """
    if not cert.get("supported", False):
        return {
            "supported": False,
            "replay_match": True,
            "note": "flux form: continuum operator bound not computed; nothing to replay",
            "unproven_claim": False,
        }
    coeffs = np.asarray(cert["coeffs"], dtype=float)
    scales = np.asarray(cert["scales"], dtype=float)
    lam = float(cert["lambda_candidate"])
    s = float(cert["velocity_sign"])
    yt = float(cert["far_field_trunc"])

    shift = 0.5 + 1.5 * lam
    kappa = 1.0 / abs(shift) if abs(shift) > 1e-12 else float("inf")
    one_plus = 1.0 + lam

    y = np.linspace(0.0, yt, 20001)
    denom = y[:, None] ** 2 + scales[None, :] ** 2
    hthy = (coeffs[None, :] / denom).sum(axis=1)
    thp = (coeffs[None, :] * (-2.0 * scales[None, :] * y[:, None]) / denom**2).sum(axis=1)
    mult = (1.0 + lam) + s * hthy
    hthy_sampled = float(np.max(np.abs(hthy)))
    thp_sampled = float(np.max(np.abs(thp)))
    mult_sampled = float(np.max(np.abs(mult)))

    rep_hthy = float(cert["htheta_over_y_sup"])
    rep_thp = float(cert["theta_prime_sup"])
    rep_mult = float(cert["multiplier_sup"])
    sup_dominates = bool(
        hthy_sampled <= rep_hthy + atol
        and thp_sampled <= rep_thp + atol
        and mult_sampled <= rep_mult + atol
    )

    if abs(one_plus) > 1e-12 and np.isfinite(kappa):
        rho_recomputed = rep_hthy * (1.0 + abs(lam) * kappa) / abs(one_plus) + rep_thp * kappa
    else:
        rho_recomputed = float("inf")

    rep_kappa = cert["scaling_inverse_norm_bound"]
    kappa_match = bool(
        rep_kappa is not None
        and abs(kappa - float(rep_kappa)) <= atol + rtol * abs(float(rep_kappa))
    )
    rep_rho = cert["neumann_rho"]
    rho_match = bool(
        rep_rho is not None
        and abs(rho_recomputed - float(rep_rho)) <= atol + rtol * abs(float(rep_rho))
    )
    closed = bool(np.isfinite(rho_recomputed) and rho_recomputed < 1.0)
    verdict_match = bool(closed == bool(cert.get("rho_closes", False)))

    inv_match = True
    if cert.get("rho_closes") and cert.get("inverse_norm_bound") is not None:
        inv_recomputed = kappa / (1.0 - rho_recomputed)
        inv_match = bool(
            abs(inv_recomputed - float(cert["inverse_norm_bound"]))
            <= atol + rtol * abs(inv_recomputed)
        )

    replay_match = bool(
        kappa_match and rho_match and verdict_match and sup_dominates and inv_match
    )
    return {
        "supported": True,
        "recomputed_scaling_inverse_norm": kappa,
        "recomputed_rho": rho_recomputed,
        "sampled_htheta_over_y_sup": hthy_sampled,
        "sampled_theta_prime_sup": thp_sampled,
        "sampled_multiplier_sup": mult_sampled,
        "kappa_match": kappa_match,
        "rho_match": rho_match,
        "verdict_match": verdict_match,
        "sup_dominates_samples": sup_dominates,
        "inverse_norm_match": inv_match,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


def recover_ccf_scaling_law(
    y: np.ndarray,
    theta: np.ndarray,
    theta_y: np.ndarray,
    *,
    h_theta: np.ndarray | None = None,
    forcing: np.ndarray | None = None,
    alpha: float = 1e-10,
    threshold: float = 1e-10,
) -> dict[str, Any]:
    r"""Recover the CCF self-similar law + ``lambda`` from sampled jets.

    Fits ``Theta ~= c0*(y*Theta') + c1*(H Theta * Theta') [ + c2*forcing ]`` with
    :func:`omnibias.symbolic.fit_sparse_equation`. For the transport CCF relation
    the exact coefficients are ``c0 = (1+lambda)/lambda``, ``c1 = 1/lambda`` and
    (when a forcing column is supplied) ``c2 = -1/lambda``; ``lambda`` is read off
    as ``1/c1``. Returns the fitted equation, the recovered ``lambda``, and
    consistency diagnostics.
    """
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    theta_y = np.asarray(theta_y, dtype=float)
    if h_theta is None:
        h_theta = periodic_hilbert(theta)
    h_theta = np.asarray(h_theta, dtype=float)

    cols = [y * theta_y, h_theta * theta_y]
    names = ["y*theta_y", "Htheta*theta_y"]
    if forcing is not None:
        cols.append(np.asarray(forcing, dtype=float))
        names.append("forcing")
    design = np.stack(cols, axis=1)
    equation = fit_sparse_equation(design, theta, names, alpha=alpha, threshold=threshold)

    coeffs = {n: float(c) for n, c in zip(names, equation.coefficients, strict=False)}
    c_nonlocal = coeffs["Htheta*theta_y"]
    lam_recovered = float(1.0 / c_nonlocal) if abs(c_nonlocal) > 1e-30 else float("inf")
    # consistency: c0 should equal c1 + 1 (i.e. (1+lam)/lam == 1/lam + 1).
    c_advect = coeffs["y*theta_y"]
    consistency = float(abs(c_advect - (c_nonlocal + 1.0)))
    residual_of_fit = theta - equation.predict(design)
    return {
        "term_names": list(names),
        "coefficients": coeffs,
        "intercept": float(equation.intercept),
        "lambda_recovered": lam_recovered,
        "advection_consistency_abs": consistency,
        "fit_rmse": float(np.sqrt(np.mean(residual_of_fit**2))),
        "formula": equation.formula(lhs="theta"),
    }


def assess_ccf_candidate(
    y: np.ndarray,
    theta: np.ndarray,
    theta_y: np.ndarray,
    lam: float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    forcing: np.ndarray | None = None,
) -> dict[str, Any]:
    """One-shot honest assessment: residual norms + recovered scaling law."""
    out: dict[str, Any] = {
        "lambda_input": float(lam),
        "form": form,
        "residual": verify_ccf_residual(
            y, theta, theta_y, lam, form=form, velocity_sign=velocity_sign,
            forcing=forcing,
        ),
        "exact_solution_claim": False,
        "navier_stokes_proof_claim": False,
    }
    if form == "transport":
        out["scaling_law"] = recover_ccf_scaling_law(y, theta, theta_y, forcing=forcing)
    return out


__all__ = [
    "assess_ccf_candidate",
    "ccf_self_similar_line_residual",
    "ccf_self_similar_residual",
    "line_even_profile_jet",
    "periodic_hilbert",
    "recover_ccf_scaling_law",
    "verify_cap_bundle",
    "verify_ccf_residual",
    "verify_ccf_selfsimilar_blowup_attempt",
]

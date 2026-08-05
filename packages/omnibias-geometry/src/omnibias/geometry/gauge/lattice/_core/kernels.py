# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Backend-agnostic SU(2) lattice kernels (the deterministic Monte-Carlo math).

Every function takes the array module ``xp`` (``torch`` or ``jax.numpy``) as its
first argument and is written functionally (no in-place mutation), so the torch
and jax lattice backends share **one** implementation and are bit-identical twins
on identical inputs. The handful of operations whose keyword spelling differs
between the backends (``roll``/``stack``/``sum``/``norm``/``clamp``) go through
the small ``_is_torch``-dispatched shims below; each shim picks the *equivalent*
op, so the numerics are unchanged.

SU(2) links are stored as unit quaternions ``q = (q0, q1, q2, q3)`` (last axis).
A full link field has shape ``(4, *lattice_shape, 4)``; a single-direction field
has shape ``(*lattice_shape, 4)``.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# backend shims (pick the equivalent op; values are identical across backends)
# ---------------------------------------------------------------------------


def _is_torch(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "torch"


def _roll(xp: Any, x: Any, amount: int, axis: int) -> Any:
    if _is_torch(xp):
        return xp.roll(x, shifts=amount, dims=axis)
    return xp.roll(x, amount, axis=axis)


def _stack(xp: Any, parts: Any, axis: int) -> Any:
    if _is_torch(xp):
        return xp.stack(parts, dim=axis)
    return xp.stack(parts, axis=axis)


def _concat(xp: Any, parts: Any, axis: int) -> Any:
    if _is_torch(xp):
        return xp.cat(parts, dim=axis)
    return xp.concatenate(parts, axis=axis)


def _sum(xp: Any, x: Any, axes: Any) -> Any:
    return x.sum(dim=axes) if _is_torch(xp) else x.sum(axis=axes)


def _mean(xp: Any, x: Any, axes: Any) -> Any:
    return x.mean(dim=axes) if _is_torch(xp) else x.mean(axis=axes)


def _norm_last(xp: Any, x: Any, *, keepdim: bool) -> Any:
    if _is_torch(xp):
        return xp.linalg.vector_norm(x, dim=-1, keepdim=keepdim)
    return xp.linalg.norm(x, axis=-1, keepdims=keepdim)


def _clamp_min(xp: Any, x: Any, lo: float) -> Any:
    return xp.clamp(x, min=lo) if _is_torch(xp) else xp.maximum(x, lo)


def _complex(xp: Any, real: Any, imag: Any) -> Any:
    return xp.complex(real, imag) if _is_torch(xp) else real + 1j * imag


# ---------------------------------------------------------------------------
# quaternion algebra
# ---------------------------------------------------------------------------


def normalize_quaternion(xp: Any, q: Any) -> Any:
    """Project ``q`` onto the unit quaternion sphere (last axis = 4)."""
    norm = _clamp_min(xp, _norm_last(xp, q, keepdim=True), 1e-30)
    return q / norm


def quat_conj(xp: Any, q: Any) -> Any:
    """Quaternion conjugate / inverse for unit links."""
    w = q[..., :1]
    v = -q[..., 1:]
    return _concat(xp, (w, v), -1)


def quat_mul(xp: Any, a: Any, b: Any) -> Any:
    """Group product matching ``U(q_a q_b) = U(q_a) @ U(q_b)`` (left path order)."""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return _stack(
        xp,
        (
            b0 * a0 - b1 * a1 - b2 * a2 - b3 * a3,
            b0 * a1 + b1 * a0 + b2 * a3 - b3 * a2,
            b0 * a2 - b1 * a3 + b2 * a0 + b3 * a1,
            b0 * a3 + b1 * a2 - b2 * a1 + b3 * a0,
        ),
        -1,
    )


def quat_to_matrix(xp: Any, q: Any) -> Any:
    """Map unit quaternion ``(q0,q1,q2,q3)`` to a 2x2 complex SU(2) matrix."""
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    u00 = _complex(xp, q0, q3)
    u01 = _complex(xp, q2, q1)
    u10 = _complex(xp, -q2, q1)
    u11 = _complex(xp, q0, -q3)
    row0 = _stack(xp, (u00, u01), -1)
    row1 = _stack(xp, (u10, u11), -1)
    return _stack(xp, (row0, row1), -2)


def matrix_to_quat(xp: Any, u: Any) -> Any:
    """Project a 2x2 matrix onto the nearest unit quaternion (last two axes 2x2)."""
    q0 = 0.5 * (u[..., 0, 0].real + u[..., 1, 1].real)
    q1 = 0.5 * (u[..., 0, 1].real + u[..., 1, 0].real)
    q2 = 0.5 * (u[..., 0, 1].imag - u[..., 1, 0].imag)
    q3 = 0.5 * (u[..., 0, 0].imag - u[..., 1, 1].imag)
    return normalize_quaternion(xp, _stack(xp, (q0, q1, q2, q3), -1))


def _identity_quat_like(xp: Any, ref: Any) -> Any:
    """Identity quaternion broadcast to ``ref`` (shape ``(..., 4)``), no device kwargs."""
    w = ref[..., 0:1] * 0.0 + 1.0
    v = ref[..., 1:4] * 0.0
    return _concat(xp, (w, v), -1)


# ---------------------------------------------------------------------------
# shifts, staples
# ---------------------------------------------------------------------------


def shift(xp: Any, x: Any, mu: int, amount: int) -> Any:
    """Periodic shift along lattice axis ``mu`` (single-direction or full field)."""
    axis = mu if x.shape[-1] == 4 else mu + 1
    return _roll(xp, x, amount, axis)


def staple_sum(xp: Any, links: Any, mu: int) -> Any:
    """Sum of forward + backward staples for direction ``mu`` (quaternion-valued)."""
    acc = xp.zeros_like(links[mu])
    for nu in range(4):
        if nu == mu:
            continue
        u_nu_x = links[nu]
        u_mu_x_plus_nu = shift(xp, links[mu], nu, -1)
        u_nu_x_plus_mu = shift(xp, links[nu], mu, -1)
        forward = quat_mul(
            xp, quat_mul(xp, u_nu_x, u_mu_x_plus_nu), quat_conj(xp, u_nu_x_plus_mu)
        )

        u_nu_x_minus_nu = shift(xp, links[nu], nu, 1)
        u_mu_x_minus_nu = shift(xp, links[mu], nu, 1)
        u_nu_x_minus_nu_plus_mu = shift(xp, u_nu_x_minus_nu, mu, -1)
        backward = quat_mul(
            xp,
            quat_mul(xp, quat_conj(xp, u_nu_x_minus_nu), u_mu_x_minus_nu),
            u_nu_x_minus_nu_plus_mu,
        )
        acc = acc + forward + backward
    return acc


def staple_hat_and_magnitude(xp: Any, staple: Any) -> tuple[Any, Any]:
    """Return unit direction ``U_hat`` and scalar magnitude ``a = ||staple||``."""
    a = _norm_last(xp, staple, keepdim=False)
    safe = _clamp_min(xp, a, 1e-30)
    u_hat = staple / safe[..., None]
    return u_hat, a


# ---------------------------------------------------------------------------
# gauge transformations (for gauge-orbit / invariance checks)
# ---------------------------------------------------------------------------


def gauge_transform_links(xp: Any, links: Any, g: Any) -> Any:
    r"""Apply a lattice gauge transformation ``U_mu(x) -> g(x) U_mu(x) g(x+mu)^\dagger``.

    ``g`` is a site field of unit quaternions, shape ``(*lattice_shape, 4)``.
    Gauge-invariant observables (plaquette, Wilson/Polyakov loops) are unchanged.
    """
    out = []
    for mu in range(4):
        g_shift = quat_conj(xp, shift(xp, g, mu, -1))
        out.append(quat_mul(xp, quat_mul(xp, g, links[mu]), g_shift))
    return _stack(xp, out, 0)


# ---------------------------------------------------------------------------
# plaquette, Wilson loops, Polyakov loop
# ---------------------------------------------------------------------------


def plaquette_trace(xp: Any, links: Any, mu: int, nu: int) -> Any:
    """``(1/2) Re tr U_plaquette`` at each site (mu, nu must differ)."""
    u_mu = links[mu]
    u_nu = links[nu]
    plaq = quat_mul(
        xp,
        quat_mul(xp, u_mu, shift(xp, u_nu, mu, -1)),
        quat_mul(xp, quat_conj(xp, shift(xp, u_mu, nu, -1)), quat_conj(xp, u_nu)),
    )
    return plaq[..., 0]


def wilson_loop_trace(
    xp: Any, links: Any, mu: int, r_extent: int, t_extent: int, *, t_dir: int = 3
) -> Any:
    """Planar ``R x T`` Wilson loop trace ``(1/2) Re tr W`` in the ``mu``-temporal plane."""
    if r_extent < 1 or t_extent < 1:
        msg = f"Wilson extents must be >= 1, got R={r_extent}, T={t_extent}"
        raise ValueError(msg)

    loop = _identity_quat_like(xp, links[mu])
    for r in range(r_extent):
        loop = quat_mul(xp, loop, shift(xp, links[mu], mu, -r))
    for t in range(t_extent):
        u_t = shift(xp, shift(xp, links[t_dir], mu, -r_extent), t_dir, -t)
        loop = quat_mul(xp, loop, u_t)
    for r in range(r_extent - 1, -1, -1):
        u_mu = shift(xp, shift(xp, quat_conj(xp, links[mu]), mu, -r), t_dir, -t_extent)
        loop = quat_mul(xp, loop, u_mu)
    for t in range(t_extent - 1, -1, -1):
        u_t = quat_conj(xp, shift(xp, links[t_dir], t_dir, -t))
        loop = quat_mul(xp, loop, u_t)
    return loop[..., 0]


def polyakov_loop_field(xp: Any, links: Any, *, t_dir: int = 3) -> Any:
    """Per-site ``(1/2) Re tr P(x)`` of the Polyakov loop winding the time axis."""
    t_len = links[t_dir].shape[t_dir]
    loop = links[t_dir]
    for k in range(1, t_len):
        loop = quat_mul(xp, loop, shift(xp, links[t_dir], t_dir, -k))
    return loop[..., 0]


# ---------------------------------------------------------------------------
# APE smearing + 0++ glueball operator
# ---------------------------------------------------------------------------


def ape_smear_spatial_links(
    xp: Any, links: Any, *, n_steps: int = 10, alpha: float = 0.5
) -> Any:
    """APE smear spatial links (mu=0,1,2); project back to SU(2) by normalization."""
    spatial = (0, 1, 2)
    cur = links
    for _ in range(n_steps):
        new_dirs = []
        for mu in range(4):
            if mu not in spatial:
                new_dirs.append(cur[mu])
                continue
            staple = xp.zeros_like(cur[mu])
            for nu in spatial:
                if nu == mu:
                    continue
                u_nu = cur[nu]
                u_mu_fwd = shift(xp, cur[mu], nu, -1)
                u_nu_fwd = shift(xp, cur[nu], mu, -1)
                fwd = quat_mul(
                    xp, quat_mul(xp, u_nu, u_mu_fwd), quat_conj(xp, u_nu_fwd)
                )
                u_nu_bwd = shift(xp, cur[nu], nu, 1)
                u_mu_bwd = shift(xp, cur[mu], nu, 1)
                u_nu_bwd_fwd = shift(xp, u_nu_bwd, mu, -1)
                bwd = quat_mul(
                    xp, quat_mul(xp, quat_conj(xp, u_nu_bwd), u_mu_bwd), u_nu_bwd_fwd
                )
                staple = staple + fwd + bwd
            smeared = (1.0 - alpha) * cur[mu] + alpha * staple
            new_dirs.append(normalize_quaternion(xp, smeared))
        cur = _stack(xp, new_dirs, 0)
    return cur


def glueball_operator_timeslice(
    xp: Any,
    links: Any,
    *,
    smeared: bool = True,
    n_smear: int = 10,
    smear_alpha: float = 0.5,
) -> Any:
    """0++ operator ``O(t)``: spatial-volume sum of spatial plaquette traces per t."""
    field = (
        ape_smear_spatial_links(xp, links, n_steps=n_smear, alpha=smear_alpha)
        if smeared
        else links
    )
    spatial = (0, 1, 2)
    parts = []
    for mu in spatial:
        for nu in spatial:
            if nu <= mu:
                continue
            parts.append(_sum(xp, plaquette_trace(xp, field, mu, nu), (0, 1, 2)))
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


# ---------------------------------------------------------------------------
# correlators + GEVP
# ---------------------------------------------------------------------------


def raw_periodic_correlator_batch(xp: Any, o_samples: Any) -> Any:
    """Batch raw symmetrized correlators; ``o_samples`` shape ``(n_meas, T)``."""
    t_len = o_samples.shape[1]
    max_tau = t_len // 2
    rows = []
    for tau in range(max_tau + 1):
        fwd = _mean(xp, o_samples * _roll(xp, o_samples, -tau, 1), 1)
        bwd = _mean(xp, o_samples * _roll(xp, o_samples, tau, 1), 1)
        rows.append(0.5 * (fwd + bwd))
    return _stack(xp, rows, 1)


def raw_cross_correlator_batch(xp: Any, o_samples: Any) -> Any:
    """Raw symmetrized ``R_ab(tau)``; ``o_samples`` shape ``(n_meas, n_op, T)``."""
    t_len = o_samples.shape[2]
    max_tau = t_len // 2
    o_a = o_samples[:, :, None, :]
    parts = []
    for tau in range(max_tau + 1):
        o_b_fwd = _roll(xp, o_samples, -tau, 2)[:, None, :, :]
        fwd = _mean(xp, o_a * o_b_fwd, -1)
        o_b_bwd = _roll(xp, o_samples, tau, 2)[:, None, :, :]
        bwd = _mean(xp, o_a * o_b_bwd, -1)
        parts.append(0.5 * (fwd + bwd))
    return _stack(xp, parts, -1)


def _mask_without(xp: Any, n: int, i: int) -> Any:
    return xp.arange(n) != i


def connected_correlator_ensemble(xp: Any, o_samples: Any) -> tuple[Any, Any]:
    """Ensemble connected glueball correlator with global vacuum subtraction."""
    if o_samples.ndim != 2:
        msg = f"o_samples must have shape (n_meas, T), got {tuple(o_samples.shape)}"
        raise ValueError(msg)

    n_meas = o_samples.shape[0]
    r_stack = raw_periodic_correlator_batch(xp, o_samples)
    r_bar = _mean(xp, r_stack, 0)
    o_bar = o_samples.mean()
    connected = r_bar - o_bar * o_bar
    if n_meas < 2:
        return connected, xp.zeros_like(connected)

    r_total = _sum(xp, r_stack, 0)
    jk_rows = []
    for i in range(n_meas):
        r_loo = (r_total - r_stack[i]) / (n_meas - 1)
        o_loo = o_samples[_mask_without(xp, n_meas, i)].mean()
        jk_rows.append(r_loo - o_loo * o_loo)
    jk = _stack(xp, jk_rows, 0)
    jk_mean = _mean(xp, jk, 0)
    err = xp.sqrt((n_meas - 1) / n_meas * _sum(xp, (jk - jk_mean) ** 2, 0))
    return connected, err


def connected_correlator_matrix_ensemble(xp: Any, o_samples: Any) -> tuple[Any, Any]:
    """Connected operator matrix ``C_ab(tau)`` with global vacuum subtraction."""
    if o_samples.ndim != 3:
        msg = f"o_samples must have shape (n_meas, n_op, T), got {tuple(o_samples.shape)}"
        raise ValueError(msg)

    n_meas = o_samples.shape[0]
    r_stack = raw_cross_correlator_batch(xp, o_samples)
    r_bar = _mean(xp, r_stack, 0)
    o_bar = _mean(xp, o_samples, (0, 2))
    vacuum = o_bar[:, None] * o_bar[None, :]
    connected = r_bar - vacuum[..., None]
    if n_meas < 2:
        return connected, xp.zeros_like(connected)

    r_total = _sum(xp, r_stack, 0)
    jk_rows = []
    for i in range(n_meas):
        mask = _mask_without(xp, n_meas, i)
        r_loo = (r_total - r_stack[i]) / (n_meas - 1)
        o_loo = _mean(xp, o_samples[mask], (0, 2))
        vac_loo = o_loo[:, None] * o_loo[None, :]
        jk_rows.append(r_loo - vac_loo[..., None])
    jk = _stack(xp, jk_rows, 0)
    jk_mean = _mean(xp, jk, 0)
    err = xp.sqrt((n_meas - 1) / n_meas * _sum(xp, (jk - jk_mean) ** 2, 0))
    return connected, err


def gevp_ground_lambda(xp: Any, c0: Any, c1: Any, *, eps: float = 1e-10) -> float:
    """Largest generalized eigenvalue of ``C(t1) v = lam C(t0) v`` (symmetrized)."""
    c0_sym = 0.5 * (c0 + xp.swapaxes(c0, -1, -2))
    c1_sym = 0.5 * (c1 + xp.swapaxes(c1, -1, -2))
    evals, evecs = xp.linalg.eigh(c0_sym)
    inv_sqrt = evecs @ xp.diag(_clamp_min(xp, evals, eps) ** -0.5) @ xp.swapaxes(evecs, -1, -2)
    reduced = inv_sqrt @ c1_sym @ inv_sqrt
    lam = xp.linalg.eigvalsh(reduced)[-1]
    return float(lam)


def gevp_plateau(
    xp: Any,
    o_samples: Any,
    *,
    t0_values: Any,
    dt_values: Any,
    rel_tol: float = 0.25,
) -> dict[str, Any]:
    """Scan ``(t0, dt)`` GEVP ground masses and report a plateau estimate.

    For each ``(t0, dt)`` the GEVP eigenvalue ``lambda ~ exp(-m dt)`` gives a mass
    ``m = -ln(lambda)/dt``. Returns every accepted point, the median as the
    plateau estimate, the spread, and a ``stable`` flag (spread within
    ``rel_tol`` of the median). The plateau is fixed-spacing evidence, not a
    continuum mass.
    """
    c_mat, _ = connected_correlator_matrix_ensemble(xp, o_samples)
    max_tau = c_mat.shape[-1] - 1
    points: list[dict[str, float]] = []
    masses: list[float] = []
    for t0 in t0_values:
        for dt in dt_values:
            if dt < 1 or t0 < 0 or t0 + dt > max_tau:
                continue
            lam = gevp_ground_lambda(xp, c_mat[:, :, t0], c_mat[:, :, t0 + dt])
            if 0.0 < lam < 1.0:
                mass = -math.log(lam) / dt
                points.append({"t0": int(t0), "dt": int(dt), "mass": float(mass)})
                masses.append(float(mass))
    if not masses:
        return {
            "points": [],
            "plateau": {"value": float("nan"), "spread": float("nan")},
            "stable": False,
        }
    ordered = sorted(masses)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    spread = max(masses) - min(masses)
    stable = len(masses) >= 2 and median > 0.0 and spread <= rel_tol * median
    return {
        "points": points,
        "plateau": {"value": float(median), "spread": float(spread)},
        "stable": bool(stable),
    }


def exp_su2(xp: Any, omega: Any) -> Any:
    r"""Exponential map ``R^3 -> SU(2)``: ``omega -> (cos|omega|, sinc|omega| * omega)``.

    ``omega`` are su(2) coordinates (last axis = 3); the result is a unit
    quaternion. The geodesic / exponential map is exact (no re-projection), so a
    Langevin step built on it has no spherical re-normalisation bias.
    """
    theta = _norm_last(xp, omega, keepdim=False)
    theta_safe = _clamp_min(xp, theta, 1e-30)
    sinc = xp.where(theta > 1e-12, xp.sin(theta) / theta_safe, xp.ones_like(theta))
    w = xp.cos(theta)[..., None]
    v = sinc[..., None] * omega
    return _concat(xp, (w, v), -1)


def langevin_link_step(xp: Any, u: Any, sigma: Any, beta: float, eps: float, xi: Any) -> Any:
    r"""One geodesic Langevin step for a batch of SU(2) links.

    SU(2) is the unit 3-sphere with Haar = round measure, so a single link's
    Wilson-action weight ``exp(-S) = exp(beta * (U . Sigma))`` (``Sigma`` = staple
    sum) is sampled by overdamped Langevin dynamics on the group. The su(2) drift
    is the gradient of ``beta (U . Sigma)``, ``g = beta * Im(Sigma * conj(U))``,
    and the step ``U' = exp(eps g + sqrt(2 eps) xi) U`` uses the exact exponential
    map with a tangent Gaussian ``xi`` (shape ``(..., 3)``). This is the Parisi-Wu
    stochastic quantisation of the link; its stationary distribution converges to
    the Kennedy-Pendleton heat-bath distribution as ``eps -> 0``.
    """
    grad = beta * quat_mul(xp, sigma, quat_conj(xp, u))[..., 1:]
    omega = eps * grad + (2.0 * eps) ** 0.5 * xi
    return quat_mul(xp, exp_su2(xp, omega), u)


def gauge_orbit_distance(xp: Any, links_a: Any, links_b: Any) -> float:
    r"""Gauge-invariant orbit-distance proxy between two link configs.

    The RMS over all 6 plaquette planes / sites of the difference of the
    (gauge-invariant) plaquette traces ``(1/2) Re tr U_{mu nu}``. It is zero iff
    the two configs share the same plaquette content, and is invariant under a
    gauge transformation of either argument -- a cheap proxy for distance on the
    gauge-orbit space ``A / G``.
    """
    diffs = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            d = plaquette_trace(xp, links_a, mu, nu) - plaquette_trace(xp, links_b, mu, nu)
            diffs.append((d * d).mean())
    total = diffs[0]
    for part in diffs[1:]:
        total = total + part
    return float((total / len(diffs)) ** 0.5)


__all__ = [
    "ape_smear_spatial_links",
    "connected_correlator_ensemble",
    "connected_correlator_matrix_ensemble",
    "exp_su2",
    "gauge_orbit_distance",
    "gauge_transform_links",
    "gevp_ground_lambda",
    "gevp_plateau",
    "glueball_operator_timeslice",
    "langevin_link_step",
    "matrix_to_quat",
    "normalize_quaternion",
    "plaquette_trace",
    "polyakov_loop_field",
    "quat_conj",
    "quat_mul",
    "quat_to_matrix",
    "raw_cross_correlator_batch",
    "raw_periodic_correlator_batch",
    "shift",
    "staple_hat_and_magnitude",
    "staple_sum",
    "wilson_loop_trace",
]

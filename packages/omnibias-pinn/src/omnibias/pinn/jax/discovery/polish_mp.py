# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multiprecision Newton polish for Hardy-basis CCF coefficients.

Float64 / GPU round-off is the ceiling DeepMind names for CCF residuals
(``O(1e-13)``). A Hardy profile has tens of parameters, so an ``mpmath`` Newton
polish can push residuals below that floor on CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import mpmath as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError("mpmath is required for multiprecision CCF polish") from exc


def _hardy_even(y: object, a: object, alpha: object) -> object:
    r = mp.sqrt(a * a + y * y)
    phi = mp.atan(y / a)
    return (r ** (-alpha)) * mp.cos(alpha * phi)


def _hardy_odd(y: object, a: object, alpha: object) -> object:
    r = mp.sqrt(a * a + y * y)
    phi = mp.atan(y / a)
    return (r ** (-alpha)) * mp.sin(alpha * phi)


def _profile(
    y: object, coeffs: list[object], scales: list[object], alpha: object
) -> tuple[object, object, object, object]:
    th = mp.mpf(0)
    thp = mp.mpf(0)
    hth = mp.mpf(0)
    hthp = mp.mpf(0)
    for c, a in zip(coeffs, scales, strict=True):
        th += c * _hardy_even(y, a, alpha)
        thp += c * (-alpha) * _hardy_odd(y, a, alpha + 1)
        hth += c * _hardy_odd(y, a, alpha)
        hthp += c * alpha * _hardy_even(y, a, alpha + 1)
    return th, thp, hth, hthp


def _residual_at(
    y: object,
    coeffs: list[object],
    scales: list[object],
    lam: object,
    *,
    form: str,
    velocity_sign: float,
) -> object:
    alpha = 1 / (1 + lam)
    th, thp, hth, hthp = _profile(y, coeffs, scales, alpha)
    linear = (1 + lam) * y * thp - lam * th
    if form == "transport":
        nonlocal_term = hth * thp
    else:
        nonlocal_term = thp * hth + th * hthp
    return linear + velocity_sign * nonlocal_term


def polish_hardy_ccf(
    *,
    coeffs: np.ndarray,
    scales: np.ndarray,
    lam: float,
    nodes: np.ndarray,
    form: str = "transport",
    velocity_sign: float = 1.0,
    dps: int = 50,
    max_iter: int = 20,
    fix_c0: bool = True,
    free_lam: bool = True,
) -> dict[str, Any]:
    """Newton-polish free unknowns ``(c_1..c_{n-1}, log a_i[, lam])`` in mpmath.

    Collocates the CCF residual at ``nodes``. Returns float64 and multiprecision
    residual reports separately. When ``free_lam`` is False, ``lambda`` is held
    fixed (required for published-digit Rung-1 anti-circularity).
    """
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    scales = np.asarray(scales, dtype=float).reshape(-1)
    nodes = np.asarray(nodes, dtype=float).reshape(-1)
    if coeffs.shape != scales.shape:
        raise ValueError("coeffs and scales shape mismatch")
    n = int(coeffs.shape[0])
    if n < 1:
        raise ValueError("need at least one Hardy term")

    with mp.workdps(int(dps)):
        c_mp = [mp.mpf(float(c)) for c in coeffs]
        a_mp = [mp.mpf(float(a)) for a in scales]
        lam_mp = mp.mpf(float(lam))
        y_mp = [mp.mpf(float(y)) for y in nodes]

        # free unknowns: c[1:], log(a[:]), optional lam  (optionally freeze c[0])
        def pack() -> list[object]:
            xs: list[object] = []
            start = 1 if fix_c0 else 0
            xs.extend(c_mp[start:])
            xs.extend([mp.log(a) for a in a_mp])
            if free_lam:
                xs.append(lam_mp)
            return xs

        def unpack(xs: list[object]) -> None:
            nonlocal lam_mp
            start = 1 if fix_c0 else 0
            k = 0
            for i in range(start, n):
                c_mp[i] = xs[k]
                k += 1
            for i in range(n):
                a_mp[i] = mp.exp(xs[k])
                k += 1
            if free_lam:
                lam_mp = xs[k]

        def residual_vec(xs: list[object]) -> list[object]:
            unpack(xs)
            return [
                _residual_at(
                    y, c_mp, a_mp, lam_mp, form=form, velocity_sign=velocity_sign
                )
                for y in y_mp
            ]

        xs = pack()
        m = len(y_mp)
        # Pad / truncate unknowns to match collocation count via least squares.
        for _ in range(int(max_iter)):
            r = residual_vec(xs)
            # numerical Jacobian
            n_u = len(xs)
            jac = [[mp.mpf(0) for _ in range(n_u)] for _ in range(m)]
            eps = mp.mpf("1e-20")
            for j in range(n_u):
                xs_p = list(xs)
                xs_p[j] = xs_p[j] + eps
                rp = residual_vec(xs_p)
                for i in range(m):
                    jac[i][j] = (rp[i] - r[i]) / eps
            # Solve least-squares J delta = -r via normal equations
            jtj = [[mp.mpf(0) for _ in range(n_u)] for _ in range(n_u)]
            jtr = [mp.mpf(0) for _ in range(n_u)]
            for j in range(n_u):
                for k in range(n_u):
                    s = mp.mpf(0)
                    for i in range(m):
                        s += jac[i][j] * jac[i][k]
                    jtj[j][k] = s
                s = mp.mpf(0)
                for i in range(m):
                    s += jac[i][j] * r[i]
                jtr[j] = -s
            try:
                delta = mp.lu_solve(jtj, jtr)
            except Exception:
                break
            xs = [xs[j] + delta[j] for j in range(n_u)]
            if max(abs(delta[j]) for j in range(n_u)) < mp.mpf("1e-25"):
                break

        unpack(xs)
        r_final = residual_vec(xs)
        max_abs_mp = float(max(abs(ri) for ri in r_final)) if r_final else float("nan")

    # float64 residual with polished coeffs
    from omnibias.symbolic.ccf import hardy_profile_numpy, ccf_self_similar_residual

    alpha = 1.0 / (1.0 + float(lam_mp))
    c_f = np.array([float(c) for c in c_mp], dtype=float)
    a_f = np.array([float(a) for a in a_mp], dtype=float)
    th, thp, hth, hthp = hardy_profile_numpy(nodes, c_f, a_f, alpha)
    r64 = ccf_self_similar_residual(
        nodes,
        th,
        thp,
        float(lam_mp),
        form=form,
        velocity_sign=velocity_sign,
        hilbert_convention="hardy_exact",
        hilbert_values=hth,
        hilbert_y_values=hthp,
    )
    return {
        "coeffs": c_f,
        "scales": a_f,
        "lam": float(lam_mp),
        "alpha": alpha,
        "max_abs_residual_float64": float(np.max(np.abs(r64))),
        "max_abs_residual_mpmath": max_abs_mp,
        "dps": int(dps),
        "nodes": nodes,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "notes": "Multiprecision Newton polish on Hardy coeffs; not a CAP.",
        },
    }


__all__ = ["polish_hardy_ccf"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Inverse imaging (theory 05-01): product API G1–G6 plus Wave-0 G7.

G7 is the locally-seeded ``alpha^(n - 5/2)`` scaling falsifier. G1–G6
exercise ``omnibias.pinn.inverse``. Do not merge the Krawczyk enclosure
with a conformal slab. ``alpha`` is a tempering scale.

Estimator
---------
On ``[0, 1]`` with interface at ``tau* = 0.37``, a jump of size ``J`` in the
``m``-th derivative is probed by channel ``n = m + 2``. Orders ``n in {3, 4}``
use two different fields (not one field probed twice):

* ``n = 3``, ``m = 1``: piecewise-linear kink
* ``n = 4``, ``m = 2``: piecewise-quadratic kink

Response ``r_n(tau) = (1/N) sum_i y_i alpha^n sigma^(n)(alpha (x_i - tau))``
with ``y_i = u(x_i) + eps_i``. Peak polish is closed-form Newton on
``r_n'(tau) = 0`` using ``r_n''`` from ``sigma^(n+1)``, ``sigma^(n+2)``.

The gated estimator is **locally seeded**: the coarse grid is centred on the
true ``tau*`` (``SEED_HALFWIDTH_KERNELS / alpha``). A ``tau*``-free global
argmax arm is also measured and reported; it fails for ``n = 4`` at large
``alpha`` because a spurious boundary maximum near ``tau -> 1`` dominates.

Pre-registered regime (never tuned until the fit works)
-------------------------------------------------------
* ``alpha = geomspace(20, 320, 13)`` — 1.2 decades.
* ``N = 20 * alpha_max = 6400`` — design density: >= 20 samples per kernel width.
* Jump ``J = 50``. Noise ``s`` per order derived as
  ``round_down_1sig(0.5 * s_max)`` from
  ``rho(alpha_max) = sd_pred(tau_hat) * alpha_max <= 0.25``:
  ``s_3 = 0.05``, ``s_4 = 1e-4``.
* Boundary contamination ratio
  ``|u(endpoint) alpha^(n-1) sigma^(n-1)(alpha(endpoint-tau*))|
  / (J alpha |sigma'(0)|) <= 1e-2`` at every alpha in the sweep.

Modes
-----
* default (smoke) and ``--full``: both run ``R = 128`` realizations per seed
  over ``SEEDS = (0, 1, 2, 3, 4)``. The gate is the worst-seed fitted
  exponent via ``require_all_seeds``. ``--full`` also copies the artifact
  under ``$OMNIBIAS_SCRATCH/inverse/``.

``N`` is identical in both tiers so the smoke is not a different experiment.
``alpha`` is the scan tempering scale — not bias collapse ``delta -> 0`` and
not temperature collapse ``beta -> inf``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from omnibias.core.polynomials import sigmoid_polynomial_coeffs

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # type: ignore[import-not-found]  # noqa: E402
    provenance,
    write_json,
)
from _gates import (  # type: ignore[import-not-found]  # noqa: E402
    gates_block,
    require_all_seeds,
    require_capture_rate,
    require_rel_error,
    require_scaling_exponent,
    require_within_stderr,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

TAU_STAR = 0.37
DOMAIN = (0.0, 1.0)
ALPHA_MIN = 20.0
ALPHA_MAX = 320.0
N_ALPHAS = 13
ALPHA_SWEEP = np.geomspace(ALPHA_MIN, ALPHA_MAX, N_ALPHAS)
SAMPLES_PER_KERNEL_WIDTH = 20
N_SAMPLES = int(SAMPLES_PER_KERNEL_WIDTH * ALPHA_MAX)  # 6400
JUMP = 50.0
RHO_MAX = 0.25
BOUNDARY_RATIO_MAX = 1e-2
SIGMA3_AT_ZERO = -0.125  # sigma'''(0) = -1/8
SIGMA1_AT_ZERO = 0.25  # sigma'(0) = 1/4
L2SQ_SIGMA3 = 1.0 / 42.0
L2SQ_SIGMA4 = 1.0 / 30.0
EXPONENT_TOL = 0.1
MIN_DECADES = 1.0
ANCHOR_ALPHA = 80.0  # discrete-sd prediction vs measurement
SEED_HALFWIDTH_KERNELS = 5.0
SEED_GRID = 81
GLOBAL_PTS_PER_WIDTH = 3.0
GLOBAL_MARGIN_KERNELS = 7.0
NEWTON_ITERS = 8
SEEDS = (0, 1, 2, 3, 4)
R_PER_SEED = 128


def _l2_sq_from_coeffs(n: int, *, nodes: int = 800) -> float:
    """``||sigma^(n)||_2^2 = int_R [P_n(sigma(y))]^2 dy`` via ``t = sigma(y)``."""
    coeffs = sigmoid_polynomial_coeffs(n)
    xg, wg = np.polynomial.legendre.leggauss(int(nodes))
    t = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    poly = np.zeros_like(t)
    for i, a in enumerate(coeffs):
        poly = poly + float(a) * t**i
    return float(np.sum(w * poly * poly / (t * (1.0 - t))))


def _continuum_C(n: int) -> float:
    """C_n = ||sigma^(n+1)||_2 / |sigma'''(0)|."""
    return math.sqrt(_l2_sq_from_coeffs(n + 1)) / abs(SIGMA3_AT_ZERO)


def _round_down_1sig(v: float) -> float:
    """Largest 1-significant-digit number <= ``v`` (positive)."""
    if v <= 0.0:
        raise ValueError(f"round_down_1sig requires positive v, got {v}")
    exp = math.floor(math.log10(v))
    mant = math.floor(v / (10.0**exp))
    return float(mant * (10.0**exp))


def _s_from_regime(n: int, *, J: float, N: int, alpha_max: float) -> float:
    """``round_down_1sig(0.5 * s_max)`` with ``rho(alpha_max) <= RHO_MAX``."""
    C = _continuum_C(n)
    s_max = RHO_MAX * J * math.sqrt(N) / (C * alpha_max ** (n - 1.5))
    s = _round_down_1sig(0.5 * s_max)
    if s > s_max:
        raise RuntimeError(
            f"n={n}: derived s={s} exceeds regime bound s_max={s_max}; "
            "INVALID EXPERIMENT"
        )
    return s


ORDER_CONFIG: dict[int, dict[str, float]] = {
    3: {
        "m": 1,
        "J": JUMP,
        "s": _s_from_regime(3, J=JUMP, N=N_SAMPLES, alpha_max=ALPHA_MAX),
        "expected_exponent": 3.0 - 2.5,
    },
    4: {
        "m": 2,
        "J": JUMP,
        "s": _s_from_regime(4, J=JUMP, N=N_SAMPLES, alpha_max=ALPHA_MAX),
        "expected_exponent": 4.0 - 2.5,
    },
}


def sigma_n(z: Any, n: int) -> Any:
    """Closed-form ``sigma^(n)(z)`` via the shared Eulerian polynomial in ``s``."""
    zz = np.asarray(z, dtype=float)
    s = 1.0 / (1.0 + np.exp(-zz))
    coeffs = sigmoid_polynomial_coeffs(n)
    out = np.zeros_like(s)
    for i, a in enumerate(coeffs):
        out = out + float(a) * s**i
    return out


def field_u(x: Any, *, m: int, J: float, tau: float = TAU_STAR) -> Any:
    """Piecewise polynomial with a jump of size ``J`` in the ``m``-th derivative."""
    xx = np.asarray(x, dtype=float)
    base = 0.5 * xx
    excess = xx - tau
    if m == 1:
        return np.where(xx >= tau, base + J * excess, base)
    if m == 2:
        return np.where(xx >= tau, base + 0.5 * J * excess * excess, base)
    raise ValueError(f"unsupported jump order m={m}")


def sample_grid(n: int = N_SAMPLES) -> np.ndarray:
    return np.linspace(DOMAIN[0], DOMAIN[1], int(n))


def response_kernel(
    x: np.ndarray,
    taus: np.ndarray,
    *,
    n: int,
    alpha: float,
) -> np.ndarray:
    """Kernel matrix ``K[i, j] = alpha^n sigma^(n)(alpha (x_i - tau_j)) / N``."""
    N = float(x.size)
    a = float(alpha)
    z = a * (x[:, None] - taus[None, :])
    return (a**n) * sigma_n(z, n) / N


def response_from_y(y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``R = Y @ K`` for ``Y`` shape ``(R, N)`` or vector ``(N,)``."""
    return np.asarray(y, dtype=float) @ K


def _admissible_band(alpha: float) -> tuple[float, float]:
    """Interior band that keeps the kernel support inside the domain."""
    a = float(alpha)
    margin = GLOBAL_MARGIN_KERNELS / a
    lo = max(DOMAIN[0], DOMAIN[0] + margin)
    hi = min(DOMAIN[1], DOMAIN[1] - margin)
    if hi <= lo:
        raise RuntimeError(
            f"admissible band empty at alpha={a}; INVALID EXPERIMENT"
        )
    return lo, hi


def polish_peak(
    y: np.ndarray,
    x: np.ndarray,
    *,
    n: int,
    alpha: float,
    tau_init: float,
    clamp: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """Newton on ``r'(tau) = 0``; returns ``(tau_hat, r_prime, r_second)``."""
    a = float(alpha)
    tau = float(tau_init)
    N = float(x.size)
    yy = np.asarray(y, dtype=float)
    lo, hi = clamp if clamp is not None else _admissible_band(a)
    for _ in range(NEWTON_ITERS):
        z = a * (x - tau)
        sn1 = sigma_n(z, n + 1)
        sn2 = sigma_n(z, n + 2)
        rp = -float(np.dot(yy, (a ** (n + 1)) * sn1) / N)
        rpp = float(np.dot(yy, (a ** (n + 2)) * sn2) / N)
        if abs(rpp) < 1e-30:
            break
        tau = float(np.clip(tau - rp / rpp, lo, hi))
    z = a * (x - tau)
    rp = -float(np.dot(yy, (a ** (n + 1)) * sigma_n(z, n + 1)) / N)
    rpp = float(np.dot(yy, (a ** (n + 2)) * sigma_n(z, n + 2)) / N)
    return tau, rp, rpp


def polish_peak_batch(
    Y: np.ndarray,
    x: np.ndarray,
    *,
    n: int,
    alpha: float,
    tau_init: np.ndarray,
    clamp: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Newton over realizations; clamped to the admissible band."""
    a = float(alpha)
    N = float(x.size)
    YY = np.asarray(Y, dtype=float)
    tau = np.asarray(tau_init, dtype=float).copy()
    lo, hi = clamp if clamp is not None else _admissible_band(a)
    for _ in range(NEWTON_ITERS):
        z = a * (x[None, :] - tau[:, None])
        sn1 = sigma_n(z, n + 1)
        sn2 = sigma_n(z, n + 2)
        rp = -np.sum(YY * ((a ** (n + 1)) * sn1), axis=1) / N
        rpp = np.sum(YY * ((a ** (n + 2)) * sn2), axis=1) / N
        step = np.where(np.abs(rpp) < 1e-30, 0.0, rp / rpp)
        tau = np.clip(tau - step, lo, hi)
    z = a * (x[None, :] - tau[:, None])
    rp = -np.sum(YY * ((a ** (n + 1)) * sigma_n(z, n + 1)), axis=1) / N
    rpp = np.sum(YY * ((a ** (n + 2)) * sigma_n(z, n + 2)), axis=1) / N
    return tau, rp, rpp


def localize_batch(
    Y: np.ndarray,
    x: np.ndarray,
    *,
    n: int,
    alpha: float,
    tau_star: float = TAU_STAR,
) -> dict[str, Any]:
    """Oracle-seeded coarse ``|r|`` argmax + Newton polish for a batch."""
    a = float(alpha)
    half = SEED_HALFWIDTH_KERNELS / a
    lo = max(DOMAIN[0], tau_star - half)
    hi = min(DOMAIN[1], tau_star + half)
    taus = np.linspace(lo, hi, SEED_GRID)
    K = response_kernel(x, taus, n=n, alpha=a)
    R = response_from_y(Y, K)  # (R, M)
    idx = np.argmax(np.abs(R), axis=1)
    tau_init = taus[idx]
    tau_hats, rps, rpps = polish_peak_batch(
        Y, x, n=n, alpha=a, tau_init=tau_init, clamp=(lo, hi)
    )
    width = 1.0 / a
    captured = np.abs(tau_hats - tau_star) <= width
    return {
        "tau_hats": tau_hats,
        "r_prime": rps,
        "r_second": rpps,
        "captured": captured,
        "n_captured": int(np.sum(captured)),
        "n_total": int(Y.shape[0]),
        "empirical_sd": float(np.std(tau_hats[captured], ddof=1))
        if int(np.sum(captured)) >= 2
        else float("nan"),
        "empirical_mean": float(np.mean(tau_hats[captured]))
        if int(np.sum(captured)) >= 1
        else float("nan"),
        "seeded": True,
        "seed_halfwidth_kernels": SEED_HALFWIDTH_KERNELS,
    }


def global_localize(
    Y: np.ndarray,
    x: np.ndarray,
    *,
    n: int,
    alpha: float,
    tau_star: float = TAU_STAR,
) -> dict[str, Any]:
    """``tau*``-free argmax over the admissible band (diagnostic, not gated)."""
    a = float(alpha)
    lo, hi = _admissible_band(a)
    n_pts = max(64, int(math.ceil(GLOBAL_PTS_PER_WIDTH * a * (hi - lo))))
    taus = np.linspace(lo, hi, n_pts)
    K = response_kernel(x, taus, n=n, alpha=a)
    R = response_from_y(Y, K)
    idx = np.argmax(np.abs(R), axis=1)
    tau_hats = taus[idx]
    width = 1.0 / a
    captured = np.abs(tau_hats - tau_star) <= width
    return {
        "tau_hats": tau_hats,
        "captured": captured,
        "n_captured": int(np.sum(captured)),
        "n_total": int(Y.shape[0]),
        "capture_rate": float(np.mean(captured)),
        "n_grid": n_pts,
        "band": [lo, hi],
    }


def predicted_sd_continuum(
    *,
    n: int,
    alpha: float,
    s: float,
    J: float,
    N: int,
) -> float:
    """``sd(tau_hat) = (s C_n / (J sqrt(N))) * alpha^(n - 5/2)``."""
    C = _continuum_C(n)
    return (s * C / (J * math.sqrt(N))) * (float(alpha) ** (n - 2.5))


def predicted_sd_rprime_discrete(
    x: np.ndarray,
    *,
    n: int,
    alpha: float,
    s: float,
    tau_star: float = TAU_STAR,
) -> float:
    """Exact discrete ``sd(r_n')`` at ``tau*`` for i.i.d. noise of std ``s``."""
    a = float(alpha)
    N = float(x.size)
    z = a * (x - tau_star)
    sn1 = sigma_n(z, n + 1)
    return s * (a ** (n + 1)) * float(np.linalg.norm(sn1)) / N


def mollifier_peak_response(
    taus: np.ndarray,
    *,
    n: int,
    alpha: float,
    J: float,
    tau_star: float = TAU_STAR,
) -> np.ndarray:
    """Noiseless closed form ``(-1)^(n-1) J alpha sigma'(alpha (tau - tau*))``."""
    a = float(alpha)
    sign = (-1.0) ** (n - 1)
    return sign * J * a * sigma_n(a * (taus - tau_star), 1)


def boundary_contamination_ratio(
    *,
    n: int,
    alpha: float,
    J: float,
    m: int,
    tau_star: float = TAU_STAR,
) -> float:
    """``max_endpoint |boundary term| / |mollifier peak|`` at ``tau*``."""
    a = float(alpha)
    peak = abs(J * a * SIGMA1_AT_ZERO)
    if peak <= 0.0:
        raise ValueError("peak height must be positive")
    # Endpoint field values of the piecewise-polynomial construction.
    u_left = float(field_u(np.array([DOMAIN[0]]), m=m, J=J, tau=tau_star)[0])
    u_right = float(field_u(np.array([DOMAIN[1]]), m=m, J=J, tau=tau_star)[0])
    left = abs(
        u_left * (a ** (n - 1)) * float(sigma_n(a * (DOMAIN[0] - tau_star), n - 1))
    )
    right = abs(
        u_right * (a ** (n - 1)) * float(sigma_n(a * (DOMAIN[1] - tau_star), n - 1))
    )
    return max(left, right) / peak


def validate_regime(
    *,
    n: int,
    s: float,
    J: float,
    N: int,
    alphas: np.ndarray,
    tau_star: float = TAU_STAR,
) -> dict[str, Any]:
    """Pre-registered inequalities; raises INVALID EXPERIMENT on violation."""
    m = int(ORDER_CONFIG[n]["m"])
    alpha_max = float(np.max(alphas))
    alpha_min = float(np.min(alphas))
    rho = predicted_sd_continuum(n=n, alpha=alpha_max, s=s, J=J, N=N) * alpha_max
    density = N / alpha_max
    ratio_min = boundary_contamination_ratio(
        n=n, alpha=alpha_min, J=J, m=m, tau_star=tau_star
    )
    ratio_max = boundary_contamination_ratio(
        n=n, alpha=alpha_max, J=J, m=m, tau_star=tau_star
    )
    if density < SAMPLES_PER_KERNEL_WIDTH:
        raise RuntimeError(
            f"n={n}: design density {density:.1f} < {SAMPLES_PER_KERNEL_WIDTH}; "
            "INVALID EXPERIMENT"
        )
    if rho > RHO_MAX:
        raise RuntimeError(
            f"n={n}: regime ratio rho(alpha_max)={rho:.4f} > {RHO_MAX}; "
            "INVALID EXPERIMENT"
        )
    if ratio_min > BOUNDARY_RATIO_MAX or ratio_max > BOUNDARY_RATIO_MAX:
        raise RuntimeError(
            f"n={n}: boundary contamination "
            f"ratio_min={ratio_min:.4e} ratio_max={ratio_max:.4e} "
            f"> {BOUNDARY_RATIO_MAX}; INVALID EXPERIMENT"
        )
    return {
        "rho_at_alpha_max": rho,
        "rho_max": RHO_MAX,
        "design_density": density,
        "boundary_ratio_at_alpha_min": ratio_min,
        "boundary_ratio_at_alpha_max": ratio_max,
        "boundary_ratio_max": BOUNDARY_RATIO_MAX,
    }


def _rng_seed(seed: int, n: int) -> int:
    return int(1000 * seed + n)


def _run_seed(
    *,
    n: int,
    seed: int,
    R: int,
    x: np.ndarray,
) -> dict[str, Any]:
    cfg = ORDER_CONFIG[n]
    m = int(cfg["m"])
    J = float(cfg["J"])
    s = float(cfg["s"])
    expected = float(cfg["expected_exponent"])
    regime = validate_regime(n=n, s=s, J=J, N=int(x.size), alphas=ALPHA_SWEEP)

    u = field_u(x, m=m, J=J)
    rng = np.random.default_rng(_rng_seed(seed, n))
    noise = rng.normal(0.0, s, size=(R, x.size))
    Y = u[None, :] + noise

    per_alpha: list[dict[str, Any]] = []
    alphas_ok: list[float] = []
    sds_ok: list[float] = []
    total_captured = 0
    total_realizations = 0
    global_per_alpha: list[dict[str, Any]] = []

    for alpha in ALPHA_SWEEP:
        a = float(alpha)
        loc = localize_batch(Y, x, n=n, alpha=a)
        require_capture_rate(
            loc["n_captured"],
            loc["n_total"],
            min_rate=1.0,
            name=f"capture_n{n}_seed{seed}_alpha_{a:g}",
        )
        total_captured += int(loc["n_captured"])
        total_realizations += int(loc["n_total"])
        pred = predicted_sd_continuum(n=n, alpha=a, s=s, J=J, N=int(x.size))
        emp = float(loc["empirical_sd"])
        glob = global_localize(Y, x, n=n, alpha=a)
        per_alpha.append(
            {
                "alpha": a,
                "empirical_sd": emp,
                "predicted_sd_continuum": pred,
                "empirical_mean": float(loc["empirical_mean"]),
                "n_captured": int(loc["n_captured"]),
                "n_total": int(loc["n_total"]),
                "rho": pred * a,
                "global_capture_rate": float(glob["capture_rate"]),
                "global_n_captured": int(glob["n_captured"]),
            }
        )
        global_per_alpha.append(
            {
                "alpha": a,
                "capture_rate": float(glob["capture_rate"]),
                "n_captured": int(glob["n_captured"]),
                "n_total": int(glob["n_total"]),
                "n_grid": int(glob["n_grid"]),
            }
        )
        alphas_ok.append(a)
        sds_ok.append(emp)

    # Discrete sd(r') prediction at the anchor alpha.
    anchor = float(ALPHA_SWEEP[np.argmin(np.abs(ALPHA_SWEEP - ANCHOR_ALPHA))])
    a = anchor
    Nf = float(x.size)
    z_star = a * (x - TAU_STAR)
    sn1 = sigma_n(z_star, n + 1)
    weights = -(a ** (n + 1)) * sn1 / Nf
    rprimes = Y @ weights
    emp_sd_rp = float(np.std(rprimes, ddof=1))
    pred_sd_rp = predicted_sd_rprime_discrete(x, n=n, alpha=a, s=s)
    sd_of_sd = emp_sd_rp / math.sqrt(2.0 * (R - 1))
    continuum_sd_rp = (
        s * (a ** (n + 0.5)) * math.sqrt(_l2_sq_from_coeffs(n + 1)) / math.sqrt(Nf)
    )
    # Reuse the already-computed per_alpha entry at the anchor.
    anchor_row = next(row for row in per_alpha if abs(row["alpha"] - a) < 1e-12)

    slope_verdict = require_scaling_exponent(
        alphas_ok,
        sds_ok,
        expected=expected,
        tol=EXPONENT_TOL,
        min_decades=MIN_DECADES,
        name=f"localization_exponent_n{n}_seed{seed}",
    )

    return {
        "n": n,
        "seed": seed,
        "rng_seed": _rng_seed(seed, n),
        "m": m,
        "J": J,
        "s": s,
        "expected_exponent": expected,
        "fitted_exponent": float(slope_verdict["fitted_exponent"]),
        "regime": regime,
        "per_alpha": per_alpha,
        "global_per_alpha": global_per_alpha,
        "alphas": alphas_ok,
        "empirical_sds": sds_ok,
        "capture_totals": {
            "n_captured": total_captured,
            "n_total": total_realizations,
        },
        "global_min_capture_rate": float(
            min(row["capture_rate"] for row in global_per_alpha)
        ),
        "anchor": {
            "alpha": a,
            "empirical_sd_rprime": emp_sd_rp,
            "predicted_sd_rprime_discrete": pred_sd_rp,
            "sd_of_sd": sd_of_sd,
            "continuum_sd_rprime": continuum_sd_rp,
            "continuum_vs_discrete_rel": abs(continuum_sd_rp - pred_sd_rp)
            / pred_sd_rp,
            "empirical_sd_tau": float(anchor_row["empirical_sd"]),
        },
    }


def _run_gates(
    order_seed_results: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    entries.append(
        require_rel_error(
            _l2_sq_from_coeffs(3),
            L2SQ_SIGMA3,
            max_rel=1e-10,
            name="l2sq_sigma3_one_over_42",
        )
    )
    entries.append(
        require_rel_error(
            _l2_sq_from_coeffs(4),
            L2SQ_SIGMA4,
            max_rel=1e-10,
            name="l2sq_sigma4_one_over_30",
        )
    )

    for n, seed_rows in order_seed_results.items():
        # Discrete prediction vs measured sd(r') at the anchor, seed 0.
        anchor = seed_rows[0]["anchor"]
        entries.append(
            require_within_stderr(
                float(anchor["predicted_sd_rprime_discrete"]),
                float(anchor["empirical_sd_rprime"]),
                float(anchor["sd_of_sd"]),
                max_sigmas=3.0,
                name=f"discrete_sd_rprime_n{n}",
            )
        )
        entries.append(
            require_all_seeds(
                seed_rows,
                key="fitted_exponent",
                expected=float(ORDER_CONFIG[n]["expected_exponent"]),
                tol=EXPONENT_TOL,
                name=f"localization_exponent_n{n}",
            )
        )

    return dict(gates_block(entries))


def _run_product_gates(*, full: bool) -> dict[str, Any]:
    """G1–G6 on ``omnibias.pinn.inverse``. Founding G7 is unchanged."""
    from omnibias.core.transfer import Layer, reflection_transmission, stack_matrix
    from omnibias.core.verified.interval import Interval
    from omnibias.pinn.inverse import (
        SUBMODULAR_GUARANTEE,
        GuaranteeKindError,
        LayerStack,
        identify_jump_order,
        invert_layered,
        level_set_front,
        locate_interface,
        merge_guarantees,
        piecewise_jump_field,
        place_sensors,
        stefan_front,
        stefan_temperature,
        track_free_boundary,
        tv_deconvolution_localize,
    )

    entries: list[dict[str, Any]] = []
    x = np.linspace(0.0, 1.0, 201)
    tau_star = 0.37
    # G1: ensemble mean |err| over 10 seeds x 3 noise levels.
    scan_err: list[float] = []
    tv_err: list[float] = []
    for seed in range(10):
        for s in (0.01, 0.03, 0.06):
            rng = np.random.default_rng(10 * seed + int(100 * s))
            y = piecewise_jump_field(x, jump_order=1, jump=2.0, tau=tau_star)
            y = y + rng.normal(0.0, s, size=x.size)
            est = locate_interface(x, y, order=3, alpha=25.0)
            scan_err.append(abs(est.tau_hat - tau_star))
            tv_err.append(abs(tv_deconvolution_localize(x, y) - tau_star))
    g1_ok = float(np.mean(scan_err)) <= 0.5 * float(np.mean(tv_err))
    entries.append(
        {
            "name": "g1_localization_vs_tv",
            "passed": g1_ok,
            "mean_scan": float(np.mean(scan_err)),
            "mean_tv": float(np.mean(tv_err)),
            "baseline": "1-D ROF TV deconvolution + |D^2| peak",
        }
    )
    # G2: 1000 noiseless interior instances; one miss is a bug.
    n_g2 = 1000 if full else 1000
    misses = 0
    unique_ok = 0
    for i in range(n_g2):
        rng = np.random.default_rng(4000 + i)
        tstar = float(rng.uniform(0.38, 0.62))
        jump = float(rng.uniform(1.5, 3.0))
        alpha = float(rng.uniform(22.0, 32.0))
        xx = np.linspace(0.0, 1.0, 161)
        yy = piecewise_jump_field(xx, jump_order=1, jump=jump, tau=tstar)
        est = locate_interface(xx, yy, order=3, alpha=alpha)
        unique_ok += int(est.unique_in_enclosure)
        misses += int(not est.location.contains(tstar))
    g2_ok = misses == 0 and unique_ok == n_g2
    entries.append(
        {
            "name": "g2_enclosure_soundness",
            "passed": g2_ok,
            "n": n_g2,
            "misses": misses,
            "unique": unique_ok,
        }
    )
    try:
        merge_guarantees(Interval(0.36, 0.38), {"kind": "conformal"})
        merge_raised = False
    except GuaranteeKindError:
        merge_raised = True
    entries.append(
        {
            "name": "g2_kinds_unmerged",
            "passed": merge_raised,
        }
    )
    # G3: jump-order confusion at SNR >= 10.
    n_g3 = 120 if full else 80
    confusion = np.zeros((3, 3), dtype=int)
    for i in range(n_g3):
        rng = np.random.default_rng(5000 + i)
        m = int(rng.integers(0, 3))
        yy = piecewise_jump_field(x, jump_order=m, jump=6.0, tau=0.45)
        yy = yy + rng.normal(0.0, 0.002, size=x.size)
        ch, _scores = identify_jump_order(x, yy, alpha=22.0)
        pred = min(max(int(ch) - 2, 0), 2)
        confusion[m, pred] += 1
    acc = float(np.trace(confusion)) / float(n_g3)
    entries.append(
        {
            "name": "g3_jump_order",
            "passed": acc >= 0.95,
            "accuracy": acc,
            "confusion": confusion.tolist(),
            "n": n_g3,
        }
    )
    # G4: five-layer split (known n invert d; known d invert n).
    true_layers = (
        Layer(1.4, 0.12),
        Layer(2.1, 0.08),
        Layer(1.5, 0.10),
        Layer(2.0, 0.09),
        Layer(1.6, 0.11),
    )
    omegas = np.linspace(0.8, 2.8, 32)
    r_obs = np.array(
        [reflection_transmission(stack_matrix(true_layers, float(w)))[0] for w in omegas],
        dtype=np.complex128,
    )
    meas = {"omega": omegas, "r": r_obs}
    init_d = LayerStack(
        layers=tuple(Layer(layer.index, layer.thickness * 1.12) for layer in true_layers),
        n_iterations=0,
        residual=0.0,
    )
    an_d = invert_layered(meas, n_layers=5, init=init_d, analytic=True, free="thickness")
    fd_d = invert_layered(
        meas, n_layers=5, init=init_d, analytic=False, free="thickness", fd_eps=1e-2
    )
    d_err = max(
        abs(a.thickness - b.thickness) / b.thickness
        for a, b in zip(an_d.layers, true_layers, strict=True)
    )
    init_n = LayerStack(
        layers=tuple(
            Layer(complex(abs(layer.index) * 1.10, 0.0), layer.thickness)
            for layer in true_layers
        ),
        n_iterations=0,
        residual=0.0,
    )
    an_n = invert_layered(meas, n_layers=5, init=init_n, analytic=True, free="index")
    fd_n = invert_layered(
        meas, n_layers=5, init=init_n, analytic=False, free="index", fd_eps=1e-2
    )
    n_err = max(
        abs(abs(a.index) - abs(b.index)) / abs(b.index)
        for a, b in zip(an_n.layers, true_layers, strict=True)
    )
    g4_ok = (
        d_err <= 0.01
        and n_err <= 0.05
        and an_d.n_iterations < fd_d.n_iterations
        and an_n.n_iterations < fd_n.n_iterations
    )
    entries.append(
        {
            "name": "g4_layered_inversion",
            "passed": g4_ok,
            "thickness_rel": float(d_err),
            "index_rel": float(n_err),
            "analytic_iters": [an_d.n_iterations, an_n.n_iterations],
            "fd_lm_iters": [fd_d.n_iterations, fd_n.n_iterations],
            "regularizer": "known_index_or_known_thickness",
        }
    )
    # G5: Stefan IFT tracker vs level-set.
    t_span = (0.2, 1.0)
    tr = track_free_boundary(
        (stefan_temperature, lambda _x, _t: 0.0), t_span=t_span, n_times=16
    )
    ls = level_set_front(t_span=t_span, n_times=16, n_space=81)
    truth = np.array([stefan_front(float(t)) for t in tr.times])
    e_tr = float(np.max(np.abs(tr.positions - truth)))
    e_ls = float(np.max(np.abs(ls.positions - truth)))
    entries.append(
        {
            "name": "g5_stefan_vs_level_set",
            "passed": e_tr <= 0.3 * e_ls,
            "tracker_max_abs": e_tr,
            "level_set_max_abs": e_ls,
            "baseline": "1-D upwind level-set",
        }
    )
    # G6: D-optimal vs uniform posterior volume.
    plan = place_sensors(
        "location",
        {"tau": 0.37, "alpha": 25.0},
        np.linspace(0.0, 1.0, 21),
        budget=4,
    )
    entries.append(
        {
            "name": "g6_sensor_design",
            "passed": plan.volume_ratio >= 2.0
            and abs(plan.guarantee - SUBMODULAR_GUARANTEE) < 1e-12,
            "volume_ratio": plan.volume_ratio,
            "guarantee": plan.guarantee,
            "baseline": "uniform 4-sensor placement",
        }
    )
    return {
        "entries": entries,
        "g1_mean_scan": float(np.mean(scan_err)),
        "g1_mean_tv": float(np.mean(tv_err)),
        "g2_misses": misses,
        "g3_accuracy": acc,
        "g3_confusion": confusion.tolist(),
        "g4_thickness_rel": float(d_err),
        "g4_index_rel": float(n_err),
        "g5_ratio": e_tr / max(e_ls, 1e-30),
        "g6_volume_ratio": plan.volume_ratio,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="acceptance artifact also copied under $OMNIBIAS_SCRATCH/inverse/",
    )
    args = parser.parse_args(argv)
    full = bool(args.full)
    R = R_PER_SEED
    artifact_name = (
        "inverse_imaging.json" if full else "inverse_imaging_smoke.json"
    )

    config = {
        "family": "logistic_scan_localization",
        "orders": [3, 4],
        "tau_star": TAU_STAR,
        "domain": list(DOMAIN),
        "alpha_sweep": {
            "lo": float(ALPHA_MIN),
            "hi": float(ALPHA_MAX),
            "n": int(N_ALPHAS),
            "decades": float(np.log10(ALPHA_MAX / ALPHA_MIN)),
        },
        "N": N_SAMPLES,
        "samples_per_kernel_width": SAMPLES_PER_KERNEL_WIDTH,
        "J": JUMP,
        "s_per_order": {str(n): ORDER_CONFIG[n]["s"] for n in (3, 4)},
        "rho_max": RHO_MAX,
        "boundary_ratio_max": BOUNDARY_RATIO_MAX,
        "R_per_seed": R,
        "full": full,
        "seeds": list(SEEDS),
        "exponent_tol": EXPONENT_TOL,
        "min_decades": MIN_DECADES,
        "anchor_alpha": ANCHOR_ALPHA,
        "seed_halfwidth_kernels": SEED_HALFWIDTH_KERNELS,
        "estimator": "locally_seeded_coarse_argmax_plus_newton",
        "g1_to_g6_earned": True,
        "pre_registration": (
            "N = 20 * alpha_max; rho(alpha_max) = sd_pred * alpha_max <= 0.25; "
            "boundary_contamination_ratio <= 1e-2 at alpha_min and alpha_max; "
            "s = round_down_1sig(0.5 * s_max) from the closed-form regime bound; "
            "never retuned after seeing the fit; five seeds gated by worst seed"
        ),
    }
    payload = provenance(schema="inverse-imaging-v1", config=config)

    t0 = time.perf_counter()
    x = sample_grid(N_SAMPLES)
    order_seed_results: dict[int, list[dict[str, Any]]] = {}
    for n in (3, 4):
        order_seed_results[n] = [
            _run_seed(n=n, seed=seed, R=R, x=x) for seed in SEEDS
        ]
    print("G1–G6 product API...")
    product = _run_product_gates(full=full)
    print("G7 alpha scaling...")
    gates_g7 = _run_gates(order_seed_results)
    entries = list(product["entries"]) + list(gates_g7["entries"])
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))

    fitted = {
        str(n): {
            "per_seed": [float(r["fitted_exponent"]) for r in rows],
            "worst_deviation": next(
                e["worst_deviation"]
                for e in gates["entries"]
                if e["name"] == f"localization_exponent_n{n}"
            ),
            "expected": float(ORDER_CONFIG[n]["expected_exponent"]),
        }
        for n, rows in order_seed_results.items()
    }

    # Global-search diagnostic: earned only if every seed/alpha captures.
    global_search_earned: dict[str, bool] = {}
    boundary_artifact: dict[str, Any] = {}
    for n, rows in order_seed_results.items():
        min_rate = min(float(r["global_min_capture_rate"]) for r in rows)
        global_search_earned[str(n)] = bool(min_rate >= 1.0)
        # Noiseless boundary vs peak magnitudes at alpha_max (documentation).
        u = field_u(x, m=int(ORDER_CONFIG[n]["m"]), J=JUMP)
        taus = np.linspace(*_admissible_band(ALPHA_MAX), 401)
        K = response_kernel(x, taus, n=n, alpha=ALPHA_MAX)
        r = response_from_y(u, K)
        peak = JUMP * ALPHA_MAX * SIGMA1_AT_ZERO
        boundary_artifact[str(n)] = {
            "alpha": ALPHA_MAX,
            "noiseless_peak": float(peak),
            "noiseless_abs_max": float(np.max(np.abs(r))),
            "noiseless_argmax_tau": float(taus[int(np.argmax(np.abs(r)))]),
            "global_min_capture_rate": min_rate,
        }

    payload.update(
        {
            "baseline": {
                "name": "continuum delta-method sd(tau_hat) ~ alpha^(n-5/2)",
                "note": (
                    "scan is not statistically efficient; Cramer-Rao is ~10x "
                    "tighter on this family (spec 05-01 section 5); gated "
                    "estimator is locally seeded on true tau*"
                ),
            },
            "seeds": list(SEEDS),
            "orders": [
                {
                    "n": n,
                    "m": int(ORDER_CONFIG[n]["m"]),
                    "J": JUMP,
                    "s": float(ORDER_CONFIG[n]["s"]),
                    "expected_exponent": float(ORDER_CONFIG[n]["expected_exponent"]),
                    "fitted": fitted[str(n)],
                    "regime": order_seed_results[n][0]["regime"],
                    "per_seed": [
                        {
                            "seed": r["seed"],
                            "rng_seed": r["rng_seed"],
                            "fitted_exponent": r["fitted_exponent"],
                            "capture_totals": r["capture_totals"],
                            "global_min_capture_rate": r["global_min_capture_rate"],
                            "anchor": r["anchor"],
                            "per_alpha": r["per_alpha"],
                            "global_per_alpha": r["global_per_alpha"],
                        }
                        for r in order_seed_results[n]
                    ],
                }
                for n in (3, 4)
            ],
            "gates": gates,
            "product": product,
            "honesty": {
                "claim_rung": 1,
                "family": "logistic_scan_localization",
                "tempering_scale_alpha": True,
                "bias_collapse": False,
                "temperature_collapse": False,
                "locally_seeded_estimator": True,
                "global_search_earned": global_search_earned,
                "boundary_artifact": boundary_artifact,
                "g1_earned": bool(product["entries"][0]["passed"]),
                "g2_earned": bool(product["entries"][1]["passed"]),
                "g3_earned": bool(product["entries"][3]["passed"]),
                "g4_earned": bool(product["entries"][4]["passed"]),
                "g5_earned": bool(product["entries"][5]["passed"]),
                "g6_earned": bool(product["entries"][6]["passed"]),
                "g7_earned": bool(
                    all(
                        e["passed"]
                        for e in gates["entries"]
                        if str(e["name"]).startswith("localization_exponent")
                    )
                ),
                "theorem_prover_verified": False,
                "mathlib_verified": False,
                "not_statistically_efficient": True,
                "no_pinn_inverse_module": False,
                "pre_registered": config["pre_registration"],
                "licensed_sentence": (
                    "for the locally-seeded logistic bias-scan localizer on "
                    "this synthetic piecewise-polynomial family, the "
                    "localization standard deviation scales as "
                    "alpha^(n - 5/2) for n in {3, 4}, measured over 1.2 "
                    "decades of tempering scale alpha across five seeds; "
                    "a tau*-free global argmax earns the same claim only "
                    "for n=3 (n=4 is dominated by a boundary artifact); "
                    "G1–G6 are earned on omnibias.pinn.inverse: scan "
                    "localization beats TV, the noiseless Krawczyk box "
                    "covers 1000/1000 instances and must not merge with a "
                    "conformal slab, jump order is identified at SNR>=10, "
                    "five-layer transfer inversion with a known-index or "
                    "known-thickness regularizer beats FD LM, the Stefan "
                    "IFT tracker beats a 1-D level-set, and D-optimal "
                    "sensors beat uniform placement by >=2x volume with "
                    "the (1-1/e) guarantee reported"
                ),
            },
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )

    out = write_json(artifact_name, payload)
    print(
        f"wrote {out}  all_passed={gates['all_passed']}  "
        f"fitted={fitted}  global_search={global_search_earned}"
    )

    if full:
        scratch_dir = SCRATCH / "inverse"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        scratch_path = scratch_dir / artifact_name
        scratch_path.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {scratch_path}")

    if not gates["all_passed"]:
        raise SystemExit(1)
    return dict(payload)


if __name__ == "__main__":
    main()

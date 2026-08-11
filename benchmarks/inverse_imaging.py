# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-0 falsifier A7: scan localization scaling ``alpha^(n - 5/2)`` (05-01 G7).

Only the ``alpha``-sweep family of ``benchmarks/inverse_imaging.py`` is
implemented here. Spec 05-01 gates G1–G6 remain unearned; no
``omnibias.pinn.inverse`` code ships in this unit.

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

Pre-registered regime (never tuned until the fit works)
-------------------------------------------------------
* ``alpha = geomspace(20, 320, 13)`` — 1.2 decades; ``alpha_min = 20`` from
  ``alpha * min(tau*, 1 - tau*) >= 7``.
* ``N = 20 * alpha_max = 6400`` — design density: >= 20 samples per kernel width.
* Jump ``J = 50``. Noise ``s`` per order from
  ``rho(alpha_max) = sd_pred(tau_hat) * alpha_max <= 0.25``:
  ``s_3 = 0.05``, ``s_4 = 1e-4``.

Modes
-----
* default (smoke): ``R = 128`` realizations; CI wiring gate.
* ``--full``: ``R = 512``; acceptance artifact also copied under
  ``$OMNIBIAS_SCRATCH/inverse/``.

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
SIGMA3_AT_ZERO = -0.125  # sigma'''(0) = -1/8
L2SQ_SIGMA3 = 1.0 / 42.0
L2SQ_SIGMA4 = 1.0 / 30.0
EXPONENT_TOL = 0.1
MIN_DECADES = 1.0
ANCHOR_ALPHA = 80.0  # discrete-sd prediction vs measurement
COARSE_HALFWIDTH_KERNELS = 5.0
COARSE_GRID = 81
NEWTON_ITERS = 8
SEED = 0

# Per-order noise from rho(alpha_max) <= RHO_MAX with the continuum prediction.
# Derived once, recorded in config; never retuned after seeing the fit.
def _continuum_C(n: int) -> float:
    """C_n = ||sigma^(n+1)||_2 / |sigma'''(0)|."""
    return math.sqrt(_l2_sq_from_coeffs(n + 1)) / abs(SIGMA3_AT_ZERO)


def _s_from_regime(n: int, *, J: float, N: int, alpha_max: float) -> float:
    """Largest round s with rho(alpha_max) <= RHO_MAX (floored, never retuned)."""
    C = _continuum_C(n)
    s_max = RHO_MAX * J * math.sqrt(N) / (C * alpha_max ** (n - 1.5))
    # Round down to a short decimal so the artifact records a clean constant.
    if n == 3:
        s = 0.05  # s_max ~ 0.119
    elif n == 4:
        s = 1e-4  # s_max ~ 2.48e-4
    else:
        raise ValueError(f"unsupported order n={n}")
    if s > s_max:
        raise RuntimeError(
            f"n={n}: rounded s={s} exceeds regime bound s_max={s_max}; "
            "INVALID EXPERIMENT"
        )
    return s


ORDER_CONFIG: dict[int, dict[str, float]] = {
    3: {"m": 1, "J": JUMP, "s": 0.05, "expected_exponent": 3.0 - 2.5},
    4: {"m": 2, "J": JUMP, "s": 1e-4, "expected_exponent": 4.0 - 2.5},
}


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
    # (N, M): x[:, None] - taus[None, :]
    z = a * (x[:, None] - taus[None, :])
    return (a**n) * sigma_n(z, n) / N


def response_from_y(y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``R = Y @ K`` for ``Y`` shape ``(R, N)`` or vector ``(N,)``."""
    return np.asarray(y, dtype=float) @ K


def polish_peak(
    y: np.ndarray,
    x: np.ndarray,
    *,
    n: int,
    alpha: float,
    tau_init: float,
) -> tuple[float, float, float]:
    """Newton on ``r'(tau) = 0``; returns ``(tau_hat, r_prime, r_second)``."""
    a = float(alpha)
    tau = float(tau_init)
    N = float(x.size)
    yy = np.asarray(y, dtype=float)
    for _ in range(NEWTON_ITERS):
        z = a * (x - tau)
        # r' = - (1/N) sum y alpha^(n+1) sigma^(n+1)
        # r'' = (1/N) sum y alpha^(n+2) sigma^(n+2)
        sn1 = sigma_n(z, n + 1)
        sn2 = sigma_n(z, n + 2)
        rp = -float(np.dot(yy, (a ** (n + 1)) * sn1) / N)
        rpp = float(np.dot(yy, (a ** (n + 2)) * sn2) / N)
        if abs(rpp) < 1e-30:
            break
        tau = tau - rp / rpp
    z = a * (x - tau)
    rp = -float(np.dot(yy, (a ** (n + 1)) * sigma_n(z, n + 1)) / N)
    rpp = float(np.dot(yy, (a ** (n + 2)) * sigma_n(z, n + 2)) / N)
    return tau, rp, rpp


def localize_batch(
    Y: np.ndarray,
    x: np.ndarray,
    *,
    n: int,
    alpha: float,
    tau_star: float = TAU_STAR,
) -> dict[str, Any]:
    """Coarse ``|r|`` argmax + Newton polish for a batch of realizations."""
    a = float(alpha)
    half = COARSE_HALFWIDTH_KERNELS / a
    lo = max(DOMAIN[0], tau_star - half)
    hi = min(DOMAIN[1], tau_star + half)
    taus = np.linspace(lo, hi, COARSE_GRID)
    K = response_kernel(x, taus, n=n, alpha=a)
    R = response_from_y(Y, K)  # (R, M)
    idx = np.argmax(np.abs(R), axis=1)
    tau_hats = np.empty(Y.shape[0], dtype=float)
    rps = np.empty(Y.shape[0], dtype=float)
    rpps = np.empty(Y.shape[0], dtype=float)
    for i in range(Y.shape[0]):
        th, rp, rpp = polish_peak(
            Y[i], x, n=n, alpha=a, tau_init=float(taus[idx[i]])
        )
        tau_hats[i] = th
        rps[i] = rp
        rpps[i] = rpp
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
    # r' = - (1/N) sum y_i * alpha^(n+1) sigma^(n+1)
    # Var = (s^2 / N^2) * sum (alpha^(n+1) sigma^(n+1))^2
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


def validate_regime(
    *,
    n: int,
    s: float,
    J: float,
    N: int,
    alphas: np.ndarray,
) -> dict[str, Any]:
    """Pre-registered inequalities; raises INVALID EXPERIMENT on violation."""
    alpha_max = float(np.max(alphas))
    rho = predicted_sd_continuum(n=n, alpha=alpha_max, s=s, J=J, N=N) * alpha_max
    density = N / alpha_max
    kernel_fit = alpha_max * min(TAU_STAR - DOMAIN[0], DOMAIN[1] - TAU_STAR)
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
    if float(np.min(alphas)) * min(TAU_STAR - DOMAIN[0], DOMAIN[1] - TAU_STAR) < 7.0:
        raise RuntimeError(
            f"n={n}: kernel does not fit in the domain at alpha_min; "
            "INVALID EXPERIMENT"
        )
    return {
        "rho_at_alpha_max": rho,
        "rho_max": RHO_MAX,
        "design_density": density,
        "kernel_fit_at_alpha_max": kernel_fit,
    }


def _run_order(
    *,
    n: int,
    R: int,
    seed: int,
    x: np.ndarray,
) -> dict[str, Any]:
    cfg = ORDER_CONFIG[n]
    m = int(cfg["m"])
    J = float(cfg["J"])
    s = float(cfg["s"])
    expected = float(cfg["expected_exponent"])
    regime = validate_regime(n=n, s=s, J=J, N=int(x.size), alphas=ALPHA_SWEEP)

    u = field_u(x, m=m, J=J)
    rng = np.random.default_rng(seed + n)
    noise = rng.normal(0.0, s, size=(R, x.size))
    Y = u[None, :] + noise

    per_alpha: list[dict[str, Any]] = []
    alphas_ok: list[float] = []
    sds_ok: list[float] = []
    total_captured = 0
    total_realizations = 0

    for alpha in ALPHA_SWEEP:
        a = float(alpha)
        loc = localize_batch(Y, x, n=n, alpha=a)
        require_capture_rate(
            loc["n_captured"],
            loc["n_total"],
            min_rate=1.0,
            name=f"capture_n{n}_alpha_{a:g}",
        )
        total_captured += int(loc["n_captured"])
        total_realizations += int(loc["n_total"])
        pred = predicted_sd_continuum(n=n, alpha=a, s=s, J=J, N=int(x.size))
        emp = float(loc["empirical_sd"])
        per_alpha.append(
            {
                "alpha": a,
                "empirical_sd": emp,
                "predicted_sd_continuum": pred,
                "empirical_mean": float(loc["empirical_mean"]),
                "n_captured": int(loc["n_captured"]),
                "n_total": int(loc["n_total"]),
                "rho": pred * a,
            }
        )
        alphas_ok.append(a)
        sds_ok.append(emp)

    # Discrete sd(r') prediction at the anchor alpha (closest in the sweep).
    anchor = float(ALPHA_SWEEP[np.argmin(np.abs(ALPHA_SWEEP - ANCHOR_ALPHA))])
    loc_anchor = localize_batch(Y, x, n=n, alpha=anchor)
    # Empirical sd of r' across realizations, evaluated at tau* (not at tau_hat):
    # rebuild r' at tau* for each y.
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

    return {
        "n": n,
        "m": m,
        "J": J,
        "s": s,
        "expected_exponent": expected,
        "regime": regime,
        "per_alpha": per_alpha,
        "alphas": alphas_ok,
        "empirical_sds": sds_ok,
        "capture_totals": {
            "n_captured": total_captured,
            "n_total": total_realizations,
        },
        "anchor": {
            "alpha": a,
            "empirical_sd_rprime": emp_sd_rp,
            "predicted_sd_rprime_discrete": pred_sd_rp,
            "sd_of_sd": sd_of_sd,
            "continuum_sd_rprime": continuum_sd_rp,
            "continuum_vs_discrete_rel": abs(continuum_sd_rp - pred_sd_rp)
            / pred_sd_rp,
            "empirical_sd_tau": float(loc_anchor["empirical_sd"]),
        },
    }


def _run_gates(order_results: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    # Exact rational L^2 constants from the shared Eulerian polynomials.
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

    for result in order_results:
        n = int(result["n"])
        # Discrete prediction vs measured sd(r') at the anchor.
        anchor = result["anchor"]
        entries.append(
            require_within_stderr(
                float(anchor["predicted_sd_rprime_discrete"]),
                float(anchor["empirical_sd_rprime"]),
                float(anchor["sd_of_sd"]),
                max_sigmas=3.0,
                name=f"discrete_sd_rprime_n{n}",
            )
        )
        # G7 proper.
        entries.append(
            require_scaling_exponent(
                result["alphas"],
                result["empirical_sds"],
                expected=float(result["expected_exponent"]),
                tol=EXPONENT_TOL,
                min_decades=MIN_DECADES,
                name=f"localization_exponent_n{n}",
            )
        )

    return dict(gates_block(entries))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="multi-realization acceptance with R=512",
    )
    args = parser.parse_args(argv)
    full = bool(args.full)
    R = 512 if full else 128
    artifact_name = (
        "inverse_imaging.json" if full else "inverse_imaging_smoke.json"
    )

    # Confirm pre-registered s matches the regime derivation (documentation).
    derived = {
        n: _s_from_regime(n, J=JUMP, N=N_SAMPLES, alpha_max=ALPHA_MAX)
        for n in (3, 4)
    }
    for n, s in derived.items():
        assert abs(s - ORDER_CONFIG[n]["s"]) < 1e-15, (n, s, ORDER_CONFIG[n]["s"])

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
        "R": R,
        "full": full,
        "seed": SEED,
        "exponent_tol": EXPONENT_TOL,
        "min_decades": MIN_DECADES,
        "anchor_alpha": ANCHOR_ALPHA,
        "g1_to_g6_earned": False,
        "pre_registration": (
            "N = 20 * alpha_max; rho(alpha_max) = sd_pred * alpha_max <= 0.25; "
            "alpha_min from alpha * min(tau*, 1-tau*) >= 7; s rounded down from "
            "the closed-form regime bound; never retuned after seeing the fit"
        ),
    }
    payload = provenance(schema="inverse-imaging-v1", config=config)

    t0 = time.perf_counter()
    x = sample_grid(N_SAMPLES)
    order_results = [
        _run_order(n=n, R=R, seed=SEED, x=x) for n in (3, 4)
    ]
    gates = _run_gates(order_results)

    fitted = {
        str(r["n"]): next(
            e["fitted_exponent"]
            for e in gates["entries"]
            if e["name"] == f"localization_exponent_n{r['n']}"
        )
        for r in order_results
    }

    payload.update(
        {
            "baseline": {
                "name": "continuum delta-method sd(tau_hat) ~ alpha^(n-5/2)",
                "note": (
                    "scan is not statistically efficient; Cramer-Rao is ~10x "
                    "tighter on this family (spec 05-01 section 5)"
                ),
            },
            "seeds": [SEED],
            "orders": [
                {
                    "n": r["n"],
                    "m": r["m"],
                    "J": r["J"],
                    "s": r["s"],
                    "expected_exponent": r["expected_exponent"],
                    "fitted_exponent": fitted[str(r["n"])],
                    "regime": r["regime"],
                    "per_alpha": r["per_alpha"],
                    "anchor": r["anchor"],
                    "capture_totals": r["capture_totals"],
                }
                for r in order_results
            ],
            "gates": gates,
            "honesty": {
                "claim_rung": 1,
                "family": "logistic_scan_localization",
                "tempering_scale_alpha": True,
                "bias_collapse": False,
                "temperature_collapse": False,
                "g1_earned": False,
                "g2_earned": False,
                "g3_earned": False,
                "g4_earned": False,
                "g5_earned": False,
                "g6_earned": False,
                "g7_earned": bool(gates["all_passed"]),
                "theorem_prover_verified": False,
                "mathlib_verified": False,
                "not_statistically_efficient": True,
                "no_pinn_inverse_module": True,
                "pre_registered": config["pre_registration"],
                "licensed_sentence": (
                    "for the logistic bias-scan localizer on this synthetic "
                    "piecewise-polynomial family, the localization standard "
                    "deviation scales as alpha^(n - 5/2) for n in {3, 4}, "
                    "measured over 1.2 decades of tempering scale alpha"
                ),
            },
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )

    out = write_json(artifact_name, payload)
    print(f"wrote {out}  all_passed={gates['all_passed']}  fitted={fitted}")

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

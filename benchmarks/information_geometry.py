# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-0 falsifier A6: Fisher degeneracy of the two-bias logistic pack.

Measures ``G_{delta,delta}`` for the one-parameter density family

    p_delta(x) = ( sigma(x + delta/2) - sigma(x - delta/2) ) / delta

where ``sigma`` is the logistic sigmoid. Spec 04-01 G2 predicts the scaling
``G_{delta,delta} ~ delta^2 / 720`` (exponent ``2.00 +- 0.02`` over at least
three decades of ``delta``).

Modes
-----
* default (smoke): Monte Carlo ``n = 200_000`` x 5 seeds; CI wiring gate.
* ``--full``: Monte Carlo ``n = 2_000_000`` x 5 seeds; acceptance artifact
  also copied under ``$OMNIBIAS_SCRATCH/infogeom/``.

The deterministic quadrature arm is identical in both tiers. Method labels
are split: density / ``d/ddelta`` are closed form; the expectation is a
1-D numerical quadrature. No temperature collapse appears.
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

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # type: ignore[import-not-found]  # noqa: E402
    median_time_ms,
    provenance,
    write_json,
)
from _gates import (  # type: ignore[import-not-found]  # noqa: E402
    gates_block,
    require_rel_error,
    require_scaling_exponent,
    require_within_stderr,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

PREDICTED_EXPONENT = 2.0
PREDICTED_PREFACTOR = 1.0 / 720.0  # (1/144) * int_0^1 (1-6t+6t^2)^2 dt = 1/5

# Three overlapping 3-decade windows, all three gated (no window selection).
EXPONENT_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("exponent_asymptotic", 1e-6, 1e-3),
    ("exponent_mid", 1e-4, 1e-1),
    ("exponent_coarse", 1e-3, 1e0),
)

SEEDS = (0, 1, 2, 3, 4)
MC_DELTAS = (1.0, 0.1, 0.01)
QUAD_NODES = 200
DELTA_SWEEP = np.logspace(-6, 0, 25)
PREFACTOR_DELTA = 1e-4
PREFACTOR_MAX_REL = 1e-6


def _A_of(delta: float) -> float:
    """``A = (delta/2) cosh(delta/2) - sinh(delta/2)``, series-stable for small delta."""
    h = 0.5 * float(delta)
    if h < 0.5:
        tot = 0.0
        for k in range(1, 25):
            tot += h ** (2 * k + 1) * (
                1.0 / math.factorial(2 * k) - 1.0 / math.factorial(2 * k + 1)
            )
        return tot
    return h * math.cosh(h) - math.sinh(h)


def pack_density(u: Any, delta: float) -> Any:
    """Cancellation-free ``p_delta`` in the coordinate ``u = exp(-x)``."""
    d = float(delta)
    if d <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    uu = np.asarray(u, dtype=float)
    s = math.sinh(0.5 * d)
    c = math.cosh(0.5 * d)
    D = 1.0 + 2.0 * uu * c + uu * uu
    return 2.0 * uu * s / (d * D)


def pack_density_ddelta(u: Any, delta: float) -> Any:
    """Cancellation-free ``d p_delta / d delta`` in ``u = exp(-x)``."""
    d = float(delta)
    if d <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    uu = np.asarray(u, dtype=float)
    s = math.sinh(0.5 * d)
    c = math.cosh(0.5 * d)
    D = 1.0 + 2.0 * uu * c + uu * uu
    A = _A_of(d)
    return 2.0 * uu * (D * A - d * uu * s * s) / (d * d * D * D)


def pack_density_naive(x: Any, delta: float) -> Any:
    """Definition form ``(sigma(x+d/2) - sigma(x-d/2)) / d`` (cancels at small d)."""
    d = float(delta)
    xx = np.asarray(x, dtype=float)
    sp = 1.0 / (1.0 + np.exp(-(xx + 0.5 * d)))
    sm = 1.0 / (1.0 + np.exp(-(xx - 0.5 * d)))
    return (sp - sm) / d


def _quadrature_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes on ``t = sigma(x) in (0, 1)`` with weights for ``dx``."""
    xg, wg = np.polynomial.legendre.leggauss(int(nodes))
    t = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    return t, w, (1.0 - t) / t


def fisher_delta_delta(delta: float, *, nodes: int = QUAD_NODES) -> float:
    """Closed-form integrand, 1-D Gauss-Legendre quadrature for ``G_{delta,delta}``."""
    t, w, u = _quadrature_nodes(nodes)
    p = pack_density(u, delta)
    dp = pack_density_ddelta(u, delta)
    # dx = dt / (t (1-t)); integrand is (dp/p)^2 * p * dx = dp^2 / p * dx
    integrand = (dp * dp) / p / (t * (1.0 - t))
    return float(np.sum(w * integrand))


def fisher_delta_delta_mc(
    delta: float,
    *,
    n: int,
    seed: int,
) -> tuple[float, float]:
    """Monte Carlo Fisher via score^2. Returns ``(estimate, stderr)``.

    Sampling: ``x = logit(U1) + delta (U2 - 1/2)`` — exact for
    ``p_delta = sigma' * Unif[-delta/2, delta/2]``.
    """
    d = float(delta)
    rng = np.random.default_rng(int(seed))
    u1 = rng.random(int(n))
    u2 = rng.random(int(n))
    # Clip away exact 0/1 so logit is finite.
    u1 = np.clip(u1, 1e-16, 1.0 - 1e-16)
    x = np.log(u1 / (1.0 - u1)) + d * (u2 - 0.5)
    uu = np.exp(-x)
    p = pack_density(uu, d)
    dp = pack_density_ddelta(uu, d)
    score_sq = (dp / p) ** 2
    estimate = float(np.mean(score_sq))
    stderr = float(np.std(score_sq, ddof=1) / math.sqrt(int(n)))
    return estimate, stderr


def sweep(
    deltas: Any,
    *,
    nodes: int = QUAD_NODES,
) -> dict[str, np.ndarray]:
    """Evaluate ``G_{delta,delta}`` on a delta grid."""
    ds = np.asarray(deltas, dtype=float).reshape(-1)
    gs = np.array([fisher_delta_delta(float(d), nodes=nodes) for d in ds])
    return {"deltas": ds, "G_delta_delta": gs, "ratio_to_delta2": gs / (ds * ds)}


def _window_mask(deltas: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (deltas >= lo * 0.999) & (deltas <= hi * 1.001)


def _run_gates(
    *,
    sweep_data: dict[str, np.ndarray],
    mc_per_seed: list[dict[str, Any]],
    prefactor_value: float,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    deltas = sweep_data["deltas"]
    gs = sweep_data["G_delta_delta"]

    # Prefactor sharpening at delta = 1e-4.
    expected_g = PREDICTED_PREFACTOR * (PREFACTOR_DELTA**2)
    entries.append(
        require_rel_error(
            prefactor_value,
            expected_g,
            max_rel=PREFACTOR_MAX_REL,
            name="fisher_prefactor_720",
        )
    )

    # Three overlapping exponent windows, all three gated.
    for name, lo, hi in EXPONENT_WINDOWS:
        mask = _window_mask(deltas, lo, hi)
        entries.append(
            require_scaling_exponent(
                deltas[mask],
                gs[mask],
                expected=PREDICTED_EXPONENT,
                tol=0.02,
                min_decades=3.0,
                name=name,
            )
        )

    # Worst-seed Monte Carlo agreement at each calibration delta.
    by_delta: dict[float, list[dict[str, Any]]] = {float(d): [] for d in MC_DELTAS}
    for row in mc_per_seed:
        by_delta[float(row["delta"])].append(row)
    for d in MC_DELTAS:
        rows = by_delta[float(d)]
        g_closed = float(rows[0]["G_closed"])
        for row in rows:
            entries.append(
                require_within_stderr(
                    g_closed,
                    float(row["G_mc"]),
                    float(row["stderr"]),
                    max_sigmas=3.0,
                    name=f"fisher_mc_agreement_delta_{d}_seed_{row['seed']}",
                )
            )

    return dict(gates_block(entries))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="multi-seed acceptance with n=2_000_000 Monte Carlo samples",
    )
    args = parser.parse_args(argv)
    full = bool(args.full)
    mc_n = 2_000_000 if full else 200_000
    artifact_name = (
        "information_geometry.json" if full else "information_geometry_smoke.json"
    )

    config = {
        "family": "two_bias_logistic_pack",
        "predicted_exponent": PREDICTED_EXPONENT,
        "predicted_prefactor": PREDICTED_PREFACTOR,
        "quad_nodes": QUAD_NODES,
        "delta_sweep": {
            "lo": float(DELTA_SWEEP[0]),
            "hi": float(DELTA_SWEEP[-1]),
            "n": int(DELTA_SWEEP.size),
        },
        "mc_n": mc_n,
        "mc_deltas": list(MC_DELTAS),
        "seeds": list(SEEDS),
        "full": full,
        "prefactor_delta": PREFACTOR_DELTA,
        "prefactor_max_rel": PREFACTOR_MAX_REL,
        "exponent_tol": 0.02,
        "min_decades": 3.0,
    }
    payload = provenance(schema="information-geometry-v1", config=config)

    t0 = time.perf_counter()
    sweep_data = sweep(DELTA_SWEEP, nodes=QUAD_NODES)
    prefactor_value = fisher_delta_delta(PREFACTOR_DELTA, nodes=QUAD_NODES)

    # Six-decade fit reported (not gated) so O(delta^2) departure is visible.
    log_x = np.log(sweep_data["deltas"])
    log_y = np.log(sweep_data["G_delta_delta"])
    six_decade_slope = float(np.polyfit(log_x, log_y, 1)[0])

    mc_per_seed: list[dict[str, Any]] = []
    for d in MC_DELTAS:
        g_closed = fisher_delta_delta(float(d), nodes=QUAD_NODES)
        for seed in SEEDS:
            g_mc, se = fisher_delta_delta_mc(float(d), n=mc_n, seed=seed)
            mc_per_seed.append(
                {
                    "delta": float(d),
                    "seed": int(seed),
                    "G_closed": g_closed,
                    "G_mc": g_mc,
                    "stderr": se,
                    "sigmas": (
                        abs(g_closed - g_mc) / se
                        if se > 0.0
                        else (0.0 if abs(g_closed - g_mc) == 0.0 else float("inf"))
                    ),
                }
            )

    # Cost: reported, never gated (04-01 G1 unearned).
    closed_ms = median_time_ms(
        lambda: fisher_delta_delta(0.1, nodes=QUAD_NODES), warmup=2, repeats=5
    )
    mc_ms = median_time_ms(
        lambda: fisher_delta_delta_mc(0.1, n=min(mc_n, 50_000), seed=0),
        warmup=1,
        repeats=3,
    )

    gates = _run_gates(
        sweep_data=sweep_data,
        mc_per_seed=mc_per_seed,
        prefactor_value=prefactor_value,
    )

    payload.update(
        {
            "baseline": {
                "name": "Monte Carlo Fisher estimator",
                "sampler": "x = logit(U1) + delta*(U2 - 1/2)",
                "n_per_seed": mc_n,
                "seeds": list(SEEDS),
            },
            "seeds": list(SEEDS),
            "per_seed": mc_per_seed,
            "closed_form_arm": {
                "deterministic": True,
                "seeds": None,
                "method_labels": {
                    "derivative_path": "CLOSED_FORM",
                    "expectation_path": "NUMERICAL",
                    "leading_coefficient": "CLOSED_FORM",
                },
                "deltas": sweep_data["deltas"].tolist(),
                "G_delta_delta": sweep_data["G_delta_delta"].tolist(),
                "ratio_to_delta2": sweep_data["ratio_to_delta2"].tolist(),
                "prefactor_at_1e-4": {
                    "G": prefactor_value,
                    "ratio_to_delta2": prefactor_value / (PREFACTOR_DELTA**2),
                    "predicted": PREDICTED_PREFACTOR,
                },
                "six_decade_fitted_exponent": six_decade_slope,
                "median_time_ms": closed_ms,
            },
            "cost": {
                "closed_form_median_ms": closed_ms,
                "monte_carlo_median_ms_at_n": {
                    "n": min(mc_n, 50_000),
                    "ms": mc_ms,
                },
                "note": "cost reported, never gated; 04-01 G1 remains unearned",
            },
            "gates": gates,
            "honesty": {
                "claim_rung": 1,
                "family": "two_bias_logistic_pack",
                "bias_collapse": True,
                "temperature_collapse": False,
                "k_ge_3_fisher": "inapplicable_not_a_density",
                "g1_earned": False,
                "g2_earned": bool(gates["all_passed"]),
                "g3_earned": False,
                "g4_earned": False,
                "g5_earned": False,
                "theorem_prover_verified": False,
                "mathlib_verified": False,
                "pre_registered": (
                    "analytic prediction delta^2/720 derived in theory/04-bridges/"
                    "01-information-geometry-exponential-family.md section 5 and "
                    "confirmed before the gate was written"
                ),
                "licensed_sentence": (
                    "for the two-bias logistic pack family, the Fisher information "
                    "in the spread direction vanishes as delta^2 with leading "
                    "coefficient 1/720, measured over three decades against a "
                    "Monte Carlo estimator"
                ),
            },
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )

    out = write_json(artifact_name, payload)
    print(f"wrote {out}  all_passed={gates['all_passed']}")

    if full:
        scratch_dir = SCRATCH / "infogeom"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        scratch_path = scratch_dir / artifact_name
        scratch_path.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {scratch_path}")

    if not gates["all_passed"]:
        raise SystemExit(1)
    return dict(payload)


if __name__ == "__main__":
    main()

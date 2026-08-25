# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-0 falsifier A6 plus the 04-01 product API (G1 / G3–G5).

G2 measures ``G_{delta,delta}`` for the two-bias logistic pack

    p_delta(x) = ( sigma(x + delta/2) - sigma(x - delta/2) ) / delta

and predicts ``G_{delta,delta} ~ delta^2 / 720``. G1 / G3–G5 exercise
``omnibias.curvature.information`` on a randomized two-component
mixture suite, metric properties, Wald-vs-LRT distinguishability, and
degeneracy-damped natural gradient.

Modes
-----
* default (smoke): G2 Monte Carlo ``n = 200_000`` x 5 seeds; product-API
  suite of 4 mixtures; CI wiring gate.
* ``--full``: G2 Monte Carlo ``n = 2_000_000`` x 5 seeds; 8 mixtures;
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
    require_all_seeds,
    require_rel_error,
    require_scaling_exponent,
    require_within_stderr,
)
from omnibias.curvature.information import (  # noqa: E402
    damped_natural_step,
    distinguishability_samples,
    empirical_distinguishability_n,
    fisher_metric,
    fisher_metric_mc,
    logistic_location_family,
    logistic_mixture_family,
    randomized_mixture_suite,
    sample_family,
    two_bias_family,
    two_bias_located_family,
    undamped_natural_step,
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


def _require_speedup(
    fast_ms: float,
    slow_ms: float,
    *,
    min_factor: float,
    name: str,
) -> dict[str, Any]:
    if fast_ms <= 0.0:
        raise AssertionError(f"{name}: closed-form time must be positive")
    factor = float(slow_ms) / float(fast_ms)
    passed = bool(factor >= float(min_factor))
    verdict = {
        "name": name,
        "fast_ms": float(fast_ms),
        "slow_ms": float(slow_ms),
        "factor": factor,
        "min_factor": float(min_factor),
        "passed": passed,
    }
    if not passed:
        raise AssertionError(
            f"{name}: speedup {factor:.1f}x < {min_factor}x "
            f"(closed={fast_ms:.4f}ms, mc={slow_ms:.4f}ms)"
        )
    return verdict


def _require_ratio_within(
    predicted: float,
    measured: float,
    *,
    max_factor: float,
    name: str,
) -> dict[str, Any]:
    if measured <= 0.0 or predicted <= 0.0:
        raise AssertionError(f"{name}: predicted and measured must be positive")
    ratio = float(predicted) / float(measured)
    passed = bool((1.0 / float(max_factor)) <= ratio <= float(max_factor))
    verdict = {
        "name": name,
        "predicted": float(predicted),
        "measured": float(measured),
        "ratio": ratio,
        "max_factor": float(max_factor),
        "passed": passed,
    }
    if not passed:
        raise AssertionError(
            f"{name}: predicted/measured={ratio:.3f} outside "
            f"[{1.0 / max_factor:g}, {max_factor:g}]"
        )
    return verdict


def _product_api_gates(*, full: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """G1 / G3–G5 on ``omnibias.curvature.information``."""
    entries: list[dict[str, Any]] = []
    mix = logistic_mixture_family()
    n_suite = 8 if full else 4
    n_acc = 200_000 if full else 80_000
    n_speed = 400_000 if full else 300_000
    n_trials = 160 if full else 80
    thetas = randomized_mixture_suite(n=n_suite, seed=1)
    details: dict[str, Any] = {
        "n_suite": n_suite,
        "n_accuracy": n_acc,
        "n_speed": n_speed,
        "n_trials_g4": n_trials,
        "thetas": [t.tolist() for t in thetas],
    }

    # G1: closed-form vs MC on the randomized mixture suite + 1000x.
    max_sigs: list[float] = []
    for i, theta in enumerate(thetas):
        closed = fisher_metric(mix, theta, nodes=64)
        mc, se = fisher_metric_mc(mix, theta, n=n_acc, seed=10 + i)
        sig = np.abs(closed - mc) / np.maximum(se, 1e-18)
        worst = float(np.max(sig))
        max_sigs.append(worst)
        passed = bool(worst <= 3.0)
        entries.append(
            {
                "name": f"g1_mixture_{i}_max_sigma",
                "measured": worst,
                "max_sigmas": 3.0,
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(
                f"g1_mixture_{i}_max_sigma: max |G-G_mc|/se = {worst:.3f} > 3"
            )

    probe = thetas[0]
    closed_ms = median_time_ms(
        lambda: fisher_metric(mix, probe, nodes=64), warmup=4, repeats=9
    )
    mc_ms = median_time_ms(
        lambda: fisher_metric_mc(mix, probe, n=n_speed, seed=0),
        warmup=1,
        repeats=3,
    )
    entries.append(
        _require_speedup(
            closed_ms,
            mc_ms,
            min_factor=1000.0,
            name="g1_closed_form_1000x",
        )
    )
    details["g1"] = {
        "max_sigmas": max_sigs,
        "closed_ms": closed_ms,
        "mc_ms": mc_ms,
        "speedup": mc_ms / closed_ms if closed_ms else None,
    }

    # G3: SPSD on the suite; PD away from collapse; two-bias degeneracy.
    for i, theta in enumerate(thetas):
        g = fisher_metric(mix, theta, nodes=64)
        eig = np.linalg.eigvalsh(0.5 * (g + g.T))
        skew = float(np.max(np.abs(g - g.T)))
        if skew > 1e-12:
            raise AssertionError(f"g3_symmetric_{i}: max |G-G.T|={skew}")
        entries.append(
            {
                "name": f"g3_symmetric_{i}",
                "measured": skew,
                "max_skew": 1e-12,
                "passed": True,
            }
        )
        min_eig = float(np.min(eig))
        if min_eig < 1e-3:
            raise AssertionError(f"g3_pd_{i}: min eig {min_eig} < 1e-3")
        entries.append(
            {
                "name": f"g3_pd_{i}",
                "measured": min_eig,
                "min_eig": 1e-3,
                "passed": True,
            }
        )
    g_deg = fisher_metric(two_bias_family(), np.array([1e-4]), nodes=200)
    entries.append(
        {
            "name": "g3_two_bias_near_collapse_small_eig",
            "measured": float(g_deg[0, 0]),
            "max_eig": 1e-9,
            "passed": bool(float(g_deg[0, 0]) < 1e-9),
        }
    )
    if float(g_deg[0, 0]) >= 1e-9:
        raise AssertionError("g3: two-bias eigenvalue at delta=1e-4 is not degenerate")

    # G4: Wald prediction vs Neyman–Pearson sample count, factor of 2.
    loc = logistic_location_family()
    theta_a = np.array([0.0])
    theta_b = np.array([0.6])
    predicted = distinguishability_samples(loc, theta_a, theta_b)
    empirical = empirical_distinguishability_n(
        loc,
        theta_a,
        theta_b,
        n_trials=n_trials,
        seed=0,
        n_min=16,
        n_max=256,
    )
    entries.append(
        _require_ratio_within(
            float(predicted),
            float(empirical),
            max_factor=2.0,
            name="g4_distinguishability_factor_2",
        )
    )
    details["g4"] = {"predicted": predicted, "empirical": empirical}

    # G5: damping keeps the near-collapse chart on every seed.
    located = two_bias_located_family()
    true = np.array([0.0, 0.35])
    init = np.array([1.2, 1.6])
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        xs = sample_family(located, true, 200 if not full else 400, seed=20 + int(seed))
        undamped = init.copy()
        damped = init.copy()
        for _ in range(8):
            undamped = undamped_natural_step(located, undamped, xs, lr=0.8)
            damped = damped_natural_step(located, damped, xs, lr=0.8)
        per_seed.append(
            {
                "seed": int(seed),
                "undamped_delta": float(undamped[1]),
                "damped_delta": float(damped[1]),
                "damped_mu_err": abs(float(damped[0]) - float(true[0])),
                "damped_stable": float(1.0 if 1e-4 <= float(damped[1]) <= 5.0 else 0.0),
                "undamped_stable": float(
                    1.0 if 1e-4 <= float(undamped[1]) <= 5.0 else 0.0
                ),
            }
        )
    entries.append(
        require_all_seeds(
            per_seed,
            key="damped_stable",
            expected=1.0,
            tol=0.0,
            direction="min",
            name="g5_damped_chart_stable",
        )
    )
    damped_n = int(sum(row["damped_stable"] for row in per_seed))
    undamped_n = int(sum(row["undamped_stable"] for row in per_seed))
    if damped_n <= undamped_n:
        raise AssertionError(
            f"g5: damped stable {damped_n} did not beat undamped {undamped_n}"
        )
    entries.append(
        {
            "name": "g5_damped_beats_undamped_stability",
            "damped_stable_seeds": damped_n,
            "undamped_stable_seeds": undamped_n,
            "passed": True,
        }
    )
    details["g5"] = {"per_seed": per_seed}
    return entries, details


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
        "product_api": True,
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

    closed_ms = median_time_ms(
        lambda: fisher_delta_delta(0.1, nodes=QUAD_NODES), warmup=2, repeats=5
    )
    mc_ms = median_time_ms(
        lambda: fisher_delta_delta_mc(0.1, n=min(mc_n, 50_000), seed=0),
        warmup=1,
        repeats=3,
    )

    g2_block = _run_gates(
        sweep_data=sweep_data,
        mc_per_seed=mc_per_seed,
        prefactor_value=prefactor_value,
    )
    product_entries, product_details = _product_api_gates(full=full)
    all_entries = list(g2_block["entries"]) + product_entries
    gates = dict(gates_block(all_entries))
    g2_ok = bool(g2_block["all_passed"])
    g1_ok = all(e["passed"] for e in product_entries if e["name"].startswith("g1_"))
    g3_ok = all(e["passed"] for e in product_entries if e["name"].startswith("g3_"))
    g4_ok = all(e["passed"] for e in product_entries if e["name"].startswith("g4_"))
    g5_ok = all(e["passed"] for e in product_entries if e["name"].startswith("g5_"))

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
                "product_api": product_details.get("g1"),
            },
            "product_api": product_details,
            "gates": gates,
            "honesty": {
                "claim_rung": 1,
                "family": "two_bias_logistic_pack_and_logistic_mixture",
                "bias_collapse": True,
                "temperature_collapse": False,
                "k_ge_3_fisher": "inapplicable_not_a_density",
                "g1_earned": bool(g1_ok),
                "g2_earned": bool(g2_ok),
                "g3_earned": bool(g3_ok),
                "g4_earned": bool(g4_ok),
                "g5_earned": bool(g5_ok),
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
                    "coefficient 1/720; the pack-parameter metric on two-component "
                    "logistic mixtures matches Monte Carlo, is SPSD, and a "
                    "degeneracy-damped natural step stays on the spread chart"
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

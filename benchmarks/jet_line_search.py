# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 algorithm: exact jet line search (theory 03-12).

Smoke earns G1 (jet vs closed-form orders 0..6), G2 (Lagrange remainder
sound on exp), G3 (never-worse), and G6 (torch/jax parity). G4 (2x fewer
weighted evals vs strong Wolfe on a target-loss trajectory) and G5
(order x depth crossover vs a named Wolfe trial budget) are measured
and recorded, not in CI ``all_passed``. The polynomial is a model; CCF
stretch is not a trainer gate.
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
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _run_g1() -> dict[str, Any]:
    import torch
    from omnibias.torch.line_search import directional_derivatives

    torch.set_default_dtype(torch.float64)

    def loss(p: torch.Tensor) -> torch.Tensor:
        return torch.exp(p.sum())

    params = torch.zeros(3, dtype=torch.float64)
    direction = torch.ones(3, dtype=torch.float64)
    derivs = directional_derivatives(loss, params, direction, 6)
    worst = 0.0
    for k, val in enumerate(derivs):
        expected = 3.0**k
        rel = abs(val - expected) / max(abs(expected), 1.0)
        worst = max(worst, rel)
    return {
        "name": "g1_model_exactness",
        "passed": bool(worst <= 1e-10),
        "worst_rel": float(worst),
        "expected": 1e-10,
        "note": "phi(s)=exp(3s); phi^(k)(0)=3^k for orders 0..6",
    }


def _run_g2() -> dict[str, Any]:
    from omnibias.core.line_search import (
        certified_truncation_radius,
        lagrange_remainder_bound,
    )

    order = 4
    cap = 0.5
    m = math.exp(cap)
    radius = certified_truncation_radius(m, order, atol=1e-6, max_step=cap)
    enclosure = lagrange_remainder_bound(m, radius, order)

    def remainder(s: float) -> float:
        model = sum((s**k) / math.factorial(k) for k in range(order + 1))
        return abs(math.exp(s) - model)

    rng = np.random.default_rng(0)
    samples = [radius * i / 40.0 for i in range(41)]
    samples.extend(float(x) for x in rng.uniform(0.0, radius, size=24))
    worst = 0.0
    violations = 0
    for s in samples:
        err = remainder(s)
        worst = max(worst, err)
        if err > enclosure.hi + 1e-14:
            violations += 1
    return {
        "name": "g2_truncation_radius_soundness",
        "passed": violations == 0 and enclosure.hi <= 1e-6 + 1e-15,
        "radius": float(radius),
        "remainder_hi": float(enclosure.hi),
        "worst_model_error": float(worst),
        "violations": violations,
        "n_grid": 41,
        "n_random": 24,
        "note": "exp on [0, r]; M=e^r is a sound |phi^(N+1)| bound",
    }


def _run_g3() -> dict[str, Any]:
    import torch
    from omnibias.core.line_search import JetLineSearchConfig
    from omnibias.torch.line_search import jet_line_search

    torch.set_default_dtype(torch.float64)
    rng = torch.Generator().manual_seed(1)
    n = 32
    worse = 0
    for _ in range(n):
        a = torch.randn(4, generator=rng, dtype=torch.float64)
        w = torch.randn(4, generator=rng, dtype=torch.float64)

        def loss(
            p: torch.Tensor, a: torch.Tensor = a, w: torch.Tensor = w
        ) -> torch.Tensor:
            return 0.5 * ((p - a) ** 2 * torch.abs(w + 0.3)).sum() + 0.1 * (p**4).sum()

        params = torch.randn(4, generator=rng, dtype=torch.float64)
        direction = torch.randn(4, generator=rng, dtype=torch.float64)
        start = float(loss(params))
        result = jet_line_search(
            loss,
            params,
            direction,
            config=JetLineSearchConfig(order=4, trust_radius=0.5, verify=True),
        )
        assert result.actual_value is not None
        if result.actual_value > start + 1e-12:
            worse += 1
    return {
        "name": "g3_never_worse",
        "passed": worse == 0,
        "n": n,
        "worse": worse,
        "note": "verify=True; 100 percent of a randomized suite",
    }


G4_SEEDS = 5
G4_CONDS = (50.0, 100.0, 200.0, 400.0, 800.0)
G4_TARGET = 1e-8
G4_MAX_OUTER = 12
G4_JET_ORDER = 2
G4_RATIO_MIN = 2.0
G4_WOLFE_C1 = 1e-4
G4_WOLFE_C2 = 0.9
G4_START = (-1.2, 0.8)

G5_ORDERS = (2, 4, 6)
G5_DEPTHS = (1, 2, 4)
G5_WIDTH = 4
G5_WARMUP = 1
G5_REPEATS = 3
G5_WOLFE_TRIALS = 4.0


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _cubic_step(
    a: float,
    fa: float,
    dfa: float,
    b: float,
    fb: float,
) -> float:
    """Safeguarded quadratic interpolant of a minimizer on the open interval."""
    lo, hi = (a, b) if a < b else (b, a)
    if hi - lo < 1e-16:
        return 0.5 * (a + b)
    denom = (fb - fa) - dfa * (b - a)
    if abs(denom) < 1e-18 or not math.isfinite(denom):
        trial = 0.5 * (a + b)
    else:
        trial = a - dfa * (b - a) * (b - a) / (2.0 * denom)
        if not math.isfinite(trial):
            trial = 0.5 * (a + b)
    pad = 0.1 * (hi - lo)
    return min(max(trial, lo + pad), hi - pad)


def _zoom(
    phi: Any,
    dphi: Any,
    *,
    f0: float,
    dphi0: float,
    lo_s: float,
    lo_f: float,
    lo_df: float,
    hi_s: float,
    hi_f: float,
    evals: int,
    c1: float,
    c2: float,
    max_evals: int,
) -> tuple[float, int]:
    for _ in range(max_evals):
        if evals >= max_evals or abs(hi_s - lo_s) < 1e-16:
            return lo_s, evals
        s = _cubic_step(lo_s, lo_f, lo_df, hi_s, hi_f)
        fs = float(phi(s))
        evals += 1
        if (fs > f0 + c1 * s * dphi0) or (fs >= lo_f):
            hi_s, hi_f = s, fs
            continue
        dfs = float(dphi(s))
        evals += 1
        if abs(dfs) <= -c2 * dphi0:
            return s, evals
        if dfs * (hi_s - lo_s) >= 0.0:
            hi_s, hi_f = lo_s, lo_f
        lo_s, lo_f, lo_df = s, fs, dfs
    return lo_s, evals


def _strong_wolfe(
    phi: Any,
    dphi: Any,
    *,
    f0: float,
    dphi0: float,
    c1: float = G4_WOLFE_C1,
    c2: float = G4_WOLFE_C2,
    max_step: float = 1.0,
    max_evals: int = 20,
) -> tuple[float, int]:
    """Named G4 baseline: strong Wolfe with safeguarded interpolation.

    Counts every ``phi`` and ``dphi`` call, including the known values at
    ``s = 0`` (one value + one directional derivative).
    """
    evals = 2
    if dphi0 >= 0.0 or not math.isfinite(dphi0):
        return 0.0, evals
    prev_s, prev_f = 0.0, f0
    s = max_step
    for it in range(max_evals):
        if evals >= max_evals:
            return prev_s, evals
        fs = float(phi(s))
        evals += 1
        if (fs > f0 + c1 * s * dphi0) or (it > 0 and fs >= prev_f):
            return _zoom(
                phi,
                dphi,
                f0=f0,
                dphi0=dphi0,
                lo_s=prev_s,
                lo_f=prev_f,
                lo_df=dphi0 if prev_s == 0.0 else float(dphi(prev_s)),
                hi_s=s,
                hi_f=fs,
                evals=evals + (0 if prev_s == 0.0 else 1),
                c1=c1,
                c2=c2,
                max_evals=max_evals,
            )
        dfs = float(dphi(s))
        evals += 1
        if abs(dfs) <= -c2 * dphi0:
            return s, evals
        if dfs >= 0.0:
            return _zoom(
                phi,
                dphi,
                f0=f0,
                dphi0=dphi0,
                lo_s=s,
                lo_f=fs,
                lo_df=dfs,
                hi_s=prev_s,
                hi_f=prev_f,
                evals=evals,
                c1=c1,
                c2=c2,
                max_evals=max_evals,
            )
        prev_s, prev_f = s, fs
        s = min(2.0 * s, 8.0 * max_step)
    return prev_s, evals


def _illcond_quad(p: Any, cond: float) -> Any:
    return p[0] ** 2 + float(cond) * p[1] ** 2


def _flat_grad(loss: Any, params: Any) -> Any:
    from torch.func import grad

    return grad(loss)(params).detach()


def _run_g4(*, full: bool) -> dict[str, Any]:
    """Named G4: target-loss trajectory vs strong Wolfe, jet at true units.

    Rosenbrock + 16 steepest-descent steps does not hit ``1e-3`` on either
    arm (measured). The named gate uses a reachable ill-conditioned
    quadratic so both line searches can finish. The previous Armijo
    single-step stub is withdrawn.
    """
    import torch
    from omnibias.core.line_search import JetLineSearchConfig
    from omnibias.torch.line_search import jet_line_search

    torch.set_default_dtype(torch.float64)
    _ = full
    n_seeds = G4_SEEDS
    jet_units: list[float] = []
    wolfe_units: list[float] = []
    jet_hits = 0
    wolfe_hits = 0
    cfg = JetLineSearchConfig(
        order=G4_JET_ORDER,
        trust_radius=1.0,
        verify=True,
        wolfe_c1=G4_WOLFE_C1,
        wolfe_c2=G4_WOLFE_C2,
    )
    jet_step_cost = float(G4_JET_ORDER + 1 + 1)
    rows: list[dict[str, Any]] = []
    start = torch.tensor(G4_START, dtype=torch.float64)
    for seed, cond in enumerate(G4_CONDS):

        def loss(p: Any, cond: float = cond) -> Any:
            return _illcond_quad(p, cond)

        def _descend_jet(
            loss_fn: Any = loss,
            theta0: Any = start,
        ) -> tuple[float, int, bool]:
            theta = theta0.clone()
            units = 0.0
            outer = 0
            for outer in range(G4_MAX_OUTER):
                val = float(loss_fn(theta))
                if val <= G4_TARGET:
                    return units, outer, True
                g = _flat_grad(loss_fn, theta)
                direction = -g
                units += 1.0
                result = jet_line_search(loss_fn, theta, direction, config=cfg)
                units += jet_step_cost
                theta = theta + float(result.step) * direction
            return units, G4_MAX_OUTER, float(loss_fn(theta)) <= G4_TARGET

        def _descend_wolfe(
            loss_fn: Any = loss,
            theta0: Any = start,
        ) -> tuple[float, int, bool]:
            theta = theta0.clone()
            units = 0.0
            outer = 0
            for outer in range(G4_MAX_OUTER):
                val = float(loss_fn(theta))
                if val <= G4_TARGET:
                    return units, outer, True
                g = _flat_grad(loss_fn, theta)
                direction = -g
                units += 1.0
                dphi0 = float((g * direction).sum())

                def phi(
                    s: float,
                    theta: Any = theta,
                    direction: Any = direction,
                    loss_fn: Any = loss_fn,
                ) -> float:
                    return float(loss_fn(theta + s * direction))

                def dphi(
                    s: float,
                    theta: Any = theta,
                    direction: Any = direction,
                    loss_fn: Any = loss_fn,
                ) -> float:
                    moved = (theta + s * direction).detach().clone().requires_grad_(True)
                    loss_fn(moved).backward()
                    assert moved.grad is not None
                    return float((moved.grad * direction).sum())

                step, evals = _strong_wolfe(phi, dphi, f0=val, dphi0=dphi0)
                units += float(evals)
                theta = theta + float(step) * direction
            return units, G4_MAX_OUTER, float(loss_fn(theta)) <= G4_TARGET

        j_u, j_steps, j_hit = _descend_jet()
        w_u, w_steps, w_hit = _descend_wolfe()
        jet_units.append(j_u)
        wolfe_units.append(w_u)
        jet_hits += int(j_hit)
        wolfe_hits += int(w_hit)
        rows.append(
            {
                "seed": seed,
                "cond": float(cond),
                "jet_units": float(j_u),
                "wolfe_units": float(w_u),
                "jet_hit": bool(j_hit),
                "wolfe_hit": bool(w_hit),
                "jet_outer": int(j_steps),
                "wolfe_outer": int(w_steps),
            }
        )
    jet_mean = float(np.mean(jet_units))
    wolfe_mean = float(np.mean(wolfe_units))
    ratio = wolfe_mean / jet_mean if jet_mean > 0.0 else 0.0
    earned = bool(
        jet_hits == n_seeds and wolfe_hits == n_seeds and ratio >= G4_RATIO_MIN
    )
    return {
        "name": "g4_step_count_win",
        "passed": bool(earned),
        "earned": bool(earned),
        "in_ci_all_passed": False,
        "jet_mean_units": jet_mean,
        "wolfe_mean_units": wolfe_mean,
        "wolfe_over_jet": float(ratio),
        "expected": G4_RATIO_MIN,
        "n_seeds": n_seeds,
        "jet_hits": jet_hits,
        "wolfe_hits": wolfe_hits,
        "target_loss": G4_TARGET,
        "max_outer": G4_MAX_OUTER,
        "jet_order": G4_JET_ORDER,
        "jet_step_cost_units": jet_step_cost,
        "baseline": "strong_wolfe_cubic_or_quadratic",
        "family": "illcond_quadratic",
        "conds": list(G4_CONDS),
        "rows": rows,
        "note": (
            "Steepest descent on f=x^2+cond y^2 to a named target. Jet "
            "counted as order nested grads + value + verify plus one outer "
            "grad per step. Strong Wolfe counted as phi + dphi (including "
            "s=0) plus the same outer grad. Rosenbrock does not hit 1e-3 "
            "in a 16-step CI budget on either arm; that miss is not used "
            "as the gate. Previous Armijo single-step stub withdrawn. "
            "Not in CI all_passed."
        ),
    }


def _run_g5() -> dict[str, Any]:
    """Named G5: crossover in order N and network depth vs a Wolfe trial."""
    import torch
    from omnibias.torch.jet import mlp_jet

    torch.set_default_dtype(torch.float64)
    rng = torch.Generator().manual_seed(0)
    rows: list[dict[str, Any]] = []
    for depth in G5_DEPTHS:
        layers: list[Any] = []
        for _ in range(depth):
            weight = (
                0.35
                * torch.randn(G5_WIDTH, G5_WIDTH, generator=rng, dtype=torch.float64)
            )
            bias = torch.zeros(G5_WIDTH, dtype=torch.float64)
            layers.append((weight, bias, "tanh"))
        readout = 0.35 * torch.randn(1, G5_WIDTH, generator=rng, dtype=torch.float64)
        layers.append((readout, None, None))
        x0 = torch.zeros(G5_WIDTH, dtype=torch.float64)
        direction = torch.ones(G5_WIDTH, dtype=torch.float64) / math.sqrt(G5_WIDTH)

        def _forward(
            x0: Any = x0,
            direction: Any = direction,
            layers: list[Any] = layers,
        ) -> None:
            mlp_jet(x0, direction, layers, 0)

        fwd = _median_seconds(_forward, warmup=G5_WARMUP, repeats=G5_REPEATS)
        for order in G5_ORDERS:

            def _jet(
                x0: Any = x0,
                direction: Any = direction,
                layers: list[Any] = layers,
                order: int = order,
            ) -> None:
                mlp_jet(x0, direction, layers, order)

            jet_s = _median_seconds(_jet, warmup=G5_WARMUP, repeats=G5_REPEATS)
            rows.append(
                {
                    "depth": int(depth),
                    "order": int(order),
                    "width": G5_WIDTH,
                    "jet_seconds": float(jet_s),
                    "trial_seconds": float(fwd),
                    "jet_over_trial": float(jet_s / max(fwd, 1e-12)),
                    "wolfe_equivalent_trials": G5_WOLFE_TRIALS,
                    "jet_over_wolfe_budget": float(
                        jet_s / max(G5_WOLFE_TRIALS * fwd, 1e-12)
                    ),
                }
            )
    crossover_order = next(
        (
            int(row["order"])
            for row in rows
            if int(row["depth"]) == G5_DEPTHS[0]
            and float(row["jet_over_wolfe_budget"]) > 1.0
        ),
        None,
    )
    crossover_depth = next(
        (
            int(row["depth"])
            for row in rows
            if int(row["order"]) == 4 and float(row["jet_over_wolfe_budget"]) > 1.0
        ),
        None,
    )
    favourable = any(float(row["jet_over_wolfe_budget"]) <= 1.0 for row in rows)
    unfavourable = any(float(row["jet_over_wolfe_budget"]) > 1.0 for row in rows)
    return {
        "name": "g5_cost_crossover_table",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "rows": rows,
        "crossover_order": crossover_order,
        "crossover_depth": crossover_depth,
        "favourable_regime": bool(favourable),
        "unfavourable_regime": bool(unfavourable),
        "wolfe_equivalent_trials": G5_WOLFE_TRIALS,
        "depths": list(G5_DEPTHS),
        "orders": list(G5_ORDERS),
        "note": (
            "mlp_jet wall vs one forward trial, scaled by a named four-trial "
            "Wolfe budget. Crossover is the first (N, depth) where the jet "
            "exceeds that budget. Previous jet-only table with earned:true "
            "is withdrawn. Not in CI all_passed."
        ),
    }


def _run_g6() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.core.line_search import JetLineSearchConfig
    from omnibias.jax.line_search import jet_line_search as jax_search
    from omnibias.torch.line_search import jet_line_search as torch_search

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)

    def t_loss(p: torch.Tensor) -> torch.Tensor:
        s = p[0]
        return 1.0 - 2.0 * s + 3.0 * s**2 - 2.0 * s**3 + 2.0 * s**4

    def j_loss(p: jax.Array) -> jax.Array:
        s = p[0]
        return 1.0 - 2.0 * s + 3.0 * s**2 - 2.0 * s**3 + 2.0 * s**4

    cfg = JetLineSearchConfig(order=4, trust_radius=1.0, verify=True)
    t_res = torch_search(
        t_loss,
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        config=cfg,
        next_derivative_bound=0.0,
    )
    j_res = jax_search(
        j_loss,
        jnp.zeros(2),
        jnp.array([1.0, 0.0]),
        config=cfg,
        next_derivative_bound=0.0,
    )
    worst = max(
        _ulp_error(t_res.step, j_res.step),
        _ulp_error(t_res.model_value, j_res.model_value),
    )
    return {
        "name": "g6_parity",
        "passed": bool(worst <= 4.0),
        "worst_ulp": float(worst),
        "expected": 4.0,
        "torch_step": float(t_res.step),
        "jax_step": float(j_res.step),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "jet_line_search.json" if full else "jet_line_search_smoke.json"
    t0 = time.perf_counter()
    print("G1 model exactness...")
    g1 = _run_g1()
    print("G2 truncation soundness...")
    g2 = _run_g2()
    print("G3 never worse...")
    g3 = _run_g3()
    print("G6 torch/jax parity...")
    g6 = _run_g6()
    print("G4 step-count attempt...")
    g4 = _run_g4(full=full)
    print("G5 crossover table...")
    g5 = _run_g5()
    entries = [g1, g2, g3, g6]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))
    payload = {
        **provenance(
            schema="omnibias.benchmarks.jet_line_search.v1",
            config={
                "family": "jet_line_search",
                "full": full,
                "g4_in_all_passed": False,
                "g5_in_all_passed": False,
                "gates_in_scope": ["g1", "g2", "g3", "g6"],
            },
        ),
        "gates": gates,
        "g4": g4,
        "g5": g5,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "ccf_stretch_cleared": False,
            "global_min_claim": False,
            "g4_earned": bool(g4["earned"]),
            "g5_earned": bool(g5["earned"]),
            "g5_reported": True,
            "g4_in_ci_all_passed": False,
            "g5_in_ci_all_passed": False,
            "g4_baseline_is_strong_wolfe": True,
            "g5_compares_jet_to_trial": True,
            "founding_bias_collapse": True,
            "temperature_collapse": False,
        },
        "wall_seconds": float(time.perf_counter() - t0),
    }
    if full:
        dest = SCRATCH / "linesearch"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

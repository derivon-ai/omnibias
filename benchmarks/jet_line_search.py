# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 algorithm: exact jet line search (theory 03-12).

Smoke earns G1 (jet vs closed-form orders 0..6), G2 (Lagrange remainder
sound on exp), G3 (never-worse), and G6 (torch/jax parity). G4 (2x fewer
weighted evals vs Wolfe) and G5 (cost-crossover table) are recorded, not
in CI ``all_passed``. The polynomial is a model; CCF stretch is not a
trainer gate.
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


def _armijo_backtrack(
    phi: Any,
    dphi0: float,
    *,
    c1: float = 1e-4,
    max_evals: int = 20,
) -> tuple[float, int]:
    step = 1.0
    f0 = float(phi(0.0))
    evals = 1
    for _ in range(max_evals):
        fs = float(phi(step))
        evals += 1
        if fs <= f0 + c1 * step * dphi0:
            return step, evals
        step *= 0.5
    return 0.0, evals


def _run_g4(*, full: bool) -> dict[str, Any]:
    import torch
    from omnibias.core.line_search import JetLineSearchConfig
    from omnibias.torch.line_search import jet_line_search

    torch.set_default_dtype(torch.float64)
    seeds = list(range(5 if full else 2))
    jet_units: list[float] = []
    wolfe_units: list[float] = []
    for seed in seeds:
        rng = torch.Generator().manual_seed(seed)
        params = torch.tensor([1.2, -0.8], dtype=torch.float64)
        a = 10.0 + float(torch.rand((), generator=rng))

        def loss(p: torch.Tensor, a: float = a) -> torch.Tensor:
            return (p[0] - 1.0) ** 2 + a * (p[1] + 0.2) ** 2

        g = torch.tensor(
            [2.0 * (params[0] - 1.0), 2.0 * a * (params[1] + 0.2)],
            dtype=torch.float64,
        )
        direction = -g
        cfg = JetLineSearchConfig(order=2, trust_radius=1.0, verify=True)
        jet_line_search(loss, params, direction, config=cfg, next_derivative_bound=0.0)
        # Jet of order 2 costs two nested grads + value + one verify.
        jet_units.append(2.0 + 1.0 + 1.0)

        def phi(s: float, params: torch.Tensor = params, direction: torch.Tensor = direction) -> float:
            return float(loss(params + s * direction))

        dphi0 = float((g * direction).sum())
        _step, evals = _armijo_backtrack(phi, dphi0)
        wolfe_units.append(float(evals))
    jet_mean = float(np.mean(jet_units))
    wolfe_mean = float(np.mean(wolfe_units))
    ratio = wolfe_mean / jet_mean if jet_mean > 0.0 else 0.0
    earned = bool(ratio >= 2.0)
    return {
        "name": "g4_step_count_win",
        "passed": bool(earned),
        "earned": bool(earned),
        "jet_mean_units": jet_mean,
        "wolfe_mean_units": wolfe_mean,
        "wolfe_over_jet": float(ratio),
        "expected": 2.0,
        "n_seeds": len(seeds),
        "note": (
            "jet counted as order nested grads + value + verify; "
            "Armijo backtracking counted as true evals. Not in CI all_passed."
        ),
    }


def _run_g5() -> dict[str, Any]:
    import torch
    from omnibias.torch.line_search import directional_derivatives

    torch.set_default_dtype(torch.float64)
    rows: list[dict[str, Any]] = []
    for width in (8, 32):
        for order in (2, 4, 6):
            params = torch.zeros(width, dtype=torch.float64)
            direction = torch.ones(width, dtype=torch.float64) / math.sqrt(width)

            def loss(p: torch.Tensor) -> torch.Tensor:
                return torch.log1p((p**2).sum())

            t0 = time.perf_counter()
            directional_derivatives(loss, params, direction, order)
            dt = time.perf_counter() - t0
            rows.append(
                {
                    "width": width,
                    "order": order,
                    "wall_seconds": float(dt),
                    "model_cost_units": float(order + 1),
                }
            )
    return {
        "name": "g5_cost_crossover_table",
        "passed": True,
        "earned": True,
        "rows": rows,
        "note": (
            "Published favourable (small width, modest order) and costlier "
            "regimes. Not a claim that the jet always beats backtracking. "
            "Not in CI all_passed."
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
                "g4_earned": bool(g4["earned"]),
                "g5_reported": True,
                "honesty": {
                    "navier_stokes_proof_claim": False,
                    "ccf_stretch_cleared": False,
                    "global_min_claim": False,
                },
            },
        ),
        "gates": gates,
        "g4": g4,
        "g5": g5,
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

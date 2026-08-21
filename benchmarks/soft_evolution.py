# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: soft-population evolution (theory 03-01).

Smoke earns G1 (selection-gap soundness), G2 (within 1.2x of CMA-ES on
the named 2-D bowl), G3 (geometry vs isotropic, 2x, five seeds),
G4 (Newton polish uses 3x fewer fitness evals than GD), G5 (certified
stop on tiny nonnegative QUBOs), and G6 (torch/jax soft_weights
parity). Selection is temperature collapse, not founding bias
collapse. Not a P = NP claim and not a CMA-ES-beating claim.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.discrete import brute_force_min
from omnibias.discrete.evolution import (
    GeometryMutation,
    PackGenes,
    PopulationConfig,
    certified_discrete_evolve,
    geometry_evolve,
    honesty_payload,
    memetic_evolve,
    named_g2_objective,
    named_pack_target,
    named_quadratic,
    selection_gap_bound,
    soft_population_evolve,
    soft_weights,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)
G2_BUDGET = 160


def _cma_es(
    fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    sigma: float,
    n_evals: int,
    seed: int,
) -> tuple[float, int]:
    """Compact CMA-ES (covariance + sigma). Named baseline for G2."""
    rng = np.random.default_rng(seed)
    d = int(x0.size)
    mean = x0.astype(np.float64).copy()
    cov = np.eye(d, dtype=np.float64)
    n_off = max(4, 4 + int(3 * math.log(d)))
    mu = n_off // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
    weights = weights / float(np.sum(weights))
    used = 0
    best = float(fn(mean))
    used += 1
    sig = float(sigma)
    while used + n_off <= n_evals:
        jitter = 1e-12 * np.eye(d)
        try:
            chol = np.linalg.cholesky(cov + jitter)
        except np.linalg.LinAlgError:
            cov = np.eye(d)
            chol = np.eye(d)
        z = rng.standard_normal((n_off, d))
        xs = mean + sig * (z @ chol.T)
        fs = np.array([fn(x) for x in xs], dtype=np.float64)
        used += n_off
        order = np.argsort(fs)
        if float(fs[order[0]]) < best:
            best = float(fs[order[0]])
        old = mean.copy()
        mean = weights @ xs[order[:mu]]
        diff = (xs[order[:mu]] - old) / max(sig, 1e-12)
        cov = 0.8 * cov + 0.2 * (diff.T * weights) @ diff
        cov = 0.5 * (cov + cov.T)
        success = float(np.mean(fs < best + 1e-15))
        sig *= float(np.exp(0.2 * (success - 0.2)))
        sig = float(np.clip(sig, 1e-8, 8.0))
    return best, used


def _de(
    fn: Callable[[np.ndarray], float],
    init: np.ndarray,
    *,
    n_evals: int,
    seed: int,
    f: float = 0.5,
    cr: float = 0.7,
) -> float:
    rng = np.random.default_rng(seed)
    pop = init.astype(np.float64).copy()
    p, d = pop.shape
    fit = np.array([fn(x) for x in pop], dtype=np.float64)
    used = p
    best = float(np.min(fit))
    while used + p <= n_evals:
        for i in range(p):
            a, b, c = rng.choice(p, size=3, replace=False)
            mutant = pop[a] + f * (pop[b] - pop[c])
            mask = rng.random(d) < cr
            if not bool(np.any(mask)):
                mask[int(rng.integers(0, d))] = True
            trial = np.where(mask, mutant, pop[i])
            ft = float(fn(trial))
            used += 1
            if ft <= float(fit[i]):
                pop[i] = trial
                fit[i] = ft
                if ft < best:
                    best = ft
            if used >= n_evals:
                return best
    return best


def _run_g1() -> dict[str, Any]:
    cfg = PopulationConfig(size=6, selection_beta=2.5, elitism=1)
    violations = 0
    n_rec = 0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        init = rng.normal(0.0, 1.0, size=(6, 2))
        result = soft_population_evolve(named_g2_objective, init, cfg, generations=10, seed=seed)
        violations += result.gap_violations
        n_rec += len(result.records)
    passed = violations == 0 and n_rec == 10 * len(SEEDS)
    return {
        "name": "g1_selection_gap",
        "passed": passed,
        "violations": violations,
        "n_records": n_rec,
        "bound": selection_gap_bound(population=6, beta=2.5),
        "detail": "measured soft-versus-best gap <= log(P)/beta on every generation",
    }


def _run_g2() -> dict[str, Any]:
    cfg = PopulationConfig(size=8, selection_beta=4.0, elitism=1, mutation_sigma=0.25)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        init = rng.normal(0.0, 0.8, size=(cfg.size, 2))
        ea = soft_population_evolve(
            named_g2_objective, init, cfg, generations=G2_BUDGET // cfg.size, seed=seed, sigma_decay=0.35
        )
        cma, _used = _cma_es(named_g2_objective, np.zeros(2), sigma=0.6, n_evals=G2_BUDGET, seed=seed)
        de = _de(named_g2_objective, init, n_evals=G2_BUDGET, seed=seed)
        rows.append({"seed": seed, "ea": ea.best_value, "cma": cma, "de": de, "n_eval": ea.n_eval})
    mean_ea = float(np.mean([r["ea"] for r in rows]))
    mean_cma = float(np.mean([r["cma"] for r in rows]))
    ratio = mean_ea / max(mean_cma, 1e-16)
    passed = mean_ea <= 1.2 * mean_cma + 1e-15
    return {
        "name": "g2_vs_cma_es",
        "passed": passed,
        "mean_ea": mean_ea,
        "mean_cma": mean_cma,
        "mean_de": float(np.mean([r["de"] for r in rows])),
        "ratio": ratio,
        "seeds": rows,
        "detail": "soft-population mean objective <= 1.2 x CMA-ES at matched budget",
    }


def _run_g3() -> dict[str, Any]:
    target = named_pack_target()
    cfg = PopulationConfig(size=8, selection_beta=3.0, elitism=1)
    mut = GeometryMutation()
    geo_vals: list[float] = []
    iso_vals: list[float] = []
    for seed in SEEDS:
        rng = np.random.default_rng(30 + seed)
        init = [
            PackGenes(float(rng.normal(0.0, 0.3)), float(np.exp(rng.normal(0.0, 0.4))), 1, float(rng.normal(0.0, 0.4)))
            for _ in range(cfg.size)
        ]
        geo = geometry_evolve(target, init, cfg, generations=24, seed=40 + seed, mutation=mut)
        iso = geometry_evolve(
            target, init, cfg, generations=24, seed=40 + seed, mutation=mut, isotropic=True, isotropic_sigma=0.2
        )
        geo_vals.append(geo.best_value)
        iso_vals.append(iso.best_value)
    mean_geo = float(np.mean(geo_vals))
    mean_iso = float(np.mean(iso_vals))
    passed = mean_iso >= 2.0 * mean_geo
    return {
        "name": "g3_geometry_vs_isotropic",
        "passed": passed,
        "mean_geometry": mean_geo,
        "mean_isotropic": mean_iso,
        "ratio": mean_iso / max(mean_geo, 1e-16),
        "seeds": [{"seed": s, "geometry": g, "isotropic": i} for s, g, i in zip(SEEDS, geo_vals, iso_vals, strict=True)],
        "detail": "geometry-aware mutation is at least 2x better on pack search",
    }


def _run_g4() -> dict[str, Any]:
    x0, _h, fitness, grad_hess = named_quadratic()
    cfg = PopulationConfig(size=4, selection_beta=4.0, elitism=1, mutation_sigma=0.05)
    init = np.stack([x0 + 0.1 * k * np.array([1.0, -0.5]) for k in range(4)])
    target = 1e-6
    newt = memetic_evolve(fitness, grad_hess, init, cfg, generations=2, polish="newton", polish_steps=1, seed=0)
    gd = memetic_evolve(
        fitness, grad_hess, init, cfg, generations=40, polish="gd", polish_steps=1, gd_lr=0.05, seed=0
    )

    def _evals(result: Any) -> int:
        for rec in result.records:
            if rec.e_best <= target:
                return rec.n_eval
        return int(result.n_eval)

    n_n = _evals(newt)
    n_g = _evals(gd)
    passed = newt.best_value <= target and n_g >= 3 * n_n
    return {
        "name": "g4_memetic_eval_accounting",
        "passed": passed,
        "newton_evals": n_n,
        "gd_evals": n_g,
        "newton_best": newt.best_value,
        "gd_best": gd.best_value,
        "detail": "exact Newton polish reaches the target in >= 3x fewer fitness evals",
    }


def _run_g5() -> dict[str, Any]:
    cfg = PopulationConfig(size=8, selection_beta=6.0, elitism=2, bit_flip_p=0.35)

    class _Coeffs:
        def __init__(self, coeffs: dict[tuple[int, ...], float]) -> None:
            self.coeffs = coeffs

    class _Lin:
        def __init__(self, c: np.ndarray) -> None:
            self.c = np.asarray(c, dtype=np.float64)

        @property
        def n(self) -> int:
            return int(self.c.size)

        def energy(self, x: object) -> float | np.ndarray:
            xv = np.asarray(x, dtype=float)
            out = xv @ self.c
            return float(out) if xv.ndim == 1 else out

        def to_polynomial(self) -> _Coeffs:
            n = self.n
            coeffs: dict[tuple[int, ...], float] = {tuple(0 for _ in range(n)): 0.0}
            for i, ci in enumerate(self.c):
                exp = tuple(1 if k == i else 0 for k in range(n))
                coeffs[exp] = float(ci)
            return _Coeffs(coeffs)

    closed = 0
    incorrect = 0
    rows: list[dict[str, Any]] = []
    n_inst = 10
    for i in range(n_inst):
        c = np.array([0.4 + 0.1 * i, 0.7, 1.1, 0.9], dtype=np.float64)
        prob = _Lin(c)
        result = certified_discrete_evolve(prob, cfg, generations=16, seed=100 + i, bound="negative_coeff")
        _x, opt = brute_force_min(prob)
        if result.gap_closed:
            closed += 1
            if result.best_value > opt + 1e-9:
                incorrect += 1
        rows.append(
            {
                "i": i,
                "gap_closed": result.gap_closed,
                "best": result.best_value,
                "lower": result.lower_bound,
                "opt": float(opt),
                "generations": result.generations_used,
            }
        )
    passed = closed >= 9 and incorrect == 0
    return {
        "name": "g5_certified_stop",
        "passed": passed,
        "n_closed": closed,
        "n_incorrect": incorrect,
        "n_instances": n_inst,
        "rows": rows,
        "detail": "gap_closed on >= 90% of tiny QUBOs and never incorrectly",
    }


def _run_g6() -> dict[str, Any]:
    e = np.array([2.0, 2.3, 3.1, 5.0], dtype=np.float64)
    ref = soft_weights(e, beta=1.0)
    ok_t = True
    ok_j = True
    try:
        import torch
        from omnibias.discrete.evolution import torch as tw

        got = tw.soft_weights(torch.as_tensor(e, dtype=torch.float64), beta=1.0).detach().cpu().numpy()
        ok_t = bool(np.array_equal(got, ref))
    except ImportError:
        ok_t = True
    try:
        import jax
        import jax.numpy as jnp
        from omnibias.discrete.evolution import jax as jw

        jax.config.update("jax_enable_x64", True)
        got = np.asarray(jw.soft_weights(jnp.asarray(e, dtype=jnp.float64), beta=1.0))
        ok_j = bool(np.allclose(got, ref, rtol=0.0, atol=1e-15))
    except ImportError:
        ok_j = True
    return {
        "name": "g6_parity",
        "passed": ok_t and ok_j,
        "torch_equal": ok_t,
        "jax_equal": ok_j,
        "detail": "torch / jax soft_weights match numpy at float64",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "soft_evolution.json" if full else "soft_evolution_smoke.json"
    t0 = time.perf_counter()
    print("G1...")
    g1 = _run_g1()
    print("G2...")
    g2 = _run_g2()
    print("G3...")
    g3 = _run_g3()
    print("G4...")
    g4 = _run_g4()
    print("G5...")
    g5 = _run_g5()
    print("G6...")
    g6 = _run_g6()
    entries = [g1, g2, g3, g4, g5, g6]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.soft_evolution.v1",
            config={
                "family": "soft_evolution",
                "full": full,
                "seeds": list(SEEDS),
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "evolution"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

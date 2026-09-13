# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Scientific neuromanifold controls, then equal-budget seeded comparisons.

Default: fast Bratu/interval controls and three fixed seeds. ``--full`` runs
twenty fixed seeds and a 10k-parameter QGT action. All discoveries here are
manufactured benchmark cases, not new physical laws or continuum theorems.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from _common import provenance, rss_mb  # type: ignore[import-not-found]
from omnibias.core.verified.interval import Interval as I
from omnibias.core.verified.transcend import cosh_iv, sinh_iv
from omnibias.dynamics.continuation import certify_event, certify_segment
from omnibias.geometry.continuation import ImplicitFamily, continue_branch, locate_fold
from omnibias.pinn.inverse.observation import (
    ObservationModel,
    identifiability_report,
    observation_information,
)
from omnibias.submodular._core.greedy import brute_force_max, greedy_maximize
from omnibias.submodular.design import information_design_problem
from omnibias.symbolic.reduction import (
    ParameterReduction,
    ParameterTerm,
    evaluate_reduction,
    reduction_candidate,
)
from omnibias.verify.neuromanifold.scientific import (
    certify_identifiability,
    certify_quantum_geometry,
    certify_reduction,
)


def _pde_controls() -> dict[str, Any]:
    def value(y: np.ndarray) -> np.ndarray:
        a, p = y
        return np.array([p * np.cosh(a / 2) ** 2 - 2 * a * a])

    def jacobian(y: np.ndarray) -> np.ndarray:
        a, p = y
        return np.array([[p * np.sinh(a) / 2 - 4 * a, np.cosh(a / 2) ** 2]])

    def second(y: np.ndarray) -> np.ndarray:
        a, p = y
        return np.array([[[p * np.cosh(a) / 2 - 4, np.sinh(a) / 2], [np.sinh(a) / 2, 0.0]]])

    family = ImplicitFamily(value, jacobian, second, name="analytic Bratu solution curve")
    event = locate_fold(family, np.array([2.4, 3.51]))
    start = np.array([1.0, 2 / np.cosh(0.5) ** 2])
    branch = continue_branch(family, start, direction=np.array([1.0, 1.0]), step=0.06, n_steps=70)

    def augmented(z: list[I]) -> list[I]:
        a, p = z
        return [p * cosh_iv(a / 2) ** 2 - 2 * a * a, p * sinh_iv(a) / 2 - 4 * a]

    def augmented_jac(z: list[I]) -> list[list[I]]:
        a, p = z
        return [
            [p * sinh_iv(a) / 2 - 4 * a, cosh_iv(a / 2) ** 2],
            [p * cosh_iv(a) / 2 - 4, sinh_iv(a) / 2],
        ]

    def conditions(z: list[I]) -> dict[str, I]:
        a, p = z
        return {
            "transversality": cosh_iv(a / 2) ** 2,
            "quadratic": (p * cosh_iv(a) / 2 - 4) / 2,
            "complement_gap": I.point(1.0),
        }

    certified = certify_event(
        "fold", augmented, augmented_jac, event.point.tolist(), conditions, radius=1e-6
    )
    segment = certify_segment(
        lambda x, s: [x[0] * x[0] - s],
        lambda x, s: [[2 * x[0]]],
        I(1, 1.01),
        [1.0025],
        slope=[0.5],
        radius=0.007,
    )
    passed = (
        event.nondegenerate
        and certified.certified
        and segment.certified
        and branch.points[-1, 0] > event.point[0]
    )
    return {
        "passed": bool(passed),
        "bratu_fold": event.point.tolist(),
        "bratu_event_interval": None if certified.root is None else certified.root.enclosure,
        "continuation_points": len(branch.points),
        "max_residual": float(max(branch.residuals)),
        "status": branch.status,
        "uniform_segment_certified": segment.certified,
        "scope": "analytic Bratu solution-family parameter fold; no arbitrary-PDE continuum inference",
    }


def _certificate_controls() -> dict[str, Any]:
    full = certify_identifiability(
        lambda box: [[I.point(1), box[0]], [I.point(0), I.point(1)]],
        [I(-0.01, 0.01)],
        physical=[0],
        nuisance=[1],
        provider_assumption="displayed analytic two-observation derivative",
    )
    duplicate = certify_identifiability(
        lambda box: [[I.point(1), I.point(1)]],
        [I(0, 1)],
        physical=[0],
        nuisance=[1],
        provider_assumption="duplicate derivative columns",
    )
    reduced = certify_reduction(
        lambda b: {0: b[0] * b[1] * b[1], 1: 2 * b[0] * b[1]},
        [I(0, 0.001), I(-1, 1)],
        orders=(0, 1),
        error_budget=0.0021,
        provider_assumption="exact error epsilon*x² and its x derivative",
    )
    nonuniform = certify_reduction(
        lambda b: {0: b[0] / (b[0] + b[1])},
        [I(0, 0.01), I(0, 1)],
        error_budget=0.1,
        max_boxes=16,
        provider_assumption="rational error away from the singular corner",
    )
    q = certify_quantum_geometry([[1, 0, 0], [1, 1, 1], [1, 2, 2]], [1, 2, 1])
    return {
        "passed": full.certified
        and not duplicate.certified
        and reduced.certified
        and not nonuniform.certified
        and q.certified,
        "full_rank_certified": full.certified,
        "degenerate_refused": not duplicate.certified,
        "uniform_reduction_certified": reduced.certified,
        "nonuniform_unresolved_boxes": len(nonuniform.unresolved_regions),
        "finite_quantum_rank": q.rank,
        "finite_quantum_nullity": len(q.null_basis),
    }


def _design_seed(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(10, 3))
    theta = np.array([0.4, -0.2, 0.7])
    noise = rng.normal(scale=0.05, size=10)

    def value(t: np.ndarray, d: np.ndarray) -> np.ndarray:
        return d @ t

    def jac(t: np.ndarray, d: np.ndarray) -> np.ndarray:
        return d

    model = ObservationModel(value, jac, ("a", "b", "c"))
    info = observation_information(model, theta, design, covariance=0.05**2)
    budget = 4
    problem = information_design_problem(
        [row[None, :] / 0.05 for row in design], np.eye(3), budget=budget
    )
    selected, score = greedy_maximize(problem.function, problem.matroid)
    _, optimum = brute_force_max(problem.function, problem.matroid)
    uniform = np.linspace(0, 9, budget).round().astype(int)
    random_indices = rng.choice(10, budget, replace=False)
    indices = np.flatnonzero(selected)
    errors = {}
    for name, index in (("designed", indices), ("uniform", uniform), ("random", random_indices)):
        prediction = np.linalg.lstsq(design[index], (design @ theta + noise)[index], rcond=None)[0]
        errors[name] = float(np.linalg.norm(prediction - theta))
    return {
        "seed": seed,
        "budget": budget,
        "selected": indices.tolist(),
        "parameter_errors": errors,
        "numerical_rank": identifiability_report(info).rank,
        "design_score": score,
        "oracle_score": optimum,
        "theoretical_greedy_ratio": 1 - 1 / np.e,
        "passed": bool(len(indices) == budget and score + 1e-10 >= (1 - 1 / np.e) * optimum),
    }


def _quantum_seed(seed: int) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.ferminet.operator_sr import matrixfree_sr_step

    rng = np.random.default_rng(seed)
    features = jnp.asarray(rng.normal(size=(16, 8)))
    theta = jnp.asarray(rng.normal(size=8) * 0.05)
    energies = jnp.asarray(rng.normal(size=16))
    damping = 0.1
    lr = 0.05

    def logpsi(p: Any, x: Any) -> Any:
        return jnp.dot(p, x)

    weights = jax.nn.softmax(2 * features @ theta)
    centered = features - jnp.sum(weights[:, None] * features, axis=0)
    matrix = centered.T @ (weights[:, None] * centered)
    grad = 2 * centered.T @ (weights * energies)
    start = time.perf_counter()
    result = matrixfree_sr_step(
        logpsi,
        theta,
        features,
        energies,
        weights=weights,
        damping=damping,
        learning_rate=lr,
        chunk_size=8,
        rtol=1e-11,
        sampling_kind="exact_enumeration",
    )
    result.params.block_until_ready()
    elapsed = time.perf_counter() - start
    dense = theta - lr * jnp.linalg.solve(matrix + damping * jnp.eye(8), grad)
    error = float(jnp.linalg.norm(result.params - dense))
    return {
        "seed": seed,
        "sample_budget": 16,
        "parameter_count": 8,
        "dense_update_error": error,
        "solve_iterations": int(result.solve.iterations),
        "relative_residual": float(result.solve.relative_residual),
        "wall_seconds": elapsed,
        "passed": bool(result.accepted and error < 1e-9),
        "scope": "same finite score/energy table and weighted sample budget; no ground-state discovery claim",
    }


def _large_operator() -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.ferminet.operator_sr import qgt_operator

    p = jnp.zeros(10000)
    samples = jnp.arange(256, dtype=float)[:, None] / 256

    def logpsi(theta: Any, x: Any) -> Any:
        return jnp.dot(theta, jnp.sin(jnp.arange(theta.size) * 0.0001 + x[0]))

    op = qgt_operator(logpsi, p, samples, chunk_size=32)
    start = time.perf_counter()
    result = op.matvec(jnp.ones_like(p))
    result.block_until_ready()
    return {
        "parameters": 10000,
        "observations": 256,
        "chunk_size": 32,
        "wall_seconds": time.perf_counter() - start,
        "dense_covariance_storage_bytes_estimate": 10000**2 * 8,
        "process_max_rss_mib": rss_mb(),
        "finite_action": bool(jnp.all(jnp.isfinite(result))),
        "scope": "operator action; dense covariance was not allocated or timed",
    }


def run(*, full: bool = False) -> dict[str, Any]:
    seeds = list(range(1729, 1729 + (20 if full else 3)))
    started = time.perf_counter()
    pde = _pde_controls()
    controls = _certificate_controls()
    design = [_design_seed(seed) for seed in seeds]
    quantum = [_quantum_seed(seed) for seed in seeds]

    def model(t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return t[0] * np.exp(-t[1] * x) + t[2] * np.exp(-t[1] * x)

    transform = ParameterReduction(
        (ParameterTerm(0), ParameterTerm(1), ParameterTerm(2, power=1)), 3
    )
    reduction = evaluate_reduction(
        reduction_candidate(model, transform),
        [0.1, 0.05, 0.025],
        np.array([2.0, 0.4, 1.0]),
        np.linspace(0, 3, 31),
    )
    aggregate = {
        name: float(np.mean([row["parameter_errors"][name] for row in design]))
        for name in ("designed", "uniform", "random")
    }
    skill = 1 - aggregate["designed"] / aggregate["uniform"]
    payload = provenance(
        schema="neuromanifold-science-v1",
        config={
            "full": full,
            "seeds": seeds,
            "equal_measurement_budget": 4,
            "quantum_sample_budget": 16,
        },
    )
    payload.update(
        {
            "pde_first": pde,
            "certificate_controls": controls,
            "design": design,
            "quantum": quantum,
            "design_mean_error": aggregate,
            "design_recovery_skill_vs_uniform": skill,
            "design_empirical_gate_earned": bool(full and skill > 0),
            "reduction_observed_orders": reduction.observed_orders.tolist(),
            "reduction_scope": reduction.scope,
            "all_passed": bool(
                pde["passed"]
                and controls["passed"]
                and all(row["passed"] for row in design + quantum)
            ),
            "new_scientific_discovery_claim": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    if full:
        payload["large_operator"] = _large_operator()
        payload["all_passed"] = bool(
            payload["all_passed"] and payload["large_operator"]["finite_action"]
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(full=args.full)
    output = args.output or Path(
        os.environ.get("OMNIBIAS_SCRATCH", "artifacts")
    ) / "neuromanifold_science" / ("full.json" if args.full else "smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "all_passed": result["all_passed"],
                "full": args.full,
                "seeds": len(result["config"]["seeds"]),
                "output": str(output),
            }
        )
    )
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

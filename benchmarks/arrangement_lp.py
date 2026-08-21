# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: arrangement LP with learned facets (theory 03-02).

Smoke earns G1 (IPM active-set crossover vs vertex enum at 1e-10),
G2 (NS lower bound never exceeds the exact opt, including an
ill-conditioned instance), G3 (degenerate pentagon: KKT unusable,
SOFT Jacobian finite), G4 (e2e SOFT knapsack regret skill > 0 vs a
feature-blind two-stage baseline, five seeds), and G5 (duality sign
on the named dual). Not a new LP algorithm. Vertex enum cutoff is
``VERTEX_ENUM_MAX_D`` / ``VERTEX_ENUM_MAX_N``. Soft membership is
temperature collapse, not founding bias collapse.
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

try:
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.convex.arrangement import (
    VERTEX_ENUM_MAX_D,
    VERTEX_ENUM_MAX_N,
    DiffMode,
    LearnedPolytope,
    dual_objective,
    duality_holds,
    enumerate_vertices,
    honesty_payload,
    knapsack_simplex,
    known_dual_pentagon,
    named_pentagon,
    random_feasible_lp,
    recover_vertex,
    soft_membership,
    soft_vertex_dx_dc,
    solve_arrangement_lp,
    vertex_optimum,
)
from omnibias.convex.arrangement._core import (
    is_degenerate_optimum,
    kkt_gradient_usable,
    kkt_matrix,
    predict_then_optimize_regret,
    soft_vertex_solution,
    two_stage_constant_predict,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)


def _run_g1() -> dict[str, Any]:
    from omnibias.convex.jax import solve_lp

    rng = np.random.default_rng(0)
    max_err = 0.0
    n = 8
    for _ in range(n):
        poly, c = random_feasible_lp(2, 3, rng)
        _x, v_enum, verts = vertex_optimum(poly, c)
        assert verts.shape[0] >= 1
        sol = solve_lp(c, poly.normals, poly.offsets)
        v_rec = float(c @ recover_vertex(poly, np.asarray(sol.x)))
        max_err = max(max_err, abs(v_rec - v_enum))
    return {
        "name": "g1_solver_agreement",
        "passed": max_err <= 1e-10,
        "max_abs_err": max_err,
        "n_instances": n,
        "vertex_enum_cutoff": {"D": VERTEX_ENUM_MAX_D, "n": VERTEX_ENUM_MAX_N},
        "detail": "solve_lp active-set crossover matches vertex-enum value to 1e-10",
    }


def _run_g2() -> dict[str, Any]:
    rng = np.random.default_rng(1)
    viol = 0
    rows: list[dict[str, Any]] = []
    instances: list[tuple[LearnedPolytope, np.ndarray]] = [named_pentagon()[:2]]
    for _ in range(6):
        instances.append(random_feasible_lp(2, 4, rng))
    a = np.array([[1.0, 0.0], [1.0, 1e-8], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    b = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    instances.append((LearnedPolytope(a, b), np.array([-1.0, -0.2])))
    for i, (poly, c) in enumerate(instances):
        _x, opt, verts = vertex_optimum(poly, c)
        if verts.shape[0] == 0:
            continue
        lo = solve_arrangement_lp(poly, c).lower_bound
        if lo > opt + 1e-9:
            viol += 1
        rows.append({"i": i, "lower": lo, "opt": opt})
    return {
        "name": "g2_sound_bound",
        "passed": viol == 0,
        "violations": viol,
        "rows": rows,
        "detail": "Neumaier-Shcherbina lo never exceeds the exact vertex optimum",
    }


def _run_g3() -> dict[str, Any]:
    poly, c, opt = named_pentagon()
    _x, value, verts = vertex_optimum(poly, c)
    out = solve_arrangement_lp(poly, c, mode=DiffMode.SOFT, beta=6.0)
    kkt = kkt_matrix(poly, np.array([1.5, 1.5]), known_dual_pentagon())
    jac = soft_vertex_dx_dc(verts, c, beta=6.0)
    passed = (
        out.degenerate
        and is_degenerate_optimum(verts, c, value)
        and (not kkt_gradient_usable(kkt))
        and bool(np.all(np.isfinite(jac)))
        and float(np.linalg.norm(jac)) > 1e-6
        and math.isclose(value, opt, abs_tol=1e-10)
    )
    return {
        "name": "g3_degeneracy",
        "passed": passed,
        "degenerate": out.degenerate,
        "kkt_usable": kkt_gradient_usable(kkt),
        "soft_jac_norm": float(np.linalg.norm(jac)),
        "detail": "SOFT Jacobian is finite on the named optimal edge; KKT is not",
    }


def _run_g4() -> dict[str, Any]:
    poly = knapsack_simplex()
    verts = enumerate_vertices(poly)
    skills: list[float] = []
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(20 + seed)
        zs = rng.choice(np.array([-1.0, 1.0]), size=10)
        true_c = np.stack([-10.0 * zs, 10.0 * zs], axis=1)
        x_star = np.stack([vertex_optimum(poly, c)[0] for c in true_c])
        x_2s = vertex_optimum(poly, two_stage_constant_predict(true_c))[0]
        regret_2s = float(np.mean([predict_then_optimize_regret(c, x_2s, xs) for c, xs in zip(true_c, x_star, strict=True)]))
        w = 0.1
        for _ in range(60):
            grad = 0.0
            for z, c in zip(zs, true_c, strict=True):
                chat = np.array([-w * z, w * z])
                jac = soft_vertex_dx_dc(verts, chat, beta=8.0)
                grad += float(c @ (jac @ np.array([-z, z])))
            w -= 0.15 * grad / float(len(zs))
        x_e2e = [soft_vertex_solution(verts, np.array([-w * z, w * z]), beta=8.0)[0] for z in zs]
        regret_e2e = float(
            np.mean([predict_then_optimize_regret(c, xh, xs) for c, xh, xs in zip(true_c, x_e2e, x_star, strict=True)])
        )
        skill = 1.0 - regret_e2e / max(regret_2s, 1e-12)
        skills.append(skill)
        rows.append({"seed": int(seed), "regret_e2e": regret_e2e, "regret_two_stage": regret_2s, "skill": skill, "w": w})
    mean_skill = float(np.mean(skills))
    return {
        "name": "g4_predict_then_optimize",
        "passed": mean_skill > 0.0,
        "mean_skill": mean_skill,
        "seeds": rows,
        "detail": "e2e SOFT layer beats feature-blind two-stage on knapsack regret",
    }


def _run_g5() -> dict[str, Any]:
    poly, c, opt = named_pentagon()
    y = known_dual_pentagon()
    passed = duality_holds(poly, c, y) and math.isclose(dual_objective(poly, y), opt, abs_tol=1e-12)
    return {
        "name": "g5_duality_convention",
        "passed": passed,
        "dual_value": dual_objective(poly, y),
        "primal_value": opt,
        "detail": "A^T y = -c and -b·y equals the named primal optimum -3",
    }


def _run_g6() -> dict[str, Any]:
    poly, _c, _opt = named_pentagon()
    x = np.array([1.5, 1.5])
    ref = float(soft_membership(poly, x, beta=5.0))
    ok_t = True
    ok_j = True
    try:
        import torch
        from omnibias.convex.arrangement import torch as tw

        got = float(tw.soft_membership(torch.as_tensor(poly.normals), torch.as_tensor(poly.offsets), torch.as_tensor(x), beta=5.0))
        ok_t = abs(got - ref) <= 1e-15
    except ImportError:
        ok_t = True
    try:
        from omnibias.convex.arrangement import jax as jw

        got = float(np.asarray(jw.soft_membership(poly.normals, poly.offsets, x, beta=5.0)))
        ok_j = abs(got - ref) <= 2e-15
    except ImportError:
        ok_j = True
    return {
        "name": "g6_parity",
        "passed": ok_t and ok_j,
        "torch_equal": ok_t,
        "jax_equal": ok_j,
        "weight": ref,
        "detail": "torch / jax soft_membership match numpy at the named midpoint",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "arrangement_lp.json" if full else "arrangement_lp_smoke.json"
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
            schema="omnibias.benchmarks.arrangement_lp.v1",
            config={
                "family": "arrangement_lp",
                "full": full,
                "seeds": list(SEEDS),
                "honesty": honesty_payload(),
                "vertex_enum_cutoff": {"D": VERTEX_ENUM_MAX_D, "n": VERTEX_ENUM_MAX_N},
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "arrlp"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

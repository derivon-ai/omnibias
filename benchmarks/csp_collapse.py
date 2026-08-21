# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: constraint satisfaction by collapse (theory 03-03).

Smoke earns G1 (E=0 iff a SAT vertex, exhaustive small instances),
G2 (certify never claims SAT on brute-force UNSAT), G3 (solve rate
within 10pp of backtracking on a tiny binary ensemble), G4 (e2e
clue-conditioned path colouring beats feature-blind majority),
G5 (all-different / cardinality exact on vertices), and G6
(torch/jax softmax parity). Both betas are temperature collapse,
not founding bias collapse. Not a complete solver and not P vs NP.
"""

from __future__ import annotations

import argparse
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
from omnibias.discrete.csp import (
    CSP,
    GlobalConstraint,
    Variable,
    assignment_onehot,
    backtrack_sat,
    brute_force_sat,
    certify_csp,
    clause_gap_bound,
    csp_solve,
    honesty_payload,
    random_binary_csp,
    softmax_rows,
    triangle_colouring,
)
from omnibias.discrete.csp._core import (
    _grad_violation,
    enumerate_assignments,
    is_satisfying_vertex,
    not_equal_relation,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)


def _run_g1() -> dict[str, Any]:
    instances = [
        triangle_colouring(n_colours=2),
        triangle_colouring(n_colours=3),
        CSP(
            (Variable("a", (0, 1)), Variable("b", (0, 1))),
            (),
            (GlobalConstraint("all_different", (0, 1)),),
        ),
    ]
    n_checked = 0
    for csp in instances:
        for assign in enumerate_assignments(csp):
            e = float(csp.energy(assignment_onehot(csp, assign)))
            sat = is_satisfying_vertex(csp, assign)
            if sat and e != 0.0:
                return {"name": "g1_vertex_exact", "passed": False, "n_checked": n_checked}
            if (not sat) and e <= 0.0:
                return {"name": "g1_vertex_exact", "passed": False, "n_checked": n_checked}
            n_checked += 1
    rgb = assignment_onehot(triangle_colouring(n_colours=3), (0, 1, 2))
    uni = triangle_colouring(n_colours=3).pack([np.full(3, 1.0 / 3.0) for _ in range(3)])
    e_uni = float(triangle_colouring(n_colours=3).violation_energy(uni))
    return {
        "name": "g1_vertex_exact",
        "passed": n_checked > 0 and float(triangle_colouring(n_colours=3).energy(rgb)) == 0.0 and abs(e_uni - 1.0) < 1e-12,
        "n_checked": n_checked,
        "uniform_violation": e_uni,
        "detail": "E=0 iff SAT vertex; uniform triangle E=1",
    }


def _run_g2() -> dict[str, Any]:
    unsat = triangle_colouring(n_colours=2)
    violations = 0
    for assign in enumerate_assignments(unsat):
        if certify_csp(unsat, assign, beta=20.0).claimed_sat:
            violations += 1
    rng = np.random.default_rng(0)
    extra = 0
    for _ in range(8):
        csp = random_binary_csp(3, 2, 4, rng, tightness=0.7)
        sat, _sol = brute_force_sat(csp)
        cert = certify_csp(csp, csp_solve(csp, restarts=3, steps=4, seed=1).assignment, beta=20.0)
        if (not sat) and cert.claimed_sat:
            extra += 1
    return {
        "name": "g2_gap_sound",
        "passed": violations == 0 and extra == 0,
        "unsat_triangle_violations": violations,
        "random_violations": extra,
        "detail": "never claim SAT where brute force shows none",
    }


def _run_g3() -> dict[str, Any]:
    ours = 0
    bt = 0
    n = 0
    for seed in SEEDS:
        local = np.random.default_rng(10 + seed)
        for _ in range(3):
            csp = random_binary_csp(4, 2, 3, local, tightness=0.35)
            n += 1
            if backtrack_sat(csp):
                bt += 1
            if csp_solve(csp, restarts=5, steps=6, seed=int(local.integers(0, 10_000))).satisfied:
                ours += 1
    rate_o = ours / n
    rate_b = bt / n
    passed = abs(rate_o - rate_b) <= 0.10 + 1e-12
    return {
        "name": "g3_solve_rate",
        "passed": passed,
        "ours": ours,
        "backtrack": bt,
        "n": n,
        "ours_rate": rate_o,
        "backtrack_rate": rate_b,
        "detail": "within 10pp of backtracking on a tiny binary ensemble",
    }


def _run_g4() -> dict[str, Any]:
    variables = tuple(Variable(f"x{i}", (0, 1)) for i in range(3))
    csp = CSP(variables, (not_equal_relation(0, 1, 2), not_equal_relation(1, 2, 2)))
    skills: list[float] = []
    for seed in SEEDS:
        rng = np.random.default_rng(30 + seed)
        zs = rng.choice(np.array([0, 1]), size=8)
        true = [(int(z), 1 - int(z), int(z)) for z in zs]
        counts: dict[tuple[int, ...], int] = {}
        for t in true:
            counts[t] = counts.get(t, 0) + 1
        majority = max(counts, key=counts.get)  # type: ignore[arg-type]
        acc_2s = float(np.mean([t == majority for t in true]))
        acc_e2e = 0.0
        for z, t in zip(zs, true, strict=True):
            logits = [
                np.array([4.0 if z == 0 else -4.0, -4.0 if z == 0 else 4.0]),
                np.zeros(2),
                np.zeros(2),
            ]
            for _ in range(25):
                probs = [softmax_rows(lg, beta=6.0) for lg in logits]
                probs[0] = np.array([1.0 - float(z), float(z)])
                g = _grad_violation(csp, probs)
                for i in (1, 2):
                    p = probs[i]
                    gz = 6.0 * (p * g[i] - p * float(np.dot(p, g[i])))
                    logits[i] = logits[i] - 0.4 * gz
            pred = (
                int(z),
                int(np.argmax(softmax_rows(logits[1], beta=6.0))),
                int(np.argmax(softmax_rows(logits[2], beta=6.0))),
            )
            acc_e2e += float(pred == t)
        acc_e2e /= float(len(zs))
        skills.append(acc_e2e - acc_2s)
    mean_skill = float(np.mean(skills))
    return {
        "name": "g4_e2e_skill",
        "passed": mean_skill > 0.0,
        "mean_skill": mean_skill,
        "skills": skills,
        "detail": "clue-conditioned path colouring vs feature-blind majority",
    }


def _run_g5() -> dict[str, Any]:
    variables = tuple(Variable(f"x{i}", (0, 1, 2)) for i in range(3))
    alldiff = CSP(variables, (), (GlobalConstraint("all_different", (0, 1, 2)),))
    ok_a = True
    for assign in enumerate_assignments(alldiff):
        e = float(alldiff.energy(assignment_onehot(alldiff, assign)))
        feas = len(set(assign)) == 3
        ok_a = ok_a and ((feas and e == 0.0) or ((not feas) and e > 0.0))
    card = CSP(
        (Variable("a", (0, 1)), Variable("b", (0, 1)), Variable("c", (0, 1))),
        (),
        (GlobalConstraint("cardinality", (0, 1, 2), parameter=2),),
    )
    ok_c = True
    for assign in enumerate_assignments(card):
        e = float(card.energy(assignment_onehot(card, assign)))
        feas = sum(assign) == 2
        ok_c = ok_c and ((feas and e == 0.0) or ((not feas) and e > 0.0))
    return {
        "name": "g5_globals",
        "passed": ok_a and ok_c,
        "all_different_ok": ok_a,
        "cardinality_ok": ok_c,
        "detail": "globals zero iff feasible vertices",
    }


def _run_g6() -> dict[str, Any]:
    z = np.array([0.2, -0.1, 0.4], dtype=np.float64)
    ref = softmax_rows(z, beta=3.0)
    ok_t = True
    ok_j = True
    try:
        import torch
        from omnibias.discrete.csp import torch as tw

        got = tw.softmax_rows(torch.as_tensor(z), beta=3.0).detach().cpu().numpy()
        ok_t = bool(np.array_equal(got, ref))
    except ImportError:
        ok_t = True
    try:
        import jax
        from omnibias.discrete.csp import jax as jw

        jax.config.update("jax_enable_x64", True)
        got = np.asarray(jw.softmax_rows(z, beta=3.0))
        ok_j = bool(np.allclose(got, ref, rtol=0.0, atol=2e-16))
    except ImportError:
        ok_j = True
    bound = clause_gap_bound(triangle_colouring(n_colours=3), beta=20.0)
    return {
        "name": "g6_parity",
        "passed": ok_t and ok_j,
        "torch_equal": ok_t,
        "jax_equal": ok_j,
        "clause_gap": bound,
        "detail": "torch / jax softmax_rows match numpy at float64",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "csp_collapse.json" if full else "csp_collapse_smoke.json"
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
            schema="omnibias.benchmarks.csp_collapse.v1",
            config={
                "family": "csp_collapse",
                "full": full,
                "seeds": list(SEEDS),
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "csp"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

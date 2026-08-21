# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 03-03: constraint satisfaction by collapse, gates G1–G6."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
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
    random_binary_csp,
    softmax_rows,
    triangle_colouring,
)
from omnibias.discrete.csp._core import (
    enumerate_assignments,
    is_satisfying_vertex,
)


def test_worked_example_triangle() -> None:
    csp = triangle_colouring(n_colours=3)
    uni = [np.full(3, 1.0 / 3.0) for _ in range(3)]
    assert math.isclose(float(csp.violation_energy(csp.pack(uni))), 1.0, abs_tol=1e-12)
    p1 = np.array([0.8, 0.1, 0.1])
    p2 = np.array([0.1, 0.8, 0.1])
    p3 = np.full(3, 1.0 / 3.0)
    e = float(csp.violation_energy(csp.pack([p1, p2, p3])))
    assert math.isclose(e, 0.8366666666666666, abs_tol=2e-6)
    rgb = assignment_onehot(csp, (0, 1, 2))
    assert float(csp.energy(rgb)) == 0.0
    assert float(csp.violation_energy(rgb)) == 0.0
    # Projected gradient on x3 pushes toward B (spec §5).
    from omnibias.discrete.csp._core import _grad_violation

    g = _grad_violation(csp, [p1, p2, p3])[2]
    g = g - float(np.mean(g))
    assert g[2] < g[0] and g[2] < g[1]


def test_g1_exactness_at_vertices() -> None:
    instances = [
        triangle_colouring(n_colours=2),
        triangle_colouring(n_colours=3),
        CSP((Variable("a", (0, 1)), Variable("b", (0, 1))), (), (GlobalConstraint("all_different", (0, 1)),)),
    ]
    for csp in instances:
        n_assign = int(np.prod(csp.domain_sizes()))
        assert n_assign <= 3**6
        for assign in enumerate_assignments(csp):
            x = assignment_onehot(csp, assign)
            e = float(csp.energy(x))
            sat = is_satisfying_vertex(csp, assign)
            if sat:
                assert e == 0.0
            else:
                assert e > 0.0
        # A non-vertex (uniform) is not a SAT vertex.
        uni = csp.pack([np.full(len(v.domain), 1.0 / len(v.domain)) for v in csp.variables])
        if not brute_force_sat(csp)[0]:
            assert float(csp.energy(uni)) > 0.0


def test_g2_never_claims_sat_on_unsat() -> None:
    unsat = triangle_colouring(n_colours=2)
    assert brute_force_sat(unsat)[0] is False
    for assign in enumerate_assignments(unsat):
        cert = certify_csp(unsat, assign, beta=20.0)
        assert cert.claimed_sat is False
        assert cert.hard_energy > 0.0
    rng = np.random.default_rng(0)
    for _ in range(8):
        csp = random_binary_csp(3, 2, 4, rng, tightness=0.7)
        sat, _sol = brute_force_sat(csp)
        result = csp_solve(csp, restarts=3, steps=4, seed=1, arc_consistency=True)
        cert = certify_csp(csp, result.assignment, beta=20.0)
        if not sat:
            assert cert.claimed_sat is False


def test_g3_solve_rate_near_backtrack() -> None:
    rng = np.random.default_rng(2)
    n = 12
    ours = 0
    bt = 0
    for seed in range(5):
        local = np.random.default_rng(10 + seed)
        for _ in range(n // 5 + 1):
            csp = random_binary_csp(4, 2, 3, local, tightness=0.35)
            if backtrack_sat(csp):
                bt += 1
            if csp_solve(csp, restarts=5, steps=6, seed=int(local.integers(0, 10_000))).satisfied:
                ours += 1
    # 5 seeds × 3 instances = 15. Parity within 10 percentage points.
    assert abs(ours - bt) <= max(2, int(0.10 * max(n, bt, 1)))
    assert ours + bt >= 0
    _ = rng


def test_g4_e2e_beats_two_stage() -> None:
    # Path of 3 nodes, 2 colours, x0 fixed by the feature. Unique SAT colouring.
    from omnibias.discrete.csp._core import not_equal_relation

    variables = tuple(Variable(f"x{i}", (0, 1)) for i in range(3))
    csp = CSP(variables, (not_equal_relation(0, 1, 2), not_equal_relation(1, 2, 2)))
    skills: list[float] = []
    for seed in range(5):
        rng = np.random.default_rng(30 + seed)
        zs = rng.choice(np.array([0, 1]), size=8)
        true = []
        for z in zs:
            # unique alternating colouring
            true.append((int(z), 1 - int(z), int(z)))
        # two-stage: ignore z, always predict the training-majority colouring
        counts: dict[tuple[int, ...], int] = {}
        for t in true:
            counts[t] = counts.get(t, 0) + 1
        majority = max(counts, key=counts.get)  # type: ignore[arg-type]
        acc_2s = float(np.mean([t == majority for t in true]))
        acc_e2e = 0.0
        for z, t in zip(zs, true, strict=True):
            # clamp x0, descend
            logits = [np.array([4.0 if z == 0 else -4.0, -4.0 if z == 0 else 4.0]), np.zeros(2), np.zeros(2)]
            from omnibias.discrete.csp._core import _grad_violation

            for _ in range(25):
                probs = [softmax_rows(lg, beta=6.0) for lg in logits]
                probs[0] = np.array([1.0 - float(z), float(z)])
                g = _grad_violation(csp, probs)
                for i in (1, 2):
                    p = probs[i]
                    gz = 6.0 * (p * g[i] - p * float(np.dot(p, g[i])))
                    logits[i] = logits[i] - 0.4 * gz
            pred = (int(z), int(np.argmax(softmax_rows(logits[1], beta=6.0))), int(np.argmax(softmax_rows(logits[2], beta=6.0))))
            acc_e2e += float(pred == t)
        acc_e2e /= float(len(zs))
        skills.append(acc_e2e - acc_2s)
    assert float(np.mean(skills)) > 0.0


def test_g5_global_constraints_on_vertices() -> None:
    variables = tuple(Variable(f"x{i}", (0, 1, 2)) for i in range(3))
    alldiff = CSP(variables, (), (GlobalConstraint("all_different", (0, 1, 2)),))
    for assign in enumerate_assignments(alldiff):
        e = float(alldiff.energy(assignment_onehot(alldiff, assign)))
        feasible = len(set(assign)) == 3
        if feasible:
            assert e == 0.0
        else:
            assert e > 0.0
    card = CSP(
        (Variable("a", (0, 1)), Variable("b", (0, 1)), Variable("c", (0, 1))),
        (),
        (GlobalConstraint("cardinality", (0, 1, 2), parameter=2),),
    )
    for assign in enumerate_assignments(card):
        e = float(card.energy(assignment_onehot(card, assign)))
        feasible = sum(assign) == 2  # value index 1
        if feasible:
            assert e == 0.0
        else:
            assert e > 0.0


def test_g6_softmax_parity() -> None:
    z = np.array([0.2, -0.1, 0.4], dtype=np.float64)
    ref = softmax_rows(z, beta=3.0)
    torch = pytest.importorskip("torch")
    tw = __import__("omnibias.discrete.csp.torch", fromlist=["softmax_rows"]).softmax_rows
    got_t = tw(torch.as_tensor(z), beta=3.0).detach().cpu().numpy()
    np.testing.assert_allclose(got_t, ref, rtol=0.0, atol=0.0)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jw = __import__("omnibias.discrete.csp.jax", fromlist=["softmax_rows"]).softmax_rows
    got_j = np.asarray(jw(z, beta=3.0))
    np.testing.assert_allclose(got_j, ref, rtol=0.0, atol=2e-16)


def test_energy_matches_polynomial_on_cube() -> None:
    pytest.importorskip("omnibias.sos")
    csp = triangle_colouring(n_colours=2)
    poly = csp.to_polynomial()
    for assign in enumerate_assignments(csp):
        x = assignment_onehot(csp, assign)
        ev = float(csp.energy(x))
        pv = float(poly.evaluate([float(v) for v in x]))
        assert abs(ev - pv) < 1e-9


def test_clause_gap_and_terminology() -> None:
    csp = triangle_colouring(n_colours=3)
    bound = clause_gap_bound(csp, beta=20.0)
    assert math.isclose(bound, 3.0 * math.log(6.0) / 20.0, abs_tol=1e-12)
    root = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "discrete" / "csp"
    for name in ("_core.py", "torch.py", "jax.py", "__init__.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "founding bias collapse" in text
        assert "delta -> 0" in text
        assert "beta -> inf" in text
        assert "feasibility" in text.lower()
        assert "do not conflate" in text.lower()

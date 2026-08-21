# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 03-02: arrangement LP / learned facets, gates G1–G5."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from omnibias.convex.arrangement import (
    DiffMode,
    LearnedPolytope,
    dual_objective,
    duality_holds,
    enumerate_vertices,
    knapsack_simplex,
    known_dual_pentagon,
    named_pentagon,
    random_feasible_lp,
    recover_vertex,
    soft_cell_gap_bound,
    soft_membership,
    soft_vertex_dx_dc,
    solve_arrangement_lp,
    sound_lower_bound,
    vertex_optimum,
)
from omnibias.convex.arrangement._core import (
    infer_box,
    is_degenerate_optimum,
    kkt_gradient_usable,
    kkt_matrix,
    predict_then_optimize_regret,
    soft_vertex_solution,
    two_stage_constant_predict,
)


def test_worked_example_soft_membership() -> None:
    poly, _c, _opt = named_pentagon()
    w = float(soft_membership(poly, np.array([1.5, 1.5]), beta=5.0))
    assert math.isclose(w, 0.4265470, rel_tol=0.0, abs_tol=2e-6)
    assert math.isclose(soft_cell_gap_bound(n_facets=5, beta=5.0), math.log(5.0) / 5.0)


def test_g5_duality_sign_convention() -> None:
    poly, c, opt = named_pentagon()
    y = known_dual_pentagon()
    assert duality_holds(poly, c, y)
    assert math.isclose(dual_objective(poly, y), opt, abs_tol=1e-12)
    assert math.isclose(float(c @ np.array([1.5, 1.5])), opt, abs_tol=1e-12)


def test_g3_degeneracy_soft_gradient() -> None:
    poly, c, opt = named_pentagon()
    x_star, value, verts = vertex_optimum(poly, c)
    assert math.isclose(value, opt, abs_tol=1e-10)
    assert is_degenerate_optimum(verts, c, value)
    out = solve_arrangement_lp(poly, c, mode=DiffMode.SOFT, beta=6.0)
    assert out.degenerate is True and out.feasible is True
    y = known_dual_pentagon()
    kkt = kkt_matrix(poly, np.array([1.5, 1.5]), y)
    assert kkt_gradient_usable(kkt) is False
    jac = soft_vertex_dx_dc(verts, c, beta=6.0)
    assert np.all(np.isfinite(jac))
    assert float(np.linalg.norm(jac)) > 1e-6
    xs, _w, _vs = soft_vertex_solution(verts, c, beta=6.0)
    np.testing.assert_allclose(xs, np.array([1.5, 1.5]), atol=0.05)


def test_g1_solver_agrees_with_vertex_enum() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from omnibias.convex.jax import solve_lp

    rng = np.random.default_rng(0)
    for _ in range(8):
        poly, c = random_feasible_lp(2, 3, rng)
        _x, v_enum, verts = vertex_optimum(poly, c)
        assert verts.shape[0] >= 1
        sol = solve_lp(c, poly.normals, poly.offsets)
        x_rec = recover_vertex(poly, np.asarray(sol.x))
        v_rec = float(c @ x_rec)
        assert abs(v_rec - v_enum) <= 1e-10


def test_g2_sound_bound_never_exceeds_opt() -> None:
    rng = np.random.default_rng(1)
    instances: list[tuple[LearnedPolytope, np.ndarray]] = [named_pentagon()[:2]]
    for _ in range(6):
        instances.append(random_feasible_lp(2, 4, rng))
    # Ill-conditioned: two nearly parallel facets.
    a = np.array(
        [[1.0, 0.0], [1.0, 1e-8], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float64,
    )
    b = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    instances.append((LearnedPolytope(a, b), np.array([-1.0, -0.2])))
    for poly, c in instances:
        _x, opt, verts = vertex_optimum(poly, c)
        if verts.shape[0] == 0:
            continue
        out = solve_arrangement_lp(poly, c)
        assert out.lower_bound <= opt + 1e-9
        y = np.zeros(poly.n)
        lo = sound_lower_bound(poly, c, y)
        assert lo <= opt + 1e-9


def test_g4_e2e_beats_two_stage() -> None:
    poly = knapsack_simplex()
    verts = enumerate_vertices(poly)
    skills: list[float] = []
    for seed in range(5):
        rng = np.random.default_rng(20 + seed)
        zs = rng.choice(np.array([-1.0, 1.0]), size=10)
        true_c = np.stack([-10.0 * zs, 10.0 * zs], axis=1)
        x_star = np.stack([vertex_optimum(poly, c)[0] for c in true_c])
        c_mean = two_stage_constant_predict(true_c)
        x_2s = vertex_optimum(poly, c_mean)[0]
        regret_2s = float(np.mean([predict_then_optimize_regret(c, x_2s, xs) for c, xs in zip(true_c, x_star, strict=True)]))
        w = 0.1
        for _ in range(60):
            grad = 0.0
            for z, c in zip(zs, true_c, strict=True):
                chat = np.array([-w * z, w * z], dtype=np.float64)
                jac = soft_vertex_dx_dc(verts, chat, beta=8.0)
                dchat = np.array([-z, z], dtype=np.float64)
                grad += float(c @ (jac @ dchat))
            w -= 0.15 * grad / float(len(zs))
        x_e2e = []
        for z in zs:
            chat = np.array([-w * z, w * z], dtype=np.float64)
            x_e2e.append(soft_vertex_solution(verts, chat, beta=8.0)[0])
        regret_e2e = float(
            np.mean([predict_then_optimize_regret(c, xh, xs) for c, xh, xs in zip(true_c, x_e2e, x_star, strict=True)])
        )
        skills.append(1.0 - regret_e2e / max(regret_2s, 1e-12))
    assert float(np.mean(skills)) > 0.0


def test_infeasible_is_reported() -> None:
    poly = LearnedPolytope(
        np.array([[1.0], [-1.0]], dtype=np.float64),
        np.array([-1.0, 0.0], dtype=np.float64),
    )
    out = solve_arrangement_lp(poly, np.array([1.0]))
    assert out.feasible is False
    assert math.isinf(out.value)


def test_vertex_enum_cutoff() -> None:
    poly = LearnedPolytope(np.eye(5, dtype=np.float64), np.ones(5))
    with pytest.raises(ValueError, match="small-instance"):
        enumerate_vertices(poly)


def test_g6_soft_membership_parity() -> None:
    poly, _c, _opt = named_pentagon()
    x = np.array([1.5, 1.5], dtype=np.float64)
    ref = float(soft_membership(poly, x, beta=5.0))
    torch = pytest.importorskip("torch")
    tw = __import__("omnibias.convex.arrangement.torch", fromlist=["soft_membership"]).soft_membership
    got_t = float(tw(torch.as_tensor(poly.normals), torch.as_tensor(poly.offsets), torch.as_tensor(x), beta=5.0))
    assert math.isclose(got_t, ref, rel_tol=0.0, abs_tol=1e-15)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jw = __import__("omnibias.convex.arrangement.jax", fromlist=["soft_membership"]).soft_membership
    got_j = float(np.asarray(jw(poly.normals, poly.offsets, x, beta=5.0)))
    assert math.isclose(got_j, ref, rel_tol=0.0, abs_tol=2e-15)


def test_terminology_note_is_present() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "convex" / "arrangement"
    for name in ("_core.py", "torch.py", "jax.py", "__init__.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "founding bias collapse" in text
        assert "delta -> 0" in text
        assert "beta -> inf" in text
        assert "feasibility" in text.lower()
        assert "do not conflate" in text.lower()


def test_infer_box_pentagon() -> None:
    poly, _c, _opt = named_pentagon()
    lo, hi = infer_box(poly)
    np.testing.assert_allclose(lo, [0.0, 0.0])
    np.testing.assert_allclose(hi, [2.0, 2.0])

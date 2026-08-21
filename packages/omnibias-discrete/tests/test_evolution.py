# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 03-01: soft-population evolution gates G1–G6."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.discrete.evolution import (
    GeometryMutation,
    PackGenes,
    PopulationConfig,
    certified_discrete_evolve,
    geometry_evolve,
    memetic_evolve,
    named_g2_objective,
    named_pack_target,
    named_quadratic,
    pack_fitness,
    selection_gap_bound,
    selection_stats,
    soft_population_evolve,
    soft_weights,
)
from omnibias.discrete.evolution._core import mutate_geometry, mutate_isotropic


def test_worked_example_selection() -> None:
    e = np.array([2.0, 2.3, 3.1, 5.0], dtype=np.float64)
    w, e_soft, e_best, gap = selection_stats(e, beta=1.0)
    assert e_best == 2.0
    assert math.isclose(e_soft, 2.347435, rel_tol=0.0, abs_tol=5e-6)
    assert math.isclose(gap, 0.347435, rel_tol=0.0, abs_tol=5e-6)
    bound = selection_gap_bound(population=4, beta=1.0)
    assert math.isclose(bound, math.log(4.0), rel_tol=0.0, abs_tol=1e-12)
    assert gap <= bound
    _w10, e_soft10, _b, gap10 = selection_stats(e, beta=10.0)
    assert math.isclose(e_soft10, 2.0142274, rel_tol=0.0, abs_tol=3e-5)
    assert gap10 <= selection_gap_bound(population=4, beta=10.0)
    assert w.shape == (4,)
    assert math.isclose(float(np.sum(w)), 1.0, abs_tol=1e-15)


def test_g1_selection_gap_sound_on_every_generation() -> None:
    cfg = PopulationConfig(size=6, selection_beta=2.5, elitism=1, mutation_sigma=0.2)
    rng = np.random.default_rng(0)
    init = rng.normal(0.0, 1.0, size=(6, 2))
    result = soft_population_evolve(named_g2_objective, init, cfg, generations=12, seed=0)
    assert result.gap_violations == 0
    assert result.records
    for rec in result.records:
        assert rec.measured_gap <= rec.bound + 1e-12
        assert rec.e_soft + 1e-12 >= rec.e_best


def test_g3_geometry_beats_isotropic() -> None:
    target = named_pack_target()
    cfg = PopulationConfig(size=8, selection_beta=3.0, elitism=1)
    mut = GeometryMutation(mean_sigma=0.05, spread_log_sigma=0.35, order_step_prob=0.25, weight_sigma=0.1)
    geo_vals: list[float] = []
    iso_vals: list[float] = []
    for seed in range(5):
        rng = np.random.default_rng(10 + seed)
        init = [
            PackGenes(float(rng.normal(0.0, 0.3)), float(np.exp(rng.normal(0.0, 0.4))), 1, float(rng.normal(0.0, 0.4)))
            for _ in range(cfg.size)
        ]
        geo = geometry_evolve(target, init, cfg, generations=24, seed=20 + seed, mutation=mut, isotropic=False)
        iso = geometry_evolve(
            target,
            init,
            cfg,
            generations=24,
            seed=20 + seed,
            mutation=mut,
            isotropic=True,
            isotropic_sigma=0.2,
        )
        geo_vals.append(geo.best_value)
        iso_vals.append(iso.best_value)
        assert geo.gap_violations == 0 and iso.gap_violations == 0
    assert float(np.mean(iso_vals)) >= 2.0 * float(np.mean(geo_vals))


def test_g4_newton_uses_fewer_evals_than_gd() -> None:
    x0, _h, fitness, grad_hess = named_quadratic()
    cfg = PopulationConfig(size=4, selection_beta=4.0, elitism=1, mutation_sigma=0.05)
    init = np.stack([x0 + 0.1 * k * np.array([1.0, -0.5]) for k in range(4)])
    target = 1e-6
    newt = memetic_evolve(
        fitness, grad_hess, init, cfg, generations=2, polish="newton", polish_steps=1, seed=0
    )
    gd = memetic_evolve(
        fitness, grad_hess, init, cfg, generations=40, polish="gd", polish_steps=1, gd_lr=0.05, seed=0
    )
    assert newt.best_value <= target
    # Count evals until each first record with e_best <= target (plus init).
    def _evals_to_target(result: object) -> int:
        recs = result.records  # type: ignore[attr-defined]
        for rec in recs:
            if rec.e_best <= target:
                return rec.n_eval
        return result.n_eval  # type: ignore[attr-defined]

    n_newton = _evals_to_target(newt)
    n_gd = _evals_to_target(gd)
    assert n_gd >= 3 * n_newton
    assert newt.n_eval <= gd.n_eval


def test_g5_certified_stop(make_toy) -> None:  # type: ignore[no-untyped-def]
    cfg = PopulationConfig(size=8, selection_beta=6.0, elitism=2, bit_flip_p=0.35)
    closed = 0
    n_inst = 10
    for i in range(n_inst):
        c = np.array([0.4 + 0.1 * i, 0.7, 1.1, 0.9], dtype=np.float64)
        prob = make_toy(np.zeros((4, 4)), c=c)
        result = certified_discrete_evolve(prob, cfg, generations=16, seed=100 + i, bound="negative_coeff")
        _opt_x, opt = __import__("omnibias.discrete", fromlist=["brute_force_min"]).brute_force_min(prob)
        if result.gap_closed:
            closed += 1
            assert result.best_value <= opt + 1e-9
            assert abs(result.best_value - result.lower_bound) <= 1e-9
        else:
            assert result.best_value + 1e-12 >= opt
        assert result.lower_bound <= opt + 1e-9
    assert closed >= 9


def test_g5_never_closes_incorrectly(make_toy) -> None:  # type: ignore[no-untyped-def]
    from omnibias.discrete import brute_force_min

    cfg = PopulationConfig(size=6, selection_beta=3.0, elitism=1)
    prob = make_toy([[0.0, 0.5], [0.5, 0.0]], c=[0.2, 0.3])
    result = certified_discrete_evolve(prob, cfg, generations=8, seed=3, bound="negative_coeff")
    _x, opt = brute_force_min(prob)
    if result.gap_closed:
        assert result.best_value <= opt + 1e-9


def test_g6_soft_weights_parity() -> None:
    e = np.array([2.0, 2.3, 3.1, 5.0], dtype=np.float64)
    ref = soft_weights(e, beta=1.0)
    torch = pytest.importorskip("torch")
    tw = __import__("omnibias.discrete.evolution.torch", fromlist=["soft_weights"]).soft_weights
    got_t = tw(torch.as_tensor(e, dtype=torch.float64), beta=1.0).detach().cpu().numpy()
    np.testing.assert_allclose(got_t, ref, rtol=0.0, atol=0.0)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jw = __import__("omnibias.discrete.evolution.jax", fromlist=["soft_weights"]).soft_weights
    got_j = np.asarray(jw(jax.numpy.asarray(e, dtype=jax.numpy.float64), beta=1.0))
    np.testing.assert_allclose(got_j, ref, rtol=0.0, atol=2e-16)


def test_evolve_seed_policy_is_deterministic() -> None:
    cfg = PopulationConfig(size=5, selection_beta=2.0, elitism=1)
    rng = np.random.default_rng(1)
    init = rng.normal(size=(5, 2))
    a = soft_population_evolve(named_g2_objective, init, cfg, generations=5, seed=7)
    b = soft_population_evolve(named_g2_objective, init, cfg, generations=5, seed=7)
    np.testing.assert_array_equal(a.best, b.best)
    assert a.best_value == b.best_value


def test_geometry_example_mutation_respects_scales() -> None:
    pack = PackGenes(0.5, 0.01, 2, 1.3)
    rng = np.random.default_rng(0)
    mut = GeometryMutation()
    geo = mutate_geometry(pack, mut, rng)
    assert geo.delta > 0.0
    iso = mutate_isotropic(pack, 0.2, np.random.default_rng(0), mut=mut)
    assert iso.delta > 0.0
    assert pack_fitness(pack, target=pack) == 0.0


def test_terminology_note_is_present() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "discrete" / "evolution"
    for name in ("_core.py", "torch.py", "jax.py", "__init__.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "founding bias collapse" in text
        assert "delta -> 0" in text
        assert "beta -> inf" in text
        assert "feasibility" in text.lower()
        assert "do not conflate" in text.lower()

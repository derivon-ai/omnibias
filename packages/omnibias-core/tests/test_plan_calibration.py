# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fitting a cost model to measurements and scoring dispatch regret."""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.plan_calibration import (
    Measurement,
    cross_validate,
    dispatch_regret,
    fit_cost_model,
    summarize_regret,
)
from omnibias.core.planner import CostModel, Work

TRUE = CostModel(
    device="synthetic",
    rate_gemm=2e12,
    rate_elem=2e11,
    bandwidth=5e11,
    launch_s=2e-5,
    compile_s_per_unit=1.5,
    efficiency={"streams": 0.4, "rows": 1.6, "jet": 1.0},
)


def _work(rng: random.Random, kind: str) -> Work:
    n = rng.uniform(1e6, 1e10)
    if kind == "streams":  # GEMM-heavy
        return Work(
            gemm_flops=n * 4,
            elem_flops=n * 0.05,
            traffic_bytes=n * 0.1,
            launches=10,
            compile_units=2.0,
        )
    if kind == "rows":  # more arithmetic and more launches
        return Work(
            gemm_flops=n * 30,
            elem_flops=n * 0.3,
            traffic_bytes=n * 0.5,
            launches=200,
            compile_units=1.0,
        )
    return Work(
        gemm_flops=n * 2, elem_flops=n * 2, traffic_bytes=n, launches=60, compile_units=3.0
    )  # jet


def _data(seed: int, groups: int, noise: float = 0.05) -> list[Measurement]:
    rng = random.Random(seed)
    out = []
    for g in range(groups):
        scale = rng.uniform(0.2, 5.0)
        for kind in ("streams", "rows", "jet"):
            w = _work(rng, kind)
            w = Work(
                **{
                    k: v * scale
                    for k, v in w.to_config().items()
                    if k != "launches" and k != "compile_units"
                },
                launches=w.launches,
                compile_units=w.compile_units,
            )
            pred = TRUE.predict_work(kind, w)
            out.append(
                Measurement(
                    f"g{g}",
                    kind,
                    w,
                    pred.seconds * math.exp(rng.gauss(0, noise)),
                    "float64",
                    pred.compile_seconds,
                )
            )
    return out


def test_fit_recovers_a_synthetic_device_and_predicts_held_out_problems() -> None:
    train, test = _data(0, 80), _data(1, 40)
    model = fit_cost_model(train, device="synthetic")
    errs = []
    for m in test:
        errs.append(abs(math.log(model.predict_work(m.algorithm, m.work).seconds / m.seconds)))
    assert sorted(errs)[len(errs) // 2] < 0.15
    assert model.compile_s_per_unit == pytest.approx(TRUE.compile_s_per_unit, rel=0.05)
    # the ranking, not only the magnitude, is right
    assert summarize_regret(dispatch_regret(test, model, expected_uses=10**7))["median"] <= 1.05


def test_uncalibrated_defaults_have_more_regret_than_the_fitted_model() -> None:
    train, test = _data(2, 80), _data(3, 60)
    fitted = fit_cost_model(train, device="synthetic")
    base = summarize_regret(dispatch_regret(test, CostModel(device="generic"), expected_uses=10**7))
    new = summarize_regret(dispatch_regret(test, fitted, expected_uses=10**7))
    assert new["mean"] <= base["mean"]
    assert new["worst"] <= max(1.3, base["worst"])


def test_regret_rows_name_the_best_and_the_choice() -> None:
    w_fast, w_slow = Work(gemm_flops=1e9), Work(gemm_flops=1e12)
    ms = [Measurement("a", "x", w_fast, 1e-3), Measurement("a", "y", w_slow, 2e-3)]
    (row,) = dispatch_regret(ms, CostModel())
    assert (row.chosen, row.best, row.regret) == ("x", "x", 1.0)
    swapped = [Measurement("a", "x", w_fast, 4e-3), Measurement("a", "y", w_slow, 1e-3)]
    (row,) = dispatch_regret(swapped, CostModel())
    assert row.chosen == "x" and row.best == "y" and row.regret == pytest.approx(4.0)
    assert summarize_regret([row])["worst"] == pytest.approx(4.0)


def test_expected_uses_trades_compile_time_against_steady_time() -> None:
    cheap_compile = Work(gemm_flops=1e12, compile_units=0.1)
    cheap_run = Work(gemm_flops=2e11, compile_units=50.0)
    ms = [Measurement("a", "a", cheap_compile, 1.0), Measurement("a", "b", cheap_run, 0.2)]
    model = CostModel(rate_gemm=1e12, compile_s_per_unit=2.0)
    assert dispatch_regret(ms, model, expected_uses=1)[0].chosen == "a"  # b compiles for 100 s
    assert dispatch_regret(ms, model, expected_uses=100_000)[0].chosen == "b"


def test_cross_validation_never_trains_on_the_problem_it_scores() -> None:
    data = _data(4, 30)
    rows = cross_validate(data, device="synthetic", folds=5)
    assert len(rows) == 30 and {r.group for r in rows} == {f"g{i}" for i in range(30)}


def test_measurement_and_work_round_trip_through_configs() -> None:
    m = Measurement("g", "a", Work(gemm_flops=3.0, launches=7.0), 1e-3, "float32", 2.5)
    assert Measurement.from_config(m.to_config()) == m
    with pytest.raises(ValueError, match="no usable measurements"):
        fit_cost_model([], device="d")


def test_fit_uses_the_planner_formula_including_solver_sync_and_serial_terms() -> None:
    """The fit and ``CostModel.predict_work`` are one formula: latencies the flop counts cannot show are identified."""
    truth = CostModel(
        device="synthetic",
        rate_gemm=2e12,
        rate_elem=2e11,
        bandwidth=5e11,
        launch_s=2e-5,
        solver_s=4e-4,
        sync_s=3e-4,
        efficiency={"dense": 1.0, "streams": 0.5},
    )
    rng = random.Random(7)
    data = []
    for g in range(90):
        n = rng.uniform(1e7, 1e10)
        kinds = (
            ("dense", Work(gemm_flops=n, launches=20, solver_calls=rng.choice([0, 1, 4]), syncs=rng.choice([0, 1, 2, 3]))),
            ("streams", Work(gemm_flops=2 * n, elem_flops=0.1 * n, traffic_bytes=0.2 * n, launches=6)),
        )
        for name, w in kinds:
            sec = truth.predict_work(name, w).seconds * math.exp(rng.gauss(0, 0.03))
            data.append(Measurement(f"g{g}", name, w, sec))
    fitted = fit_cost_model(data, device="synthetic")
    assert fitted.solver_s == pytest.approx(truth.solver_s, rel=0.35)
    assert fitted.sync_s == pytest.approx(truth.sync_s, rel=0.35)
    for m in data[:40]:  # the fitted model's prediction is the planner's, term for term
        got = fitted.predict_work(m.algorithm, m.work).seconds
        assert 0.5 < got / m.seconds < 2.0


def test_latency_terms_stay_at_their_base_values_when_no_measurement_carries_them() -> None:
    base = CostModel(device="synthetic", solver_s=7e-5, sync_s=2e-4, units=64.0)
    fitted = fit_cost_model(_data(4, 30), device="synthetic", base=base)
    assert (fitted.solver_s, fitted.sync_s, fitted.units) == (7e-5, 2e-4, 64.0)

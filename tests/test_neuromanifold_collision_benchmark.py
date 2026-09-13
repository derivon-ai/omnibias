# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Absolute acceptance gates for the small, reproducible collision experiment."""

import importlib.util
from pathlib import Path


def test_smoke_equal_budgets_conditioning_and_attained_boundary() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "neuromanifold_collisions.py"
    spec = importlib.util.spec_from_file_location("collision_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run()
    assert report["seeds"] == [0, 1]
    assert report["budget"] == {"steps_per_chart_seed": 30, "samples": 33, "parameters_per_chart": 4}
    assert len(report["trials"]) == 4
    for trial in report["trials"]:
        assert trial["steps"] == 30 and trial["samples"] == 33 and trial["trainable_parameters"] == 4
        assert trial["sigmoid_calls"] == (64 if trial["chart"] == "ordinary" else 128)
        assert trial["seconds"] > 0 and trial["python_peak_bytes"] > 0
        assert trial["parameter_gradient_optimizer_bytes"] > 0
    assert report["boundary_demo"]["attained_zero"]
    assert report["boundary_demo"]["derivative_atom_linf"] == 0
    moderate = report["conditioning"][2]
    assert moderate["ordinary"]["condition"] > 1e10
    assert moderate["confluent"]["condition"] < 200
    assert moderate["confluent"]["numerical_rank"] == 4
    near = report["conditioning"][-1]
    assert near["confluent_linf_to_decimal_reference"] < 5e-16
    assert near["ordinary_linf_to_decimal_reference"] > 1e-10

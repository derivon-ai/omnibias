# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-06: exact-Jacobian jet Lohner steps and orbit recovery."""

from __future__ import annotations

import math

import pytest
from omnibias.core.proof.certificate import make_certificate
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import (
    DISCLAIMER,
    WidthBudget,
    coefficient_arithmetic_report,
    honesty_payload,
    horizon_report,
    jacobian_containment_report,
    lohner_flow_jet,
    lohner_regression_snapshot,
    lohner_step_jet,
    naive_vs_lohner_wrapping,
    named_tower_fields,
    orbit_recovery_report,
    run_schema_errors,
    seal_run,
    singularity_step_report,
    tower_field,
    tower_jacobian,
)
from omnibias.core.verified.lohner import (
    LohnerSet,
    constant_jacobian,
    linear_field,
    lohner_flow,
    lohner_step,
)


def test_g1_width_budget_on_every_run() -> None:
    spec = named_tower_fields()[0]
    run = lohner_flow_jet(
        tower_field(spec),
        tower_jacobian(spec),
        [Interval(0.2, 0.3)],
        0.05,
        8,
        order=8,
    )
    assert run.budget.dominant in {"truncation", "jacobian", "wrapping", "rounding"}
    assert run.budget.total >= 0.0
    assert run_schema_errors(run) == []
    sealed = seal_run(run)
    assert "theorem_prover_verified" not in sealed["honesty"]
    assert honesty_payload()["theorem_prover_verified"] is False
    assert "not a continuum existence theorem" in DISCLAIMER
    diagnosis = sealed["payload"]["diagnosis"]
    assert diagnosis["dominant"] == run.budget.dominant
    assert diagnosis["action"] in {"raise_order", "subdivide", "shrink_step", "stop_floor"}
    if diagnosis["dominant"] == "jacobian":
        assert diagnosis["action"] == "shrink_step"
        assert "bias" not in diagnosis["reason"]


def test_g2_jacobian_containment() -> None:
    report = jacobian_containment_report(n_boxes=1000, seed=3)
    assert report["fields"] == 10
    assert report["checked"] == 10000
    assert report["misses"] == 0
    assert report["containment"] is True


def test_g3_horizon_extension() -> None:
    rows = horizon_report()
    assert len(rows) == 10
    assert all("ratio" in row for row in rows)
    wins = sum(1 for row in rows if row["win"])
    assert wins == 10, rows


def test_g4_orbit_recovery() -> None:
    rows = orbit_recovery_report()
    assert len(rows) == 3
    assert all(row["recovered"] for row in rows), rows
    for row in rows:
        assert row["jet_enclosure"] is not None
        lo, hi = row["jet_enclosure"][0]
        assert hi >= lo


def test_g5_singularity_step_soundness() -> None:
    rows = singularity_step_report()
    assert len(rows) == 3
    assert all(row["sound"] for row in rows), rows
    riccati = next(row for row in rows if row["name"] == "riccati")
    assert riccati["step"] <= 1.0
    tan_row = next(row for row in rows if row["name"] == "tan")
    assert tan_row["step"] <= 0.5 * math.pi


def test_g6_lohner_flow_bit_unchanged() -> None:
    snap_a = lohner_regression_snapshot()
    snap_b = lohner_regression_snapshot()
    assert snap_a == snap_b
    a = [[0.0, -1.0], [1.0, 0.0]]
    field = linear_field(a)
    jac = constant_jacobian(a)
    y0 = [Interval(0.999, 1.001), Interval(-0.001, 0.001)]
    classic = lohner_flow(field, jac, y0, 0.05, 20, order=12)
    # The jet wrapper must reproduce the same step as lohner_step.
    state = LohnerSet.from_box(y0)
    jet_state, budget = lohner_step_jet(state, field, jac, order=12, h=0.05)
    plain = lohner_step(field, jac, state, 0.05, 12)
    for left, right in zip(jet_state.to_box(), plain.to_box(), strict=True):
        assert left.lo == right.lo and left.hi == right.hi
    assert isinstance(budget, WidthBudget)
    wrap = naive_vs_lohner_wrapping()
    assert wrap["naive"] > wrap["lohner"]
    # Existing path still encloses the rotation centre.
    box = classic.to_box()
    assert box[0].lo <= math.cos(1.0) <= box[0].hi


def test_coefficient_arithmetic_chooses_interval() -> None:
    report = coefficient_arithmetic_report()
    assert report["choice"] == "interval"
    assert report["tm_required"] is False


def test_seal_refuses_forged_kernel_flag() -> None:
    with pytest.raises(ValueError):
        make_certificate(
            claim="finite-horizon Lohner enclosure of one trajectory",
            payload={"disclaimer": DISCLAIMER},
            honesty={"theorem_prover_verified": True},
        )

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-30: locked catalog through the unified inequality machine."""

from __future__ import annotations

from omnibias.core.proof import Conjecture
from omnibias.core.proof.inequality import (
    INEQUALITY_KIND,
    InequalitySystem,
    build_inequality_machine,
    load_inequality_stack,
    locked_catalog,
)


def test_catalog_statuses_and_replay() -> None:
    loaded = load_inequality_stack()
    assert loaded
    machine = build_inequality_machine()
    assert INEQUALITY_KIND in machine.kinds()
    for row in locked_catalog():
        system = InequalitySystem.from_mapping(row)
        verdict = machine.evaluate(
            Conjecture(system.name, INEQUALITY_KIND, system.as_dict()),
            replay=True,
        )
        assert verdict.status == row["expected"], (
            f"{system.name}: {verdict.status} != {row['expected']} ({verdict.detail})"
        )
        assert verdict.replay_ok is True
        assert verdict.schema_ok is True


def test_forged_complexity_claims_block() -> None:
    machine = build_inequality_machine()
    row = next(item for item in locked_catalog() if item["name"] == "linear_box_sat")
    for key in ("p_equals_np_claim", "new_lp_algorithm_claim", "complete_solver"):
        verdict = machine.evaluate(
            Conjecture(
                f"forged_{key}",
                INEQUALITY_KIND,
                row,
                claims={key: True},
            )
        )
        assert verdict.status == "BLOCKED", key
        assert verdict.honesty_ok is False


def test_pipeline_recorded_on_linear_and_poly() -> None:
    machine = build_inequality_machine()
    for name in ("linear_box_sat", "poly_constant_one"):
        row = next(item for item in locked_catalog() if item["name"] == name)
        verdict = machine.evaluate(
            Conjecture(name, INEQUALITY_KIND, row),
            replay=True,
        )
        assert verdict.status == "PROVED"
        assert verdict.certificate is not None
        pipeline = verdict.certificate["payload"]["pipeline"]
        assert pipeline == ["propose", "rationalize", "check"], name


def test_boolean_skips_propose() -> None:
    machine = build_inequality_machine()
    row = next(item for item in locked_catalog() if item["name"] == "boolean_and_sat")
    verdict = machine.evaluate(Conjecture("bool", INEQUALITY_KIND, row))
    assert verdict.status == "PROVED"
    assert verdict.certificate is not None
    assert verdict.certificate["payload"]["pipeline"] == ["rationalize", "check"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The persisted plan format is versioned: a frozen sample keeps loading, older versions migrate, newer ones are refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omnibias.core import planner
from omnibias.core.planner import PLAN_FORMAT, PLAN_MIGRATIONS, Plan, PlanCache, migrate_plan_config

SAMPLE = Path(__file__).parent / "data" / "plan_format_1.json"


def test_the_frozen_format_1_sample_loads_and_round_trips():
    config = json.loads(SAMPLE.read_text())
    assert config["format"] == 1
    plan = Plan.from_config(config)
    assert plan.algorithm == "projected" and plan.operator == "laplacian"
    assert plan.to_config() == migrate_plan_config(config)
    assert {c["name"] for c in plan.candidates} >= {"projected", "nested_autodiff"}
    assert [r.algorithm for r in plan.refusals] == ["forward_polylaplacian", "hutchinson"]


def test_a_newer_format_is_refused_and_an_unknown_one_is_not_misread():
    config = json.loads(SAMPLE.read_text())
    with pytest.raises(ValueError, match="newer"):
        Plan.from_config({**config, "format": PLAN_FORMAT + 1})
    with pytest.raises(ValueError, match="not an integer"):
        Plan.from_config({**config, "format": "1"})
    with pytest.raises(ValueError, match="no migration"):
        Plan.from_config({**config, "format": 0})


def test_migrations_chain_in_order(monkeypatch):
    """A format-0 shape (``algo`` for ``algorithm``) reaches format 1 through the registered step."""
    config = json.loads(SAMPLE.read_text())
    legacy = {k: v for k, v in config.items() if k != "algorithm"}
    legacy.update(format=0, algo=config["algorithm"])

    def step(c):
        c = dict(c)
        c["algorithm"] = c.pop("algo")
        return c

    monkeypatch.setitem(PLAN_MIGRATIONS, 0, step)
    plan = Plan.from_config(legacy)
    assert plan.algorithm == config["algorithm"] and plan.to_config()["format"] == PLAN_FORMAT


def test_a_cache_file_of_another_format_is_ignored_not_misread(tmp_path):
    config = json.loads(SAMPLE.read_text())
    path = tmp_path / "plans.json"
    path.write_text(json.dumps({"format": PLAN_FORMAT + 1, "plans": {"k": config}}))
    assert len(PlanCache(path)) == 0
    path.write_text(json.dumps({"format": PLAN_FORMAT, "plans": {"k": config}}))
    cache = PlanCache(path)
    assert len(cache) == 1 and cache.get("k") is not None
    path.write_text(json.dumps({"format": PLAN_FORMAT, "plans": {"k": {**config, "format": 99}}}))
    assert PlanCache(path).get("k") is None  # an entry whose own format is unreadable is a miss


def test_the_exports_and_the_policy_comment_agree():
    assert "PLAN_MIGRATIONS" in planner.__all__ and "migrate_plan_config" in planner.__all__
    assert PLAN_FORMAT == 2

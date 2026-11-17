# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``compile_operator(safe_find=True)``: candidates are timed in their own process; a crash is a refusal, a timeout ranks last."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import _safe_find, compile_operator  # noqa: E402
from omnibias.jax.experimental._safe_find import (  # noqa: E402
    SafeFindError,
    time_candidate_in_subprocess,
)


def _layers(D=3, H=8, seed=0):
    g = np.random.default_rng(seed)
    return [
        (
            jnp.asarray(g.standard_normal((H, D)) / np.sqrt(D)),
            jnp.asarray(0.1 * g.standard_normal(H)),
            "tanh",
        ),
        (jnp.asarray(g.standard_normal((1, H)) / np.sqrt(H)), jnp.asarray(np.zeros(1)), None),
    ]


def test_a_candidate_is_timed_in_a_child_process():
    layers, x = _layers(), jnp.asarray(np.random.default_rng(1).standard_normal((4, 3)))
    seconds = time_candidate_in_subprocess(
        [(W, b, s) for W, b, s in layers],
        x,
        "nested_autodiff",
        {"operator": "laplacian"},
        x64=True,
        timeout=300,
        repeats=2,
    )
    assert 0.0 < seconds < 60.0


def test_a_failing_candidate_is_an_error_not_a_crash_of_the_caller():
    layers, x = _layers(), jnp.asarray(np.random.default_rng(1).standard_normal((4, 3)))
    with pytest.raises(SafeFindError, match="exited with status") as info:
        time_candidate_in_subprocess(
            layers, x, "no_such_algorithm", {"operator": "laplacian"}, x64=True, timeout=300
        )
    assert info.value.returncode not in (None, 0)
    with pytest.raises(SafeFindError, match="timed out"):
        time_candidate_in_subprocess(
            layers, x, "nested_autodiff", {"operator": "laplacian"}, x64=True, timeout=0.01
        )


def test_find_refuses_a_crashed_candidate_reports_it_and_keeps_the_plan(monkeypatch, tmp_path):
    from omnibias.core.planner import PlanCache

    layers, x = _layers(), jnp.asarray(np.random.default_rng(1).standard_normal((6, 3)))
    calls = []

    def fake(layers_, x_, name, kwargs, **kw):
        calls.append(name)
        if len(calls) == 1:
            raise SafeFindError(name, "its process exited with status -11: segfault", -11)
        return 1.0 + len(calls)

    monkeypatch.setattr(_safe_find, "time_candidate_in_subprocess", fake)
    path = tmp_path / "plans.json"
    op = compile_operator(layers, x, mode="find", safe_find=True, cache=str(path))
    plan = op.plan
    crashed = calls[0]
    assert plan.algorithm != crashed and crashed not in plan.measured  # a refusal, not a time
    failures = [f for f in op.explain()["executor_failures"] if f["algorithm"] == crashed]
    assert failures and "segfault" in failures[0]["reason"]
    refusals = [r for r in plan.refusals if r.algorithm == crashed]
    assert refusals and refusals[0].reason.startswith("find:") and "segfault" in refusals[0].reason
    # a crash repeats: the plan is cached with the refusal, and the next call times nothing
    assert len(PlanCache(str(path))) == 1
    n = len(calls)
    again = compile_operator(layers, x, mode="find", safe_find=True, cache=str(path))
    assert len(calls) == n and again.algorithm == plan.algorithm
    assert any(r.algorithm == crashed for r in again.plan.refusals)
    np.testing.assert_allclose(
        op(layers, x),
        compile_operator(layers, x, force="nested_autodiff")(layers, x),
        rtol=1e-8,
        atol=1e-10,
    )


def test_a_timed_out_candidate_ranks_last_and_the_plan_is_not_cached(monkeypatch, tmp_path):
    from omnibias.core.planner import PlanCache

    layers, x = _layers(), jnp.asarray(np.random.default_rng(1).standard_normal((6, 3)))
    calls = []

    def fake(layers_, x_, name, kwargs, **kw):
        calls.append(name)
        if len(calls) == 1:
            raise SafeFindError(name, "timed out after 1 s in its own process")  # load, not necessarily the program
        return 1.0 + len(calls)

    monkeypatch.setattr(_safe_find, "time_candidate_in_subprocess", fake)
    path = tmp_path / "plans.json"
    op = compile_operator(layers, x, mode="find", safe_find=True, cache=str(path))
    assert op.plan.algorithm != calls[0] and op.plan.measured[calls[0]] == float("inf")
    assert len(PlanCache(str(path))) == 0


def test_find_with_every_candidate_crashing_raises(monkeypatch, tmp_path):
    layers, x = _layers(), jnp.asarray(np.random.default_rng(1).standard_normal((6, 3)))

    def fake(layers_, x_, name, kwargs, **kw):
        raise SafeFindError(name, "its process exited with status -6", -6)

    monkeypatch.setattr(_safe_find, "time_candidate_in_subprocess", fake)
    with pytest.raises(ValueError, match="every timed candidate failed"):
        compile_operator(layers, x, mode="find", safe_find=True, cache=str(tmp_path / "plans.json"))


def test_safe_find_needs_a_layer_model(tmp_path):
    params = [(jnp.eye(3), jnp.zeros(3))]

    def model(p, x):
        return jnp.sum(jnp.tanh(x @ p[0][0] + p[0][1]), axis=-1)

    with pytest.raises(ValueError, match="layer-list model"):
        compile_operator(
            model, (params, jnp.ones((4, 3))), mode="find", safe_find=True, cache=str(tmp_path / "plans.json")
        )

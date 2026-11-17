# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The precision policy in ``compile_operator`` and ``compile_residual``."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from omnibias.core.planner import PlanCache  # noqa: E402
from omnibias.jax.experimental import Residual, compile_operator, compile_residual  # noqa: E402

D, H = 6, 8


def _params(dtype=jnp.float32, depth=2, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [D] + [H] * depth + [1]
    return [
        (
            jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a), dtype),
            jnp.asarray(rng.standard_normal(b) * 0.1, dtype),
        )
        for a, b in zip(sizes[:-1], sizes[1:], strict=True)
    ]


def _model(params, x):
    a = x
    for W, b in params[:-1]:
        a = jnp.tanh(a @ W + b)
    W, b = params[-1]
    return (a @ W + b)[..., 0]


def _x(dtype=jnp.float32, n=5):
    return jnp.asarray(np.random.default_rng(3).standard_normal((n, D)), dtype)


def test_the_default_policy_is_reported_and_leaves_float32_results_unchanged():
    params, x = _params(), _x()
    op = compile_operator(_model, (params, x), cache=PlanCache())
    entry = op.explain()["precision"]
    assert entry == {"matmul": "highest", "allow_low_precision": False, "notes": []}
    plain = compile_operator(_model, (params, x), cache=PlanCache(), precision="default")
    np.testing.assert_allclose(op(params, x), plain(params, x), rtol=1e-4, atol=1e-5)


def test_tf32_is_an_opt_in_that_is_reported():
    params, x = _params(), _x()
    op = compile_operator(_model, (params, x), cache=PlanCache(), precision="tf32")
    entry = op.explain()["precision"]
    assert entry["matmul"] == "tf32" and any("TF32" in n for n in entry["notes"])


def test_half_precision_is_refused_for_second_order_operators():
    params, x = _params(jnp.bfloat16), _x(jnp.bfloat16)
    with pytest.raises(ValueError, match="refused for derivative order 2"):
        compile_operator(_model, (params, x), cache=PlanCache())
    op = compile_operator(_model, (params, x), cache=PlanCache(), allow_low_precision=True)
    assert any("not bounded" in n for n in op.explain()["precision"]["notes"])


def test_first_order_operators_admit_half_precision_with_a_note():
    params, x = _params(jnp.bfloat16), _x(jnp.bfloat16)
    vec = lambda p, z: jnp.stack([_model(p, z)] * D, axis=-1)  # noqa: E731
    op = compile_operator(vec, (params, x), operator="divergence", cache=PlanCache())
    assert any("order 1" in n for n in op.explain()["precision"]["notes"])


def test_an_unknown_mode_is_an_error():
    params, x = _params(), _x()
    with pytest.raises(ValueError, match="one of"):
        compile_operator(_model, (params, x), cache=PlanCache(), precision="fp8")


def _blocks(x, demand=("grad", "lap")):
    def fn(q):
        return q.lap + q.u if "lap" in demand else q.u

    return [(x, Residual(fn, demand))]


def test_residual_compile_reports_the_policy_and_derives_the_order_from_the_demand():
    params, x = _params(), _x()
    op = compile_residual(_model, _blocks(x), example_params=params, cache=PlanCache())
    assert op.explain()["precision"]["matmul"] == "highest"
    lo = compile_residual(
        _model, _blocks(x, ("value",)), example_params=params, cache=PlanCache(), precision="tf32"
    )
    assert lo.explain()["precision"]["matmul"] == "tf32"
    params16, x16 = _params(jnp.bfloat16), _x(jnp.bfloat16)
    with pytest.raises(ValueError, match="refused for derivative order 2"):
        compile_residual(_model, _blocks(x16), example_params=params16, cache=PlanCache())
    ok = compile_residual(
        _model, _blocks(x16, ("value",)), example_params=params16, cache=PlanCache()
    )
    assert any("order 0" in n for n in ok.explain()["precision"]["notes"])  # reported, not refused
    allowed = compile_residual(
        _model,
        _blocks(x16),
        example_params=params16,
        cache=PlanCache(),
        allow_low_precision=True,
    )
    assert any("not bounded" in n for n in allowed.explain()["precision"]["notes"])


def test_a_forced_residual_program_that_cannot_run_is_an_error():
    params, x = _params(), _x()
    with pytest.raises(ValueError):
        compile_residual(
            _model, _blocks(x), example_params=params, cache=PlanCache(), force="no_such_program"
        )

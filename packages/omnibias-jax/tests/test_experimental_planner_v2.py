# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Planner v2 on the JAX executors: new pool members, training objective, estimators, auto modes, abstract planning."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("OMNIBIAS_PALLAS_INTERPRET", "1")
os.environ.setdefault(
    "OMNIBIAS_FOLX_ENGINE", "off"
)  # these tests pin the built-in pool; folx has its own test

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.planner import AUTO_TRAINING_USES, PlanCache  # noqa: E402
from omnibias.jax.experimental import compile_operator  # noqa: E402


def _init(widths, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (
            jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
            jnp.asarray(0.2 * rng.standard_normal(b)),
        )
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    ]


def _mlp(act):
    def model(params, x):
        a = x
        for W, b in params[:-1]:
            a = act(a @ W + b)
        W, b = params[-1]
        out = a @ W + b
        return out[..., 0] if out.shape[-1] == 1 else out

    return model


TANH = _mlp(jnp.tanh)


def _x(batch=6, dim=3, seed=1):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((batch, dim)))


def _hess_trace(model, params, x, power=1):
    def point(xi):
        return model(params, xi[None])[0]

    g = point
    for _ in range(power):
        g = (lambda f: lambda z: jnp.trace(jax.hessian(f)(z)))(g)
    return jax.vmap(g)(x)


def _op(model, params, x, **kw):
    kw.setdefault("cache", PlanCache())
    return compile_operator(model, (params, x), **kw)


# -------------------------------------------------------------------------------------------- pool members


@pytest.mark.parametrize(
    "name", ["projected", "projected_cholesky", "taylor_closed", "fused_streams"]
)
def test_every_exact_laplacian_program_agrees_with_the_dense_hessian(name):
    params, x = _init([3, 16, 16, 1]), _x()
    ref = _hess_trace(TANH, params, x)
    op = _op(TANH, params, x, force=name)
    assert op.algorithm == name
    np.testing.assert_allclose(np.asarray(op(params, x)), np.asarray(ref), rtol=1e-9, atol=1e-10)


def test_taylor_closed_bilaplacian_and_coordinate_subset():
    params, x = _init([3, 12, 12, 1], seed=2), _x(5, 3, seed=2)
    op = _op(TANH, params, x, power=2, force="taylor_closed")
    ref = _hess_trace(TANH, params, x, power=2)
    np.testing.assert_allclose(np.asarray(op(params, x)), np.asarray(ref), rtol=1e-7, atol=1e-8)
    sub = _op(TANH, params, x, power=2, coordinates=[0, 2], force="taylor_closed")

    def point(xi):
        f = lambda z: TANH(params, z[None])[0]  # noqa: E731

        def lap(z):
            H = jax.hessian(f)(z)
            return H[0, 0] + H[2, 2]

        H2 = jax.hessian(lap)(xi)
        return H2[0, 0] + H2[2, 2]

    np.testing.assert_allclose(
        np.asarray(sub(params, x)), np.asarray(jax.vmap(point)(x)), rtol=1e-7, atol=1e-8
    )


def test_planned_bilaplacian_parameter_gradients_compile_with_block_storage():
    """Differentiating the planned Δ² into the parameters under ``jit`` at ``min(D, H) >= 32`` (block storage).

    The block layout once captured arrays made inside the first trace in its ``custom_jvp`` rule, so a training step
    failed with "No constant handler for type DynamicJaxprTracer" at the first width above 31 while smaller ones passed.
    """
    from omnibias.jax.experimental.projected_bilaplacian import projected_deep_bilaplacian

    dim = 34
    params, x = _init([dim, dim, dim, 1], seed=11), _x(3, dim, seed=11)
    op = _op(TANH, params, x, power=2, force="projected", differentiate="params")

    def planned(p):
        return jnp.sum(op(p, x))

    def direct(p):
        layers = [(W.T, b, "tanh") for W, b in p[:-1]] + [(p[-1][0].T, p[-1][1], None)]
        return jnp.sum(projected_deep_bilaplacian(x, layers)[3])

    compiled = jax.jit(jax.grad(planned))(params)
    again = jax.jit(jax.grad(planned))(params)  # a second trace must not see constants of the first
    expected = jax.grad(direct)(params)
    for (a, b), (c, d), (e, f) in zip(compiled, again, expected, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.asarray(b), np.asarray(f), rtol=1e-9, atol=1e-11)
        np.testing.assert_array_equal(np.asarray(c), np.asarray(a))
        np.testing.assert_array_equal(np.asarray(d), np.asarray(b))


def test_taylor_closed_serves_softplus_without_jax_jet():
    sp = _mlp(jax.nn.softplus)
    params, x = _init([3, 12, 1], seed=4), _x(4, 3, seed=4)
    op = _op(sp, params, x, force="taylor_closed")
    ref = _hess_trace(sp, params, x)
    np.testing.assert_allclose(np.asarray(op(params, x)), np.asarray(ref), rtol=1e-9, atol=1e-10)


def test_fused_gradients_match_the_stream_engine_and_second_order_is_refused():
    params, x = _init([3, 16, 16, 1], seed=5), _x(9, 3, seed=5)
    fused = _op(TANH, params, x, force="fused_streams", differentiate="params")
    streams = _op(TANH, params, x, force="projected", differentiate="params")

    def loss(op):
        return lambda p: jnp.sum(op(p, x) ** 2)

    gf = jax.grad(loss(fused))(params)
    gs = jax.grad(loss(streams))(params)
    for (a, b), (c, d) in zip(gf, gs, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(np.asarray(b), np.asarray(d), rtol=1e-8, atol=1e-10)
    with pytest.raises(NotImplementedError, match="first order"):
        jax.jvp(jax.grad(loss(fused)), (params,), (params,))
    # the planner knows: a second derivative into the parameters refuses the fused kernels up front
    both = _op(TANH, params, x, differentiate="params2")
    assert "fused_streams" not in [c["name"] for c in both.explain()["candidates"]]


def test_fused_streams_is_refused_on_a_cpu_device_class(monkeypatch):
    monkeypatch.delenv(
        "OMNIBIAS_PALLAS_INTERPRET"
    )  # the interpreter switch keeps accelerator kernels in the pool
    params, x = _init([3, 16, 1]), _x()
    op = _op(TANH, params, x)  # device_class('cpu') -> an accelerator program is not a candidate
    explained = op.explain()
    assert any(
        r["algorithm"] == "fused_streams" and "accelerator" in r["reason"]
        for r in explained["refused"]
    )


# ----------------------------------------------------------------------------------------------- estimators


@pytest.mark.parametrize("name", ["hutchinson", "stde", "sdgd"])
def test_estimators_are_unbiased_and_sized_by_accuracy(name):
    D = 6
    params, x = _init([D, 10, 1], seed=6), _x(3, D, seed=6)
    ref = np.asarray(_hess_trace(TANH, params, x))
    accuracy = (
        0.9 if name == "sdgd" else 0.3
    )  # sdgd needs fewer than all D coordinates to be worth offering
    op = _op(TANH, params, x, exact=False, force=name, accuracy=accuracy)
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    draws = np.stack([np.asarray(op(params, x, key=k)) for k in keys])
    err = draws.std(axis=0, ddof=1) / np.sqrt(len(keys))
    assert np.all(
        np.abs(draws.mean(axis=0) - ref) <= 5.0 * err + 1e-9
    )  # unbiased to five standard errors
    assert np.all(draws.std(axis=0) > 0)  # and genuinely random


def test_hutchinson_needs_a_key_in_training_and_draws_independent_probes_per_point():
    params = _init([4, 8, 1], seed=7)
    x = jnp.tile(_x(1, 4, seed=7), (6, 1))  # six identical points
    train = _op(TANH, params, x, exact=False, force="hutchinson", differentiate="params")
    with pytest.raises(ValueError, match="PRNG key"):
        train(params, x)
    out = np.asarray(train(params, x, key=jax.random.PRNGKey(3)))
    assert np.ptp(out) > 0  # identical points, independent probes: different estimates
    forward = _op(TANH, params, x, exact=False, force="hutchinson")
    a = np.asarray(forward(params, x))  # a forward evaluation keeps its fixed default seed
    np.testing.assert_array_equal(a, np.asarray(forward(params, x)))


def test_accuracy_target_sets_the_probe_count_the_plan_priced():
    params, x = _init([4, 8, 1]), _x(3, 4)
    loose = _op(TANH, params, x, exact=False, accuracy=0.8, force="stde")
    assert loose.explain()["request"]["accuracy"] == 0.8


# ---------------------------------------------------------------------------- objective, auto modes, abstract


def test_training_objective_is_planned_timed_and_recorded(tmp_path):
    params, x = _init([3, 16, 16, 1]), _x(32, 3)
    op = _op(
        TANH,
        params,
        x,
        differentiate="params",
        mode="auto",
        expected_uses="auto",
        cache=str(tmp_path / "plans.json"),
    )
    info = op.explain()
    assert info["objective"] == "params" and info["mode"] == "find"
    assert info["uses_source"] == "auto" and info["expected_uses"] == AUTO_TRAINING_USES
    assert info["measured_seconds"]  # value-and-gradient was timed for the leading candidates
    again = _op(
        TANH,
        params,
        x,
        differentiate="params",
        mode="auto",
        expected_uses="auto",
        cache=str(tmp_path / "plans.json"),
    )
    assert again.algorithm == op.algorithm  # the persistent plan cache answers


def test_auto_mode_defaults_to_the_persistent_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIBIAS_PLAN_CACHE", str(tmp_path / "auto.json"))
    params, x = _init([3, 8, 1]), _x(8, 3)
    compile_operator(TANH, (params, x), differentiate="params", mode="auto", expected_uses="auto")
    assert (tmp_path / "auto.json").exists()


def test_training_hint_lowers_the_uses_and_keeps_the_heuristic():
    params, x = _init([3, 8, 1]), _x(8, 3)
    op = _op(
        TANH, params, x, differentiate="params", mode="auto", expected_uses="auto", uses_hint=5
    )
    info = op.explain()
    assert (
        info["mode"] == "heuristic" and info["uses_source"] == "hint" and info["expected_uses"] == 5
    )


def test_planning_from_shape_dtype_structs_and_find_refuses_them():
    params, x = _init([3, 16, 1]), _x(32, 3)
    sds = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), (params, x))
    op = compile_operator(TANH, sds, cache=PlanCache())
    concrete = _op(TANH, params, x)
    assert op.algorithm == concrete.algorithm
    np.testing.assert_allclose(
        np.asarray(op(params, x)), np.asarray(concrete(params, x)), rtol=1e-12
    )
    with pytest.raises(ValueError, match="concrete arrays"):
        compile_operator(TANH, sds, mode="find", cache=PlanCache())
    auto = compile_operator(TANH, sds, mode="auto", expected_uses=10**6, cache=PlanCache())
    assert auto.explain()["mode"] == "heuristic"  # nothing to time: auto falls back to the model


def test_training_objective_needs_explicit_parameters():
    params, x = _init([3, 8, 1]), _x(4, 3)
    with pytest.raises(ValueError, match="explicit parameters"):
        compile_operator(
            lambda z: TANH(params, z), (x,), params_argnum=None, differentiate="params"
        )


def test_folx_is_registered_as_an_engine_when_installed(monkeypatch):
    pytest.importorskip("folx")
    from omnibias.jax.experimental import compiler as c

    monkeypatch.delenv("OMNIBIAS_FOLX_ENGINE", raising=False)
    monkeypatch.setattr(c, "_DEFAULTS_DONE", False)
    params, x = _init([3, 8, 1]), _x(5, 3)
    try:
        op = _op(TANH, params, x)
        assert "folx" in [cand["name"] for cand in op.explain()["candidates"]]
        forced = _op(TANH, params, x, force="folx")
        np.testing.assert_allclose(
            np.asarray(forced(params, x)), np.asarray(_hess_trace(TANH, params, x)), rtol=1e-9
        )
    finally:
        if ("laplacian", "folx") in c._EXECUTORS:
            c.unregister_engine("folx", "laplacian")
        monkeypatch.setattr(c, "_DEFAULTS_DONE", False)


def test_an_operator_planned_for_parameter_gradients_traces_under_the_training_layout(monkeypatch):
    """On an accelerator the split stream layout took 0.5 to 0.7 of the stack's time in the value-and-gradient pass, and the
    stack stayed the layout of a forward call: ``differentiate="params"`` is what tells the operator which pass it serves, so its
    programs (the ones ``find`` times included) are traced under :func:`training_layout`; a forward operator keeps the stack."""
    from omnibias.jax.experimental import streams

    monkeypatch.setattr(streams.jax, "default_backend", lambda: "gpu")
    params = _init([3, 8, 8, 1])
    x = _x()
    seen: list[str] = []
    real = streams.resolve_layout

    def spy(layout=None, *arrays, hidden=None, dtype=None):  # operands, depth and dtype are ignored: only the training context may decide here
        out = real(layout)
        seen.append(out)
        return out

    monkeypatch.setattr(streams, "resolve_layout", spy)
    forward = _op(TANH, params, x, force="projected")
    forward(params, x)
    assert seen and set(seen) == {"stack"}
    seen.clear()
    train = _op(TANH, params, x, force="projected", differentiate="params")
    seen.clear()  # construction probes the output shapes outside the training context

    def loss(p):
        value = train(p, x)
        return jnp.sum(value * value)

    g = jax.grad(loss)(params)
    assert seen and set(seen) == {"split"}
    assert real() == "stack"  # the context does not leak out of the trace
    ref = jax.grad(lambda p: jnp.sum(_hess_trace(TANH, p, x) ** 2))(params)
    for (gw, gb), (rw, rb) in zip(g, ref, strict=True):
        np.testing.assert_allclose(gw, rw, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(gb, rb, rtol=1e-8, atol=1e-10)

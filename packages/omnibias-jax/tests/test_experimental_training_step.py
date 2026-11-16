# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The compiled training step: equivalence with an eager loop, grouping, ensembles, planning."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.experimental.training_step import (
    StepBlock,
    adam,
    compile_training_step,
    enable_compile_cache,
    planned_evaluator,
    sgd,
)

jax.config.update("jax_enable_x64", True)

D, H = 2, 12


def _params(seed=0, scale=1.0):
    g = np.random.default_rng(seed)
    dims = [D, H, H, 1]
    return [
        (
            jnp.asarray(scale * g.standard_normal((a, b)) / np.sqrt(a)),
            jnp.asarray(0.1 * g.standard_normal(b)),
        )
        for a, b in zip(dims[:-1], dims[1:], strict=True)
    ]


def _u(p, x):
    a = x
    for W, b in p[:-1]:
        a = jnp.tanh(a @ W + b)
    W, b = p[-1]
    return (a @ W + b)[..., 0]


def _lap(p, x):
    def point(z):
        return jnp.trace(jax.hessian(lambda y: _u(p, y))(z))

    return jax.vmap(point)(x)


class _Counter:
    """An ``evaluate`` that counts the traces that reach it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, p, x):
        self.calls += 1
        return {"u": _u(p, x), "lap": _lap(p, x)}


def _blocks(ev, n_in=24, n_bd=16):
    def s_in(k):
        return jax.random.uniform(k, (n_in, D), minval=-1.0, maxval=1.0, dtype=jnp.float64)

    def s_bd(k):
        t = jax.random.uniform(k, (n_bd,), minval=-1.0, maxval=1.0, dtype=jnp.float64)
        return jnp.stack([t, jnp.ones_like(t)], axis=-1)

    def r_in(p, x, out, aux):
        return out["lap"] + 2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1])

    def r_bd(p, x, out, aux):
        return out["u"]

    return [
        StepBlock("interior", s_in, r_in, evaluate=ev, order=2, weight=1.0),
        StepBlock("boundary", s_bd, r_bd, evaluate=ev, order=2, weight=10.0),
    ]


def _eager_reference(blocks, params, key, n_blocks, K, lr):
    """Same keys, same points, same loss, written by hand with one launch per block."""
    opt = adam(lr)
    st = opt.init(params)
    step = 0
    nb = len(blocks)
    key, k0 = (lambda k: (k[0], k[1]))(jax.random.split(key))
    pending = k0
    losses = []
    for _ in range(n_blocks):
        flat = jax.random.split(pending, K * nb)
        keys = flat.reshape(K, nb, *flat.shape[1:])
        key, pending = (lambda k: (k[0], k[1]))(jax.random.split(key))
        for s in range(K):
            pts = [b.sample(keys[s, i]) for i, b in enumerate(blocks)]

            def loss(p):
                tot = 0.0
                for b, x in zip(blocks, pts, strict=True):  # noqa: B023
                    out = {"u": _u(p, x), "lap": _lap(p, x)}
                    tot = tot + b.weight * jnp.mean(b.residual(p, x, out, None) ** 2)
                return tot

            value, g = jax.value_and_grad(loss)(params)
            up, st = opt.update(g, st, params)
            params = jax.tree.map(lambda a, b: a + b, params, up)
            losses.append(float(value))
            step += 1
    return params, np.asarray(losses)


def test_scan_matches_an_eager_loop_exactly():
    blocks = _blocks(_Counter())
    ts = compile_training_step(blocks, adam(1e-2), steps_per_call=4)
    p0 = _params()
    state = ts.init(p0, jax.random.PRNGKey(7))
    state, _, hist = ts.run(state, 3)
    losses = np.concatenate([np.asarray(m["loss"]) for m in hist])
    ref_params, ref_losses = _eager_reference(blocks, p0, jax.random.PRNGKey(7), 3, 4, 1e-2)
    np.testing.assert_allclose(losses, ref_losses, rtol=1e-9)
    for (W, b), (Wr, br) in zip(state.params, ref_params, strict=True):
        np.testing.assert_allclose(W, Wr, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(b, br, rtol=1e-9, atol=1e-12)
    assert int(state.step) == 12


def test_training_reduces_the_loss():
    ts = compile_training_step(_blocks(_Counter()), adam(3e-3), steps_per_call=25)
    state = ts.init(_params(), 0)
    state, first = ts(state)
    state, _, _ = ts.run(state, 8)
    state, last = ts(state)
    assert float(last["loss"][-1]) < 0.5 * float(first["loss"][0])
    assert set(last) == {"loss", "blocks", "grad_norm"}
    assert last["blocks"].shape == (25, 2)


def test_blocks_sharing_an_evaluate_are_one_launch():
    shared = _Counter()
    ts = compile_training_step(_blocks(shared), steps_per_call=2)
    state = ts.init(_params(), 0)
    ts(state)
    assert shared.calls == 1  # one traced evaluation for both blocks
    groups = ts.explain()["groups"]
    assert (
        len(groups) == 1
        and groups[0]["blocks"] == ["interior", "boundary"]
        and groups[0]["shared_evaluate"]
    )

    a, b = _Counter(), _Counter()
    bl = _blocks(a)
    bl[1] = StepBlock(bl[1].name, bl[1].sample, bl[1].residual, evaluate=b, order=2, weight=10.0)
    ts2 = compile_training_step(bl, steps_per_call=2)
    ts2(ts2.init(_params(), 0))
    assert a.calls == 1 and b.calls == 1 and len(ts2.explain()["groups"]) == 2


def test_prepare_runs_with_the_sampler_and_reaches_the_residual():
    seen = {}

    def residual(p, x, out, aux):
        seen["aux_shape"] = np.shape(aux)
        return _u(p, x) - aux

    blk = StepBlock(
        "fit",
        lambda k: jax.random.normal(k, (8, D), dtype=jnp.float64),
        residual,
        prepare=lambda x: jnp.sin(x[:, 0]),
    )
    ts = compile_training_step([blk], sgd(1e-2), steps_per_call=3)
    ts(ts.init(_params(), 1))
    assert seen["aux_shape"] == (8,)


def test_ensemble_matches_independent_runs():
    n = 3
    keys = jax.random.split(jax.random.PRNGKey(5), n)
    singles = [_params(seed=i) for i in range(n)]
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *singles)
    blocks = _blocks(_Counter())
    ens = compile_training_step(blocks, adam(1e-2), steps_per_call=3, n_models=n)
    st = ens.init(stacked, jax.random.PRNGKey(5))
    st, _, hist = ens.run(st, 2)
    ens_losses = np.concatenate([np.asarray(m["loss"]) for m in hist], axis=1)  # (n, 6)
    one = compile_training_step(blocks, adam(1e-2), steps_per_call=3)
    for j in range(n):
        s = one.init(singles[j], keys[j])
        s, _, h = one.run(s, 2)
        np.testing.assert_allclose(
            ens_losses[j], np.concatenate([np.asarray(m["loss"]) for m in h]), rtol=1e-9
        )
        for (W, _b), (We, _be) in zip(s.params, st.params, strict=True):
            np.testing.assert_allclose(W, We[j], rtol=1e-9, atol=1e-12)
    assert ens_losses.shape == (n, 6)
    assert not np.allclose(ens_losses[0], ens_losses[1])


def test_ensemble_validates_the_model_axis():
    ts = compile_training_step(_blocks(_Counter()), n_models=2)
    with pytest.raises(ValueError, match="leading model axis"):
        ts.init(_params(), 0)


def test_eager_step_and_clipping():
    ts = compile_training_step(_blocks(_Counter()), sgd(1e-2), steps_per_call=4, clip_norm=0.5)
    state = ts.init(_params(), 2)
    new, m = ts.step(state)
    assert int(new.step) == 1 and float(m["loss"]) > 0
    # the update is bounded by lr * clip
    for (W0, _), (W1, _) in zip(_params(), new.params, strict=True):
        assert float(jnp.max(jnp.abs(W1 - W0))) <= 1e-2 * 0.5 + 1e-12


def test_project_constraint_is_applied():
    ts = compile_training_step(
        _blocks(_Counter()),
        sgd(1e-1),
        steps_per_call=2,
        project=lambda p: jax.tree.map(lambda a: jnp.clip(a, -0.05, 0.05), p),
    )
    state = ts.init(_params(), 0)
    state, _ = ts(state)
    assert max(float(jnp.max(jnp.abs(W))) for W, _ in state.params) <= 0.05 + 1e-12


def test_block_name_and_size_validation():
    b = _blocks(None)[0]
    with pytest.raises(ValueError, match="unique"):
        compile_training_step([b, b])
    with pytest.raises(ValueError, match="at least one"):
        compile_training_step([])
    with pytest.raises(ValueError, match="steps_per_call"):
        compile_training_step([b], steps_per_call=0)


def test_planned_evaluator_defaults_to_the_exhaustive_find_for_training():
    from omnibias.core.planner import PlanCache

    p = _params()
    x = jnp.zeros((24, D), jnp.float64)
    ev = planned_evaluator(_u, p, x, steps=200, cache=PlanCache())
    info = ev.explain()
    assert info["mode"] == "find" and info["objective"] == "params"
    assert len(info["measured_seconds"]) > 2  # not only the two the cost model ranks first
    cheaper = planned_evaluator(_u, p, x, steps=200, cache=PlanCache(), mode="auto", top_k=2).explain()
    assert cheaper["mode"] == "find" and len(cheaper["measured_seconds"]) == 2
    heuristic = planned_evaluator(_u, p, x, steps=200, cache=PlanCache(), mode="heuristic").explain()
    assert heuristic["mode"] == "heuristic" and not heuristic["measured_seconds"]


def test_planned_evaluator_serves_the_step():
    p = _params()
    layers = [(W.T, bb, "tanh" if i < len(p) - 1 else None) for i, (W, bb) in enumerate(p)]
    x = jnp.zeros((24, D), jnp.float64)
    ev = planned_evaluator(_u, p, x, steps=200)  # model callable (params, x) -> (B,)
    out = ev(p, x + 0.3)
    np.testing.assert_allclose(out, _lap(p, x + 0.3), rtol=1e-8, atol=1e-9)

    def r_in(pp, xx, lap, aux):
        return lap + 1.0

    blk = StepBlock(
        "pde",
        lambda k: jax.random.uniform(k, (24, D), dtype=jnp.float64),
        r_in,
        evaluate=ev,
        order=2,
    )
    ts = compile_training_step([blk], adam(1e-2), steps_per_call=3)
    state, m = ts(ts.init(p, 0))
    assert np.isfinite(np.asarray(m["loss"])).all()
    del layers


def test_compile_cache_helper_is_explicit(tmp_path):
    before = jax.config.jax_compilation_cache_dir
    try:
        enable_compile_cache(str(tmp_path / "xla"))
        assert jax.config.jax_compilation_cache_dir == str(tmp_path / "xla")
    finally:
        jax.config.update("jax_compilation_cache_dir", before)


def test_run_does_not_synchronise_and_reports_blocks():
    seen = []
    ts = compile_training_step(_blocks(_Counter()), steps_per_call=2)
    state, last, hist = ts.run(ts.init(_params(), 0), 3, on_block=lambda j, s, m: seen.append(j))
    assert seen == [0, 1, 2] and len(hist) == 3
    assert ts.time_steps(state, n_blocks=2)["seconds_per_step"] > 0


def test_step_draws_only_the_first_batch_of_the_stream_and_run_keeps_the_key_chain():
    K = 5
    blocks = _blocks(_Counter())
    ts = compile_training_step(blocks, sgd(1e-2), steps_per_call=K)
    p0 = _params()
    state = ts.init(p0, jax.random.PRNGKey(3))
    new, _ = ts.step(state)

    # the first of the K batches of the key the host loop would have drawn
    key, kn = jax.random.split(jax.random.PRNGKey(3))
    flat = jax.random.split(kn, K * len(blocks)).reshape(K, len(blocks), 2)
    pts = [b.sample(flat[0, i]) for i, b in enumerate(blocks)]

    def loss(p):
        return sum(
            b.weight * jnp.mean(b.residual(p, x, {"u": _u(p, x), "lap": _lap(p, x)}, None) ** 2)
            for b, x in zip(blocks, pts, strict=True)
        )

    g = jax.grad(loss)(p0)
    for (W, b), (Wp, bp), (gW, gb) in zip(new.params, p0, g, strict=True):
        np.testing.assert_allclose(W, Wp - 1e-2 * gW, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(b, bp - 1e-2 * gb, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(new.key, key)

    # run: the key after n blocks is the chain of n + 1 splits, however they are issued
    state, _, _ = ts.run(ts.init(p0, jax.random.PRNGKey(3)), 4)
    k = jax.random.PRNGKey(3)
    for _ in range(5):
        k, _ = jax.random.split(k)
    np.testing.assert_array_equal(state.key, k)


def test_step_program_samples_one_batch_where_a_block_samples_k():
    blk = StepBlock("fit", lambda k: jax.random.normal(k, (4, D), dtype=jnp.float64), lambda p, x, out, aux: _u(p, x))
    ts = compile_training_step([blk], sgd(1e-2), steps_per_call=50)
    key = jax.random.PRNGKey(0)
    assert jax.eval_shape(ts._first, key)[0][0].shape == (50, 4, D)
    assert jax.eval_shape(ts._first_one, key)[0][0].shape == (4, D)
    block_points = ts._first(key)[0][0]
    np.testing.assert_array_equal(ts._first_one(key)[0][0], block_points[0])


def test_compiled_training_step_traces_its_evaluate_under_the_training_layout(monkeypatch):
    from omnibias.jax.experimental import streams

    monkeypatch.setattr(streams.jax, "default_backend", lambda: "gpu")  # an accelerator takes the stack outside training
    seen = []

    def ev(p, x):
        seen.append(streams.resolve_layout())
        return {"u": _u(p, x), "lap": _lap(p, x)}

    assert streams.resolve_layout() == "stack"
    ts = compile_training_step(_blocks(ev), adam(1e-3), steps_per_call=2)
    state = ts.init(_params(), 0)
    ts.run(state, 1, block_until_ready=True)
    assert seen and set(seen) == {"split"}
    assert streams.resolve_layout() == "stack"  # the context does not leak out of the trace

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cold-cost levers: shape buckets, staged compilation, compile policies, the persistent cache and export."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import (  # noqa: E402
    Residual,
    StagedProgram,
    bucket_size,
    compile_operator,
    compile_residual,
    enable_compilation_cache,
    masked_residual,
    pad_block,
)
from omnibias.jax.experimental.coldcost import (  # noqa: E402
    BUCKET_POLICIES,
    COMPILE_POLICIES,
    ExportedProgram,
    compilation_cache_dir,
    pad_rows,
    resolve_compile_policy,
)

D = 5


def _init(widths, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (
            jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
            jnp.asarray(0.2 * rng.standard_normal(b)),
        )
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    ]


def mlp(params, x):
    a = x
    for W, b in params[:-1]:
        a = jnp.tanh(a @ W + b)
    W, b = params[-1]
    return (a @ W + b)[..., 0]


def _x(n, seed=1):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((n, D)))


# ----------------------------------------------------------------------------------------------------- buckets


@pytest.mark.parametrize("policy", ["pow2", "geometric", 8, None, "none"])
def test_bucket_holds_the_rows_and_never_shrinks(policy):
    sizes = [bucket_size(n, policy) for n in range(1, 200)]
    assert all(s >= n for s, n in zip(sizes, range(1, 200), strict=True))
    assert sizes == sorted(sizes)  # monotone
    if policy not in (None, "none"):
        assert len(set(sizes)) < 199


def test_bucket_values():
    assert [bucket_size(n, "pow2") for n in (1, 2, 3, 5, 8, 9, 100)] == [1, 2, 4, 8, 8, 16, 128]
    assert [bucket_size(n, "geometric") for n in (1, 2, 3, 4, 5, 6, 7, 9, 13, 17, 25)] == [
        1,
        2,
        3,
        4,
        6,
        6,
        8,
        12,
        16,
        24,
        32,
    ]
    assert bucket_size(10, 8) == 16
    assert bucket_size(7, None) == 7
    # padding stays below 50 percent for the geometric policy
    assert all(bucket_size(n, "geometric") < 1.5 * n + 1e-9 for n in range(1, 500))
    assert set(BUCKET_POLICIES) == {"none", "pow2", "geometric"}


@pytest.mark.parametrize("bad", ["fibonacci", True, 0])
def test_bucket_rejects_unknown_policy(bad):
    with pytest.raises((ValueError, TypeError)):
        bucket_size(5, bad)
    with pytest.raises(ValueError):
        bucket_size(0, "pow2")


def test_pad_rows_repeats_last_row_and_refuses_to_shrink():
    a = jnp.arange(6.0).reshape(3, 2)
    out = pad_rows(a, 5)
    assert out.shape == (5, 2)
    np.testing.assert_array_equal(np.asarray(out[3:]), np.asarray(jnp.repeat(a[-1:], 2, axis=0)))
    assert pad_rows(a, 3) is a or np.array_equal(pad_rows(a, 3), a)
    with pytest.raises(ValueError):
        pad_rows(a, 2)


def test_bucketed_operator_equals_unbucketed_and_shares_an_executable():
    params = _init([D, 12, 12, 1])
    plain = compile_operator(mlp, (params, _x(7)))
    bucketed = compile_operator(mlp, (params, _x(7)), bucket="pow2")
    for n in (5, 6, 7, 8):
        x = _x(n, seed=n)
        res = bucketed.evaluate(params, x)
        ref = plain.evaluate(params, x)
        assert res.value.shape == (n,)
        np.testing.assert_allclose(res.value, ref.value, rtol=1e-12, atol=1e-12)
        for k, v in ref.aux.items():
            assert res.aux[k].shape == v.shape
            np.testing.assert_allclose(res.aux[k], v, rtol=1e-12, atol=1e-12)
    # 5, 6, 7 and 8 all round to 8 rows: one executable
    assert bucketed.explain()["cold_cost"]["executables"] == 1
    assert plain.explain()["cold_cost"]["executables"] == 4
    bucketed(params, _x(9))  # 16 rows: a second bucket
    assert bucketed.explain()["cold_cost"]["executables"] == 2


def test_bucketed_operator_gradients_ignore_padding():
    params = _init([D, 10, 1])
    bucketed = compile_operator(mlp, (params, _x(6)), bucket="pow2")
    plain = compile_operator(mlp, (params, _x(6)))
    x = _x(5)

    def loss(op):
        return lambda p: jnp.sum(op(p, x) ** 2)

    g1 = jax.grad(loss(bucketed))(params)
    g2 = jax.grad(loss(plain))(params)
    for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_bucketed_solve_pads_the_right_hand_side():
    params = _init([D, 8, D])

    def field(p, x):
        a = x
        for W, b in p[:-1]:
            a = jnp.tanh(a @ W + b)
        W, b = p[-1]
        return 0.1 * (a @ W + b)

    x = _x(5)
    rhs = _x(5, seed=9)
    bucketed = compile_operator(field, (params, x), operator="solve", bucket="pow2")
    plain = compile_operator(field, (params, x), operator="solve")
    np.testing.assert_allclose(
        bucketed(params, x, rhs), plain(params, x, rhs), rtol=1e-10, atol=1e-12
    )


# ----------------------------------------------------------------------------------------------- staged stages


def test_staged_program_reports_each_stage_and_reuses_executables():
    fn = StagedProgram(lambda a, b: (a @ b).sum(0))
    a, b = jnp.ones((4, 3)), jnp.ones((3, 2))
    st = fn.compile(a, b)
    assert fn.signatures() == 1
    assert st.trace_seconds >= 0 and st.lower_seconds >= 0 and st.compile_seconds > 0
    assert st.first_run_seconds is None  # nothing ran yet
    np.testing.assert_allclose(fn(a, b), jnp.full(2, 12.0))
    assert fn.stages[0].first_run_seconds is not None
    fn(a, b)
    assert fn.signatures() == 1
    fn(jnp.ones((5, 3)), b)  # new shape: new executable
    assert fn.signatures() == 2
    assert [s["signature"] for s in fn.report()] == [s.signature for s in fn.stages]
    assert "cold_seconds" in fn.report()[0]


def test_staged_program_distinguishes_python_scalars_from_arrays():
    fn = StagedProgram(lambda x, d: x * d)
    np.testing.assert_allclose(fn(jnp.ones(3), 0.5), 0.5)
    np.testing.assert_allclose(fn(jnp.ones(3), 0.25), 0.25)  # same weak-typed signature: reused
    assert fn.signatures() == 1
    np.testing.assert_allclose(fn(jnp.ones(3), jnp.asarray(2.0, dtype=jnp.float32)), 2.0)
    assert fn.signatures() == 2


def test_staged_program_falls_back_when_the_backend_rejects_an_option():
    fn = StagedProgram(lambda x: x + 1, options={"xla_no_such_option_for_omnibias": True})
    out = fn(jnp.ones(2))
    np.testing.assert_allclose(out, 2.0)
    st = fn.stages[0]
    assert st.options_error is not None  # recorded, not raised
    assert dict(st.options) == {}


def test_operator_warmup_moves_the_cost_out_of_the_first_call():
    params = _init([D, 8, 1])
    x = _x(6)
    op = compile_operator(mlp, (params, x))
    stages = op.warmup(params, x)
    assert stages["compile_seconds"] > 0
    ex = op.explain()["cold_cost"]
    assert ex["executables"] == 1 and ex["stages"][0]["first_run_seconds"] is None
    op(params, x)
    assert op.explain()["cold_cost"]["stages"][0]["first_run_seconds"] is not None
    assert op.explain()["cold_cost"]["predicted_seconds"] is not None
    with pytest.raises(ValueError):
        compile_operator(mlp, (params, x), jit=False).warmup(params, x)


def test_explain_reports_the_policy_and_cache():
    params = _init([D, 8, 1])
    x = _x(4)
    ex = compile_operator(mlp, (params, x), compile_policy="fast").explain()["cold_cost"]
    assert ex["compile_policy"]["requested"] == "fast"
    assert ex["compile_policy"]["applied"] is False  # CPU: accelerator options are not applied
    assert "cpu" in ex["compile_policy"]["reason"].lower()
    assert "persistent_cache_dir" in ex


# ------------------------------------------------------------------------------------------------ policies


def test_policy_resolution():
    cpu = resolve_compile_policy(
        "fast", platform="cpu", predicted_seconds=1.0, cold_seconds=10.0, expected_uses=1
    )
    assert not cpu.applied and dict(cpu.options) == {}
    gpu = resolve_compile_policy(
        "fast", platform="gpu", predicted_seconds=1.0, cold_seconds=10.0, expected_uses=1
    )
    assert gpu.applied and dict(gpu.options) == COMPILE_POLICIES["fast"]
    # auto: a single use of a program that compiles for 10 s and runs for 10 ms takes the fast policy ...
    short = resolve_compile_policy(
        "auto", platform="gpu", predicted_seconds=0.01, cold_seconds=10.0, expected_uses=1
    )
    assert short.name == "fast" and short.applied
    # ... a long run of the same program does not (0.15 * 1e6 * 0.01 s of steady-state cost > 7.2 s saved)
    long_run = resolve_compile_policy(
        "auto", platform="gpu", predicted_seconds=0.01, cold_seconds=10.0, expected_uses=1_000_000
    )
    assert long_run.name == "default" and not long_run.applied
    # auto without a prediction or without an accelerator keeps the default
    assert (
        resolve_compile_policy(
            "auto", platform="gpu", predicted_seconds=None, cold_seconds=None, expected_uses=1
        ).name
        == "default"
    )
    assert (
        resolve_compile_policy(
            "auto", platform="cpu", predicted_seconds=0.01, cold_seconds=10.0, expected_uses=1
        ).name
        == "default"
    )
    with pytest.raises(ValueError):
        resolve_compile_policy(
            "turbo", platform="gpu", predicted_seconds=1.0, cold_seconds=1.0, expected_uses=1
        )
    assert COMPILE_POLICIES["default"] == {}


# ------------------------------------------------------------------------------------------ persistent cache


def test_persistent_cache_is_explicit_and_restorable(tmp_path):
    before = compilation_cache_dir()
    prev = enable_compilation_cache(str(tmp_path), min_compile_seconds=0.0)
    try:
        assert compilation_cache_dir() == str(tmp_path)
        op = compile_operator(mlp, (_init([D, 6, 1]), _x(3)))
        assert op.explain()["cold_cost"]["persistent_cache_dir"] == str(tmp_path)
    finally:
        for k, v in prev.items():
            jax.config.update(k, v)
    assert compilation_cache_dir() == before


# ---------------------------------------------------------------------------------------------------- export


def test_export_round_trip_in_process():
    params = _init([D, 8, 1])
    x = _x(4)
    op = compile_operator(mlp, (params, x))
    program = op.export(params, x)
    assert isinstance(program, ExportedProgram)
    assert program.metadata["selected_strategy"] == op.algorithm
    p, xx, extra = op.prepared(params, x)
    value, aux = program.call(p, xx, extra, None)
    np.testing.assert_allclose(value, op(params, x), rtol=1e-12, atol=1e-12)
    assert "value" in aux or isinstance(aux, dict)


def test_export_serialises_when_flatbuffers_is_present():
    pytest.importorskip("flatbuffers")
    params = _init([D, 8, 1])
    x = _x(4)
    op = compile_operator(mlp, (params, x))
    program = op.export(params, x)
    blob = program.serialize()
    again = ExportedProgram.deserialize(blob, program.metadata)
    p, xx, extra = op.prepared(params, x)
    np.testing.assert_allclose(
        again.call(p, xx, extra, None)[0], op(params, x), rtol=1e-12, atol=1e-12
    )


# ------------------------------------------------------------------------------------------------ residuals


def test_residual_program_reports_cold_cost_and_pad_block_shares_an_executable():
    params = _init([D, 8, 1])
    rng = np.random.default_rng(2)
    stages = []
    for n in (5, 6, 7):
        x = jnp.asarray(rng.standard_normal((n, D)))
        target = jnp.asarray(rng.standard_normal(n))
        xp, mask, tp = pad_block(x, target, bucket=8)
        res = masked_residual(Residual(lambda q, t=tp: q.u[:, 0] - t, ["value"]), mask)
        op = compile_residual(mlp, [(xp, res)], example_params=params, quantity="gram")
        stages.append(op.warmup(params)["signature"])
        r, K = op(params)
        # the unpadded Gram is the leading block
        res0 = Residual(lambda q, t=target: q.u[:, 0] - t, ["value"])
        r0, K0 = compile_residual(mlp, [(x, res0)], example_params=params, quantity="gram")(params)
        np.testing.assert_allclose(K[:n, :n], K0, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(r[:n], r0, rtol=1e-12, atol=1e-12)
        assert float(jnp.abs(K[n:]).max()) == 0.0
        ex = op.explain()["cold_cost"]
        assert ex["executables"] == 1 and ex["stages"][0]["first_run_seconds"] is not None
    assert len(set(stages)) == 1  # every batch size mapped to the same shapes


def test_residual_damping_may_be_a_python_float():
    params = _init([D, 8, 1])
    x = _x(5)
    res = Residual(lambda q: q.u[:, 0] - 0.1, ["value"])
    op = compile_residual(mlp, [(x, res)], example_params=params)
    r, K, step = op(params, damping=1e-3)
    r2, K2, step2 = op(params, damping=2e-3)
    assert op.explain()["cold_cost"]["executables"] == 1
    assert not np.allclose(
        np.asarray(jax.tree_util.tree_leaves(step)[0]),
        np.asarray(jax.tree_util.tree_leaves(step2)[0]),
    )


def test_staged_operator_still_composes_with_jit_grad_and_vmap():
    params = _init([D, 8, 1])
    x = _x(4)
    op = compile_operator(mlp, (params, x), bucket="pow2")
    ref = compile_operator(mlp, (params, x), jit=False)
    eager = op(params, x)
    np.testing.assert_allclose(jax.jit(lambda p: op(p, x))(params), eager, rtol=1e-12, atol=1e-12)
    g = jax.jit(jax.grad(lambda p: jnp.sum(op(p, x))))(params)
    g0 = jax.grad(lambda p: jnp.sum(ref(p, x)))(params)
    for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(g0), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    xs = jnp.stack([_x(4, seed=s) for s in (1, 2, 3)])
    out = jax.vmap(lambda xx: op(params, xx))(xs)
    np.testing.assert_allclose(out[1], op(params, xs[1]), rtol=1e-12, atol=1e-12)
    # transformed calls are traced into the caller: they never add an executable
    assert op.explain()["cold_cost"]["executables"] == 1


def test_residual_program_composes_with_grad():
    params = _init([D, 8, 1])
    x = _x(5)
    res = Residual(lambda q: q.u[:, 0] - 0.1, ["value"])
    op = compile_residual(mlp, [(x, res)], example_params=params, quantity="gram")
    g = jax.grad(lambda p: jnp.sum(op(p)[1]))(params)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(g))


# ------------------------------------------------------------------------------------ memory refusals


def test_staged_program_refuses_an_executable_above_the_budget():
    from omnibias.jax.experimental.coldcost import MemoryBudgetExceeded

    fn = StagedProgram(lambda a: (a @ a.T).sum(0), memory_budget=1.0)
    a = jnp.ones((64, 32))
    with pytest.raises(MemoryBudgetExceeded) as info:
        fn(a)
    assert info.value.peak_bytes > 1 and "GiB" in str(info.value)
    assert fn.signatures() == 0  # nothing was kept, nothing ran
    ok = StagedProgram(lambda a: (a @ a.T).sum(0), memory_budget=1e9)
    ok(a)
    mem = ok.report()[0]["memory"]
    assert mem is not None and mem["peak_bytes"] >= mem["argument_bytes"] > 0


def test_operator_retries_the_next_candidate_when_the_compiled_program_is_refused(monkeypatch):
    from omnibias.jax.experimental import coldcost

    params = _init([D, 8, 1])
    x = _x(5)
    ref = compile_operator(mlp, (params, x))
    first = ref.algorithm
    op = compile_operator(mlp, (params, x), memory_budget=1e12)
    assert op.algorithm == first
    calls = {"n": 0}
    real = coldcost._memory_of

    def fake(executable):
        calls["n"] += 1
        m = real(executable)
        if (
            calls["n"] == 1 and m is not None
        ):  # the first program built claims to need far more than the budget
            return {**m, "peak_bytes": int(2e12)}
        return m

    monkeypatch.setattr(coldcost, "_memory_of", fake)
    out = op(params, x)
    np.testing.assert_allclose(out, ref(params, x), rtol=1e-10, atol=1e-12)
    ex = op.explain()
    assert op.algorithm != first and ex["mode"].endswith("+memory_retry")
    assert any(
        f["algorithm"] == first and "refused before running" in f["reason"]
        for f in ex["executor_failures"]
    )


def test_operator_with_every_candidate_over_budget_raises_the_refusals(monkeypatch):
    from omnibias.jax.experimental import coldcost

    params = _init([D, 8, 1])
    x = _x(5)
    op = compile_operator(mlp, (params, x), memory_budget=1e12)
    real = coldcost._memory_of
    monkeypatch.setattr(
        coldcost, "_memory_of", lambda e: {**(real(e) or {}), "peak_bytes": int(2e12)}
    )
    with pytest.raises(coldcost.MemoryBudgetExceeded) as info:
        op(params, x)
    assert any("every candidate was refused" in n for n in info.value.__notes__)


def test_residual_program_retries_on_memory_refusal(monkeypatch):
    from omnibias.jax.experimental import coldcost

    params = _init([D, 8, 1])
    res = Residual(lambda q: q.lap[:, 0] - 0.1, ["lap"])
    op = compile_residual(
        mlp, [(_x(5), res)], example_params=params, quantity="gram", memory_budget=1e12
    )
    first = op.algorithm
    calls = {"n": 0}
    real = coldcost._memory_of

    def fake(executable):
        calls["n"] += 1
        m = real(executable)
        return {**m, "peak_bytes": int(2e12)} if calls["n"] == 1 and m is not None else m

    monkeypatch.setattr(coldcost, "_memory_of", fake)
    r, K = op(params)
    assert op.algorithm != first
    assert K.shape == (5, 5)


def test_call_signature_separates_shape_dtype_and_weak_type():
    """The per-call key reads device-array attributes directly; it must still distinguish every compile-relevant fact."""
    from omnibias.jax.experimental import coldcost

    a = jnp.ones((3, 4))
    base = coldcost._signature((a, 1.0))
    assert coldcost._signature((jnp.zeros((3, 4)), 2.0)) == base  # values do not matter
    assert coldcost._signature((jnp.ones((4, 3)), 1.0)) != base  # shape
    assert coldcost._signature((a.astype(jnp.float32), 1.0)) != base  # dtype
    assert (
        coldcost._signature((a, jnp.asarray(1.0, dtype=jnp.float64))) != base
    )  # a Python scalar is weakly typed
    assert (
        coldcost._signature((np.ones((3, 4)), 1.0)) == base
    )  # a host array is the same program as a device array
    assert "float64[3, 4]" in coldcost._describe(base[1])


# ----------------------------------------------------------------------------------- compile retry on XLA failures
class _Flaky(StagedProgram):
    """Fails like XLA's GPU autotuner for the option sets in ``fail`` (a stand-in: the failure needs a GPU)."""

    def __init__(self, fn, *, fail, message="Failed to get configs for the GEMM fusion", **kw):
        super().__init__(fn, **kw)
        self.fail, self.message, self.seen = fail, message, []

    def _compile_lowered(self, lowered, options):
        self.seen.append(dict(options))
        if tuple(sorted(options.items())) in self.fail:
            raise RuntimeError(self.message)
        return lowered.compile()


def test_retryable_errors_are_recognised():
    from omnibias.jax.experimental.coldcost import RETRY_LADDER, is_retryable_xla_error

    assert is_retryable_xla_error(RuntimeError("INTERNAL: Failed to get configs for fusion"))
    assert is_retryable_xla_error(RuntimeError("CUDA_ERROR_MISALIGNED_ADDRESS: misaligned address"))
    assert not is_retryable_xla_error(ValueError("shapes do not match"))
    assert RETRY_LADDER[0] == {"xla_gpu_autotune_level": 0}
    assert RETRY_LADDER[1]["xla_gpu_enable_triton_gemm"] is False


def test_auto_policy_retries_with_the_next_options_and_records_the_rung():
    x = jnp.arange(4.0)
    default = ()
    level0 = (("xla_gpu_autotune_level", 0),)
    p = _Flaky(lambda a: a * 2.0, fail={default, level0}, retry=True)
    st = p.compile(x)
    assert p.seen == [
        {},
        {"xla_gpu_autotune_level": 0},
        {"xla_gpu_autotune_level": 0, "xla_gpu_enable_triton_gemm": False},
    ]
    assert (
        st.retried == 2
        and st.options == p.seen[-1]
        and "Failed to get configs" in (st.options_error or "")
    )
    np.testing.assert_allclose(p(x), 2.0 * np.arange(4.0))
    assert p.report()[0]["retried"] == 2


def test_without_retry_an_autotune_failure_is_raised_and_other_errors_never_retry():
    x = jnp.arange(4.0)
    with pytest.raises(RuntimeError, match="Failed to get configs"):
        _Flaky(lambda a: a + 1.0, fail={()}, retry=False).compile(x)
    other = _Flaky(lambda a: a + 1.0, fail={()}, retry=True, message="out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        other.compile(x)
    assert other.seen == [{}]


def test_runtime_autotune_failure_at_first_execution_reruns_with_the_fallback():
    x = jnp.arange(4.0)
    p = StagedProgram(lambda a: a * 3.0, retry=True)
    p.compile(x)
    key = next(iter(p._table))
    good = p._table[key]
    calls = {"n": 0}

    def broken(*args):
        calls["n"] += 1
        raise RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered")

    p._table[key] = broken
    out = p(x)
    np.testing.assert_allclose(out, 3.0 * np.arange(4.0))
    assert calls["n"] == 1 and p._table[key] is not broken and p._table[key] is not good
    assert p.stages[0].retried == 1
    q = StagedProgram(lambda a: a * 3.0, retry=False)
    q.compile(x)
    q._table[next(iter(q._table))] = broken
    with pytest.raises(RuntimeError, match="illegal memory access"):
        q(x)


def test_compile_policy_auto_turns_retry_on():
    params, x = _init([D, 8, 1]), _x(4)
    op = compile_operator(mlp, (params, x), compile_policy="auto")
    assert op._staged is not None and op._staged.retry is True
    default = compile_operator(mlp, (params, x))
    assert default._staged is not None and default._staged.retry is False

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``compile_operator``: chain matching, planning, execution and the generic fallback."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.planner import PlanCache  # noqa: E402
from omnibias.jax.experimental import (  # noqa: E402
    compile_operator,
    match_dense_chain,
    projected_deep_laplacian,
)

D = 6


def _init(widths, seed=0):
    rng = np.random.default_rng(seed)
    params = []
    for a, b in zip(widths[:-1], widths[1:], strict=True):
        params.append(
            (
                jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
                jnp.asarray(0.2 * rng.standard_normal(b)),
            )
        )
    return params


def _mlp(act):
    def model(params, x):
        a = x
        for W, b in params[:-1]:
            a = act(a @ W + b)
        W, b = params[-1]
        out = a @ W + b
        return out[..., 0] if out.shape[-1] == 1 else out

    return model


def _x(batch=5, dim=D, seed=1):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((batch, dim)))


def _trace_hessian(fn, params, x, A=None):
    """Reference ``tr(A hess f)`` per point by dense autodiff (``A = I`` by default)."""

    def point(xi):
        return fn(params, xi[None])[0]

    def one(xi):
        H = jax.hessian(point)(xi)
        return jnp.trace(H if A is None else A @ H, axis1=-2, axis2=-1)

    return jax.vmap(one)(x)


MLP = _mlp(jnp.tanh)


# ------------------------------------------------------------------------------------------------- matching


def test_matches_a_tanh_chain_and_confirms_numerically() -> None:
    params, x = _init([D, 8, 8, 1]), _x()
    match, why = match_dense_chain(MLP, (params, x))
    assert why is None and match is not None
    assert [lm.activation for lm in match.layers] == ["tanh", "tanh", None]
    assert match.squeeze_output and match.batched
    assert [lm.weight_form for lm in match.layers] == ["in_out", "in_out", "in_out"]


def test_matches_transposed_weights_vector_readout_and_scalar_bias() -> None:
    rng = np.random.default_rng(3)
    params = (
        (jnp.asarray(rng.standard_normal((8, D))), jnp.asarray(rng.standard_normal(8))),
        jnp.asarray(rng.standard_normal(8)),
        jnp.asarray(0.3),
    )

    def model(p, x):
        (W, b), w, c = p
        return jnp.tanh(x @ W.T + b) @ w + c

    x = _x()
    match, why = match_dense_chain(model, (params, x))
    assert why is None and match is not None
    assert [lm.weight_form for lm in match.layers] == ["out_in", "vector"]
    op = compile_operator(model, (params, x))
    np.testing.assert_allclose(
        np.asarray(op(params, x)),
        np.asarray(_trace_hessian(lambda p, z: model(p, z), params, x)),
        rtol=1e-10,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    ("act", "name"),
    [
        (jax.nn.silu, "silu"),
        (jax.nn.softplus, "softplus"),
        (jax.nn.sigmoid, "sigmoid"),
        (jnp.sin, "sin"),
    ],
)
def test_named_and_primitive_activations_are_recognised(act, name) -> None:
    params, x = _init([D, 7, 1]), _x()
    match, why = match_dense_chain(_mlp(act), (params, x))
    assert why is None and match is not None, why
    assert match.layers[0].activation == name


def test_relu_is_recognised_and_the_a_e_zero_second_derivative_is_refused() -> None:
    params, x = _init([D, 7, 1]), _x()
    model = _mlp(jax.nn.relu)
    match, why = match_dense_chain(model, (params, x))
    assert why is None and match is not None and match.layers[0].activation == "relu"
    op = compile_operator(model, (params, x))
    info = op.explain()
    refused = {r["algorithm"]: r["reason"] for r in info["refused"]}
    assert "projected" in refused and "allow_almost_everywhere" in refused["projected"]
    assert info["selected_strategy"] in ("taylor_jet", "nested_autodiff", "folx")
    allowed = compile_operator(model, (params, x), allow_almost_everywhere=True)
    assert allowed.explain()["selected_strategy"] in ("projected", "forward_laplacian")


@pytest.mark.parametrize("variant", ["fourier", "residual", "gelu", "mislabelled"])
def test_programs_that_are_not_chains_are_refused_with_a_reason(variant) -> None:
    params, x = _init([D, 8, 1]), _x()
    if variant == "fourier":

        def model(p, z):
            h = jnp.concatenate([jnp.sin(z @ p[0][0]), jnp.cos(z @ p[0][0])], axis=-1)
            return jnp.sum(h, axis=-1) + p[1][1][0]

    elif variant == "residual":

        def model(p, z):
            h = jnp.tanh(z @ p[0][0] + p[0][1])
            return (h + h) @ p[1][0][:, 0] + p[1][1][0]

    elif variant == "gelu":

        def model(p, z):
            return (jax.nn.gelu(z @ p[0][0] + p[0][1]) @ p[1][0] + p[1][1])[..., 0]

    else:

        @jax.jit
        def silu(
            v,
        ):  # a user function that shares the name of a known activation but computes something else
            return jnp.tanh(v)

        def model(p, z):
            return (silu(z @ p[0][0] + p[0][1]) @ p[1][0] + p[1][1])[..., 0]

    match, why = match_dense_chain(model, (params, x))
    assert match is None and why
    op = compile_operator(model, (params, x))
    info = op.explain()
    assert info["matched_patterns"] == [] and info["fallback_reason"] == why
    assert info["selected_strategy"] == "taylor_jet"
    want = _trace_hessian(lambda p, z: jnp.atleast_1d(model(p, z)), params, x)
    np.testing.assert_allclose(np.asarray(op(params, x)), np.asarray(want), rtol=1e-9, atol=1e-10)


# ----------------------------------------------------------------------------------------- the Laplacian


@pytest.mark.parametrize(
    "algorithm", ["projected", "forward_laplacian", "taylor_jet", "nested_autodiff"]
)
def test_every_laplacian_executor_agrees_with_dense_autodiff(algorithm) -> None:
    params, x = _init([D, 8, 8, 1]), _x()
    op = compile_operator(MLP, (params, x), force=algorithm)
    assert op.algorithm == algorithm and op.explain()["mode"] == "forced"
    want = _trace_hessian(MLP, params, x)
    np.testing.assert_allclose(np.asarray(op(params, x)), np.asarray(want), rtol=1e-10, atol=1e-11)


def test_default_dispatch_is_logged_and_gradients_reach_the_parameters() -> None:
    params, x = _init([D, 8, 8, 1]), _x()
    op = compile_operator(MLP, (params, x))
    info = op.explain()
    assert info["matched_patterns"] == ["dense_chain"] and info["fallback_reason"] is None
    assert info["selected_strategy"] in {c["name"] for c in info["candidates"]}
    assert info["chain"]["layers"][0]["in_features"] == D
    grads = jax.grad(lambda p: jnp.sum(op(p, x) ** 2))(params)
    want = jax.grad(lambda p: jnp.sum(_trace_hessian(MLP, p, x) ** 2))(params)
    for got, ref in zip(
        jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(want), strict=True
    ):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("power", [2, 3])
def test_laplace_powers_agree_with_nested_autodiff(power) -> None:
    dim = 4
    params, x = _init([dim, 6, 6, 1], seed=4), _x(3, dim, seed=5)
    got = compile_operator(MLP, (params, x), power=power, force="projected")(params, x)
    nested = compile_operator(MLP, (params, x), power=power, force="nested_autodiff")(params, x)
    np.testing.assert_allclose(np.asarray(got), np.asarray(nested), rtol=1e-8, atol=1e-9)


def test_selected_coordinates_and_diffusion_agree_across_executors() -> None:
    params, x = _init([D, 8, 8, 1]), _x()
    coords = [0, 2, 3]
    s = np.array([0.5, 1.0, 1.5])
    kw = dict(coordinates=coords, diffusion=s)
    A = np.zeros((D, D))
    A[coords, coords] = s**2
    want = _trace_hessian(MLP, params, x, jnp.asarray(A))
    for algorithm in ("projected", "taylor_jet", "nested_autodiff"):
        got = compile_operator(MLP, (params, x), force=algorithm, **kw)(params, x)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-9, atol=1e-10, err_msg=algorithm
        )
    # the forward-Laplacian program cannot do a subset or a metric: refused with the reason, not silently wrong
    info = compile_operator(MLP, (params, x), **kw).explain()
    assert any(r["algorithm"] == "forward_laplacian" for r in info["refused"])
    with pytest.raises(ValueError, match="cannot force"):
        compile_operator(MLP, (params, x), force="forward_laplacian", **kw)


def test_evaluate_returns_the_by_products_of_the_program() -> None:
    params, x = _init([D, 8, 1]), _x()
    op = compile_operator(MLP, (params, x), force="projected")
    out = op.evaluate(params, x)
    np.testing.assert_allclose(np.asarray(out.aux["value"]), np.asarray(MLP(params, x)), rtol=1e-12)
    layers = [(W.T, b, "tanh") for W, b in params[:-1]] + [(params[-1][0].T, params[-1][1], None)]
    _, gs, _, _ = projected_deep_laplacian(x, layers)
    np.testing.assert_allclose(np.asarray(out.aux["grad_sq"]), np.asarray(gs[:, 0]), rtol=1e-12)


def test_unbatched_input_and_bound_models() -> None:
    params, x = _init([D, 8, 1]), _x(1)[0]
    op = compile_operator(MLP, (params, x))
    assert op(params, x).shape == ()
    bound = compile_operator(lambda z: MLP(params, z), (x,), params_argnum=None)
    assert bound.explain()["matched_patterns"] == ["dense_chain"]
    np.testing.assert_allclose(float(bound(x)), float(op(params, x)), rtol=1e-12)


def test_layer_list_models_are_passed_again_at_call_time() -> None:
    params, x = _init([D, 8, 8, 1]), _x()
    layers = [(W.T, b, "tanh") for W, b in params[:-1]] + [(params[-1][0].T, params[-1][1], None)]
    op = compile_operator(layers, x, operator="laplacian", force="projected")
    _, _, lap, _ = projected_deep_laplacian(x, layers)
    np.testing.assert_allclose(np.asarray(op(layers, x)), np.asarray(lap), rtol=1e-12)
    changed = [(W, b, "sin") for W, b, _ in layers[:-1]] + [layers[-1]]
    with pytest.raises(ValueError, match="activations differ"):
        op(changed, x)
    specs = [s for _, _, s in layers]
    arrays = [(W, b) for W, b, _ in layers]
    grad = jax.grad(
        lambda arr: jnp.sum(op([(W, b, s) for (W, b), s in zip(arr, specs, strict=True)], x))
    )(arrays)
    assert jnp.all(jnp.isfinite(grad[0][0])) and float(jnp.abs(grad[0][0]).max()) > 0.0


# ------------------------------------------------------------------------------------ vector-field programs


def _field(dim=4):
    params = _init([dim, 10, 10, dim], seed=7)
    return params, _mlp(jnp.tanh), _x(3, dim, seed=8)


def _jac(model, params, x):
    return jax.vmap(jax.jacfwd(lambda xi: model(params, xi[None])[0]))(x)


def test_divergence_logdet_and_solve_match_the_dense_jacobian() -> None:
    params, model, x = _field()
    J = _jac(model, params, x)
    div = compile_operator(model, (params, x), operator="divergence")
    np.testing.assert_allclose(
        np.asarray(div(params, x)), np.asarray(jnp.trace(J, axis1=-2, axis2=-1)), rtol=1e-10
    )
    logdet = compile_operator(model, (params, x), operator="logdet")
    eye = jnp.eye(J.shape[-1])
    np.testing.assert_allclose(
        np.asarray(logdet(params, x)),
        np.asarray(jnp.linalg.slogdet(eye + J)[1]),
        rtol=1e-10,
        atol=1e-11,
    )
    rhs = _x(3, J.shape[-1], seed=9)
    for transpose in (False, True):
        solve = compile_operator(model, (params, x, rhs), operator="solve", transpose=transpose)
        M = eye + (jnp.swapaxes(J, -1, -2) if transpose else J)
        want = jnp.linalg.solve(M, rhs[..., None])[..., 0]
        np.testing.assert_allclose(
            np.asarray(solve(params, x, rhs)), np.asarray(want), rtol=1e-9, atol=1e-10
        )


def test_vector_field_generic_executors_agree_with_the_structured_ones() -> None:
    params, model, x = _field()
    for operator, structured, generic in (
        ("divergence", "projected", "jacobian_trace"),
        ("logdet", "projected", "dense_jacobian"),
    ):
        a = compile_operator(model, (params, x), operator=operator, force=structured)(params, x)
        b = compile_operator(model, (params, x), operator=operator, force=generic)(params, x)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-11)


def test_residual_programs_are_planned_elsewhere() -> None:
    params, x = _init([D, 8, 1]), _x()
    with pytest.raises(ValueError, match="compile_residual"):
        compile_operator(MLP, (params, x), operator="gram")


# -------------------------------------------------------------------------------------- planning behaviour


def test_plans_are_cached_and_persist(tmp_path) -> None:
    params, x = _init([D, 8, 1]), _x()
    path = tmp_path / "plans.json"
    first = compile_operator(MLP, (params, x), cache=str(path))
    assert path.exists()
    cache = PlanCache(str(path))
    second = compile_operator(MLP, (params, x), cache=cache)
    assert cache.hits == 1 and second.algorithm == first.algorithm


def test_find_mode_measures_the_leading_candidates() -> None:
    params, x = _init([D, 8, 1]), _x()
    op = compile_operator(MLP, (params, x), mode="find", cache=PlanCache())
    info = op.explain()
    assert info["mode"] == "find" and len(info["measured_seconds"]) >= 2
    assert info["selected_strategy"] == min(
        info["measured_seconds"], key=info["measured_seconds"].get
    )


def test_find_mode_survives_a_candidate_that_cannot_run(monkeypatch) -> None:
    from omnibias.jax.experimental import compiler

    key = ("laplacian", "projected")
    real = compiler._EXECUTORS[key]

    def broken(ctx):
        real(ctx)

        def boom(*_args, **_kwargs):
            raise RuntimeError("out of memory (injected)")

        return boom

    monkeypatch.setitem(compiler._EXECUTORS, key, broken)
    params, x = _init([D, 8, 1]), _x()
    info = compile_operator(MLP, (params, x), mode="find", cache=PlanCache()).explain()
    assert info["selected_strategy"] != "projected"
    assert any(
        f["algorithm"] == "projected" and "find: RuntimeError" in f["reason"]
        for f in info["executor_failures"]
    )


def test_find_times_the_program_the_operator_runs_and_refuses_what_exceeds_the_budget(monkeypatch) -> None:
    """The timed program carries the operator's precision and compute dtype, and a compiled size over the budget is refused."""
    from omnibias.jax.experimental import compiler
    from omnibias.jax.experimental.coldcost import MemoryBudgetExceeded

    seen_precision: list[str] = []
    real_policy = compiler._with_policy

    def spy(run, precision, compute_dtype, *args):
        seen_precision.append(precision.matmul)
        return real_policy(run, precision, compute_dtype, *args)

    monkeypatch.setattr(compiler, "_with_policy", spy)
    params, x = _init([D, 8, 1]), _x()
    compile_operator(MLP, (params, x), mode="find", cache=PlanCache(), precision="tf32")
    assert seen_precision and set(seen_precision) == {"tf32"}

    budgets: list[object] = []
    real_staged = compiler.StagedProgram

    class Refusing(real_staged):  # the first program built exceeds the budget once compiled
        def compile(self, *args):
            budgets.append(self.memory_budget)
            if len(budgets) == 1:
                raise MemoryBudgetExceeded("injected", 2**40, 1.0)
            return super().compile(*args)

    monkeypatch.setattr(compiler, "StagedProgram", Refusing)
    info = compile_operator(MLP, (params, x), mode="find", cache=PlanCache()).explain()
    assert info["selected_strategy"] in info["measured_seconds"]
    assert any("MemoryBudgetExceeded" in f["reason"] and "find:" in f["reason"] for f in info["executor_failures"])


def test_find_with_top_k_none_times_every_feasible_program_and_the_cache_is_keyed_on_it() -> None:
    params, x = _init([D, 8, 1]), _x()
    cache = PlanCache()
    default = compile_operator(MLP, (params, x), mode="find", cache=cache).explain()
    every = compile_operator(MLP, (params, x), mode="find", top_k=None, cache=cache).explain()
    feasible = {c["name"] for c in every["candidates"]}
    find_refused = {r["algorithm"] for r in every["refused"] if r["reason"].startswith("find:")}
    assert len(default["measured_seconds"]) == 2 < len(feasible)  # the default trusts the ranking for the rest
    assert set(every["measured_seconds"]) == feasible - find_refused
    assert every["selected_strategy"] == min(every["measured_seconds"], key=every["measured_seconds"].get)
    assert len(cache) == 2  # an exhaustive find is its own measurement
    compile_operator(MLP, (params, x), mode="find", top_k=None, cache=cache)
    assert cache.hits == 1 and len(cache) == 2


def test_a_compiled_size_over_the_budget_is_a_refusal_the_cached_plan_keeps(monkeypatch) -> None:
    from omnibias.jax.experimental import compiler
    from omnibias.jax.experimental.coldcost import MemoryBudgetExceeded

    refused: list[int] = []
    real_staged = compiler.StagedProgram

    class Refusing(real_staged):  # the first program built exceeds the budget once compiled
        def compile(self, *args):
            if not refused:
                refused.append(1)
                raise MemoryBudgetExceeded("injected", 2**40, 1.0)
            return super().compile(*args)

    monkeypatch.setattr(compiler, "StagedProgram", Refusing)
    params, x = _init([D, 8, 1]), _x()
    cache = PlanCache()
    first = compile_operator(MLP, (params, x), mode="find", cache=cache)
    assert len(cache) == 1  # the failure repeats for this problem and budget, so the plan is kept
    kept = [r for r in first.explain()["refused"] if r["reason"].startswith("find: MemoryBudgetExceeded")]
    assert len(kept) == 1
    assert kept[0]["algorithm"] not in first.explain()["measured_seconds"]
    again = compile_operator(MLP, (params, x), mode="find", cache=cache)
    assert cache.hits == 1 and again.algorithm == first.algorithm
    assert [r for r in again.explain()["refused"] if "MemoryBudgetExceeded" in r["reason"]] == kept
    assert len(refused) == 1  # the second call timed nothing


def test_a_failure_that_may_not_repeat_keeps_the_plan_out_of_the_cache(monkeypatch) -> None:
    from omnibias.jax.experimental import compiler

    key = ("laplacian", "projected")
    real = compiler._EXECUTORS[key]

    def broken(ctx):
        real(ctx)

        def boom(*_args, **_kwargs):
            raise RuntimeError("out of memory (injected)")

        return boom

    monkeypatch.setitem(compiler._EXECUTORS, key, broken)
    params, x = _init([D, 8, 1]), _x()
    cache = PlanCache()
    compile_operator(MLP, (params, x), mode="find", top_k=None, cache=cache)
    assert len(cache) == 0


def test_find_keys_carry_the_library_versions(monkeypatch) -> None:
    from omnibias.jax.experimental import compiler

    params, x = _init([D, 8, 1]), _x()
    cache = PlanCache()
    compile_operator(MLP, (params, x), mode="find", cache=cache)
    compile_operator(MLP, (params, x), mode="find", cache=cache)
    assert cache.hits == 1 and len(cache) == 1
    real = compiler._versions
    monkeypatch.setattr(compiler, "_versions", lambda: {**real(), "jax": "0.0.0-another"})
    compile_operator(MLP, (params, x), mode="find", cache=cache)
    assert cache.hits == 1 and len(cache) == 2  # measured again, not reused across a library upgrade


def test_find_cache_keys_name_the_precision_and_compute_dtype() -> None:
    params, x = _init([D, 8, 1]), _x()
    cache = PlanCache()
    compile_operator(MLP, (params, x), mode="find", cache=cache)
    compile_operator(MLP, (params, x), mode="find", cache=cache)
    assert cache.hits == 1 and len(cache) == 1  # the defaults keep the key they always had
    compile_operator(MLP, (params, x), mode="find", cache=cache, precision="tf32")
    assert len(cache) == 2  # another matmul mode is another measurement
    compile_operator(MLP, (params, x), mode="find", cache=cache, compute_dtype="float64")
    assert len(cache) == 3


def test_memory_budget_refuses_with_reasons() -> None:
    params, x = _init([D, 8, 1]), _x()
    with pytest.raises(ValueError, match="memory_budget"):
        compile_operator(MLP, (params, x), memory_budget=16.0)


def test_inexact_requests_admit_the_hutchinson_estimator() -> None:
    params, x = _init([D, 8, 1]), _x()
    exact = compile_operator(MLP, (params, x))
    assert exact.explain()["selected_strategy"] != "hutchinson"
    assert any(r["algorithm"] == "hutchinson" for r in exact.explain()["refused"])
    est = compile_operator(MLP, (params, x), exact=False, force="hutchinson", n_probes=4096)
    key = jax.random.PRNGKey(0)
    np.testing.assert_allclose(
        np.asarray(est(params, x, key=key)),
        np.asarray(exact(params, x)),
        rtol=0.25,
        atol=0.1 * float(jnp.abs(exact(params, x)).max()),
    )


def test_first_call_cost_is_recorded() -> None:
    params, x = _init([D, 8, 1]), _x()
    op = compile_operator(MLP, (params, x))
    assert op.explain()["first_call_seconds"] is None
    op(params, x)
    assert op.explain()["first_call_seconds"] > 0.0


def test_the_planned_context_carries_the_precision_policy_the_fused_program_follows():
    model, params, x = _mlp(jnp.tanh), _init((D, 16, 16, 1)), _x()
    modes = {}
    for name in ("highest", "tf32", "default"):
        op = compile_operator(model, (params, x), precision=name, cache=PlanCache())
        modes[name] = op._ctx.matmul
    assert modes == {"highest": "highest", "tf32": "tf32", "default": "default"}

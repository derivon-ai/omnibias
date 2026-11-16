# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Third-party engines as planner executors: registration, refusal, forcing, timing and removal, on both back ends."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core import engines
from omnibias.core.planner import Descriptor, OperatorRequest, PlanCache, Work, algorithms_for


def test_work_must_be_priced_and_only_laplacian_and_divergence_are_served():
    with pytest.raises(ValueError, match="engines can serve"):
        engines.engine_algorithm("x", "solve", work="projected", backend="jax")
    with pytest.raises(ValueError, match="no registered algorithm 'nope'"):
        engines.engine_algorithm("x", "laplacian", work="nope", backend="jax")
    with pytest.raises(ValueError, match="identifier"):
        engines.engine_algorithm("not valid", "laplacian", work="nested_autodiff", backend="jax")
    algo = engines.engine_algorithm(
        "mine", "laplacian", work=lambda d: Work(gemm_flops=1.0), backend="torch", max_power=2
    )
    assert algo.operators == frozenset({"laplacian"}) and algo.backend == "torch" and algo.exact


def test_an_engine_is_refused_for_what_it_cannot_serve():
    algo = engines.engine_algorithm("mine", "laplacian", work="nested_autodiff", backend="jax")
    from omnibias.core.chain import ChainDescriptor

    chain = ChainDescriptor.from_widths([8, 4, 1], ["tanh", None])

    def refusal(**request):
        return algo.requires(Descriptor(OperatorRequest("laplacian", **request), chain, 4))

    assert refusal() is None
    assert "serves Laplace powers up to 1" in refusal(power=2)
    assert "coordinate subset" in refusal(selected=3)
    assert "metric" in refusal(metric="diagonal")
    assert "drift" in refusal(directions=2)


def _jax_case():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    D, H = 9, 6
    params = [
        (jnp.asarray(rng.standard_normal((D, H)) / 3), jnp.zeros(H)),
        (jnp.asarray(rng.standard_normal((H, 1)) / 3), jnp.zeros(1)),
    ]

    def model(p, x):
        return (jnp.tanh(x @ p[0][0] + p[0][1]) @ p[1][0] + p[1][1])[:, 0]

    return jax, jnp, model, params, jnp.asarray(rng.standard_normal((5, D)))


def test_jax_engine_is_forced_ranked_differentiated_and_removed():
    jax, jnp, model, params, X = _jax_case()
    from omnibias.jax.operators import compile_operator, register_engine, unregister_engine

    seen = []

    def trace_hessian(f, x, power=1):
        seen.append((x.shape, power))
        return jax.vmap(lambda z: jnp.trace(jax.hessian(f)(z)))(x)

    register_engine("hess_trace", "laplacian", trace_hessian, work="nested_autodiff")
    try:
        with pytest.raises(ValueError, match="already"):
            register_engine("hess_trace", "laplacian", trace_hessian, work="nested_autodiff")
        reference = compile_operator(model, (params, X), operator="laplacian", cache=PlanCache())
        forced = compile_operator(
            model, (params, X), operator="laplacian", force="hess_trace", cache=PlanCache()
        )
        explained = forced.explain()
        assert explained["selected_strategy"] == "hess_trace"
        assert "hess_trace" in [c["name"] for c in explained["candidates"]]
        np.testing.assert_allclose(forced(params, X), reference(params, X), rtol=1e-11, atol=1e-12)
        assert seen and seen[0][1] == 1
        got = jax.grad(lambda p: forced(p, X).sum())(params)
        want = jax.grad(lambda p: reference(p, X).sum())(params)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)
        with pytest.raises(ValueError, match="coordinate subset"):
            compile_operator(
                model,
                (params, X),
                operator="laplacian",
                force="hess_trace",
                coordinates=[0, 1, 2],
                cache=PlanCache(),
            )
        found = compile_operator(
            model, (params, X), operator="laplacian", mode="find", cache=PlanCache()
        )
        assert "hess_trace" in found.explain()["measured_seconds"] or found.explain()["mode"]
    finally:
        unregister_engine("hess_trace", "laplacian")
    assert "hess_trace" not in [a.name for a in algorithms_for("laplacian")]
    with pytest.raises(ValueError, match="built-in"):
        unregister_engine("projected", "laplacian")
    with pytest.raises(KeyError):
        unregister_engine("hess_trace", "laplacian")


def test_a_generic_jax_engine_serves_a_model_that_is_not_a_chain_by_force():
    jax, jnp, _, _, X = _jax_case()
    from omnibias.jax.operators import compile_operator, register_engine, unregister_engine

    def opaque(p, x):  # the input also enters elementwise: not a dense chain
        return jnp.sum(jnp.sin(x) * x * p, axis=-1)

    p = jnp.linspace(0.5, 1.5, X.shape[1])
    register_engine(
        "hess_trace",
        "laplacian",
        lambda f, x, power=1: jax.vmap(lambda z: jnp.trace(jax.hessian(f)(z)))(x),
        work="nested_autodiff",
        generic=True,
    )
    try:
        op = compile_operator(
            opaque, (p, X), operator="laplacian", force="hess_trace", cache=PlanCache()
        )
        ref = compile_operator(opaque, (p, X), operator="laplacian", cache=PlanCache())
        assert op.explain()["selected_strategy"] == "hess_trace"
        np.testing.assert_allclose(op(p, X), ref(p, X), rtol=1e-11, atol=1e-12)
    finally:
        unregister_engine("hess_trace", "laplacian")
    with pytest.raises(ValueError, match="cannot force"):
        compile_operator(
            opaque, (p, X), operator="laplacian", force="hess_trace", cache=PlanCache()
        )


def test_torch_engine_is_forced_and_differentiated():
    torch = pytest.importorskip("torch")
    from omnibias.torch.operators import compile_operator, register_engine, unregister_engine

    torch.manual_seed(0)
    layers = [
        (torch.randn(6, 9, dtype=torch.float64) / 3, torch.zeros(6, dtype=torch.float64), "tanh"),
        (torch.randn(1, 6, dtype=torch.float64) / 3, torch.zeros(1, dtype=torch.float64), None),
    ]
    X = torch.randn(5, 9, dtype=torch.float64)

    def trace_hessian(f, x, power=1):
        return torch.func.vmap(lambda z: torch.trace(torch.func.hessian(f)(z)))(x)

    register_engine("hess_trace", "laplacian", trace_hessian, work="nested_autodiff")
    try:
        reference = compile_operator(layers, X, operator="laplacian", cache=PlanCache())
        forced = compile_operator(
            layers, X, operator="laplacian", force="hess_trace", cache=PlanCache()
        )
        assert forced.explain()["selected_strategy"] == "hess_trace"
        torch.testing.assert_close(forced(X), reference(X), rtol=1e-10, atol=1e-11)
        weight = layers[0][0].clone().requires_grad_(True)
        layers_g = [(weight, *layers[0][1:]), layers[1]]
        forced_g = compile_operator(
            layers_g, X, operator="laplacian", force="hess_trace", cache=PlanCache()
        )
        forced_g(X).sum().backward()
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
    finally:
        unregister_engine("hess_trace", "laplacian")
    assert "hess_trace" not in [a.name for a in algorithms_for("laplacian")]


def test_folx_is_an_executor_the_planner_can_force():
    pytest.importorskip("folx")
    jax, jnp, model, params, X = _jax_case()
    from omnibias.jax.folx_compat import register_folx_engine
    from omnibias.jax.operators import compile_operator, unregister_engine

    register_folx_engine(replace=True)  # compile_operator registers it by default; this test pins the explicit call
    try:
        reference = compile_operator(model, (params, X), operator="laplacian", cache=PlanCache())
        viafolx = compile_operator(
            model, (params, X), operator="laplacian", force="folx", cache=PlanCache()
        )
        assert viafolx.explain()["selected_strategy"] == "folx"
        np.testing.assert_allclose(viafolx(params, X), reference(params, X), rtol=1e-11, atol=1e-12)
    finally:
        unregister_engine("folx", "laplacian")

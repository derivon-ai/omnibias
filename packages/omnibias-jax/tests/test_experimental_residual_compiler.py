# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``compile_residual``: every engine serves the same Gram, step and diagonal, and the planner routes by demand."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from omnibias.core.planner import PlanCache  # noqa: E402
from omnibias.jax.experimental import Residual, compile_residual, legacy_projected  # noqa: E402

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


def _blocks(seed=1):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((6, D)))
    xb = jnp.asarray(rng.standard_normal((3, D)))
    target = jnp.asarray(rng.standard_normal(3))
    interior = Residual(lambda q: q.lap[:, 0] + 0.5 * q.u[:, 0], ["value", "lap"])
    boundary = Residual(lambda q: q.u[:, 0] - target, ["value"])
    return [(x, interior), (xb, boundary)]


def _step_leaves(step):
    return jax.tree_util.tree_leaves(step)


def test_the_three_engines_agree_on_gram_step_and_diagonal() -> None:
    params = _init([D, 7, 6, 1])
    blocks = _blocks()
    results = {}
    for algorithm in ("projected_streams", "full_streams", "autodiff_rows"):
        gn = compile_residual(
            mlp, blocks, example_params=params, quantity="gauss_newton", force=algorithm
        )
        assert gn.algorithm == algorithm
        results[algorithm] = (
            gn(params, damping=1e-2),
            compile_residual(
                mlp, blocks, example_params=params, quantity="ntk_diagonal", force=algorithm
            )(params),
        )
    (r0, K0, step0), (_, diag0) = results["autodiff_rows"]
    for algorithm in ("projected_streams", "full_streams"):
        (r, K, step), (_, diag) = results[algorithm]
        np.testing.assert_allclose(
            np.asarray(r), np.asarray(r0), rtol=1e-10, atol=1e-11, err_msg=algorithm
        )
        np.testing.assert_allclose(
            np.asarray(K), np.asarray(K0), rtol=1e-9, atol=1e-10, err_msg=algorithm
        )
        np.testing.assert_allclose(
            np.asarray(diag), np.asarray(diag0), rtol=1e-9, atol=1e-10, err_msg=algorithm
        )
        # the step has the structure of the parameters and the same numbers
        assert jax.tree_util.tree_structure(step) == jax.tree_util.tree_structure(params)
        for got, ref in zip(_step_leaves(step), _step_leaves(step0), strict=True):
            assert got.shape == ref.shape
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(ref), rtol=1e-7, atol=1e-9, err_msg=algorithm
            )


def test_find_times_every_program_with_top_k_none() -> None:
    params = _init([D, 7, 6, 1])
    blocks = _blocks()
    cache = PlanCache()
    default = compile_residual(mlp, blocks, example_params=params, mode="find", cache=cache).explain()
    every = compile_residual(mlp, blocks, example_params=params, mode="find", top_k=None, cache=cache).explain()
    feasible = {c["name"] for c in every["candidates"]}
    assert len(default["measured_seconds"]) == 2 < len(feasible) == len(every["measured_seconds"])
    assert len(cache) == 2  # the exhaustive find has its own key


def test_a_program_that_fails_when_timed_is_a_failed_measurement_not_an_error(monkeypatch) -> None:
    from omnibias.jax.experimental import residual_compiler as rc

    params = _init([D, 7, 6, 1])
    blocks = _blocks()
    leader = compile_residual(mlp, blocks, example_params=params, cache=PlanCache()).algorithm

    def broken(ctx):
        def run(*_args, **_kwargs):
            raise RuntimeError("out of memory (injected)")

        return run

    monkeypatch.setitem(rc._RUNNERS, leader, broken)
    cache = PlanCache()
    info = compile_residual(mlp, blocks, example_params=params, mode="find", top_k=None, cache=cache).explain()
    assert info["selected_strategy"] != leader
    assert any(
        f["algorithm"] == leader and "find: RuntimeError" in f["reason"] for f in info["executor_failures"]
    )
    assert len(cache) == 0  # the failure may not repeat: nothing is cached


def test_vector_damping_gives_one_step_per_value() -> None:
    params = _init([D, 6, 1], seed=2)
    blocks = _blocks()
    damping = jnp.asarray([1e-1, 1.0])
    ref = compile_residual(mlp, blocks, example_params=params, force="autodiff_rows")(
        params, damping=damping
    )[2]
    for algorithm in ("projected_streams", "full_streams"):
        step = compile_residual(mlp, blocks, example_params=params, force=algorithm)(
            params, damping=damping
        )[2]
        for got, want in zip(_step_leaves(step), _step_leaves(ref), strict=True):
            assert got.shape == want.shape and got.shape[0] == 2
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-7, atol=1e-9)


def test_default_dispatch_is_logged_and_points_can_be_replaced_at_call_time() -> None:
    params = _init([D, 7, 1], seed=3)
    blocks = _blocks()
    op = compile_residual(mlp, blocks, example_params=params, quantity="gram")
    info = op.explain()
    assert info["matched_patterns"] == ["dense_chain"] and info["fallback_reason"] is None
    assert info["selected_strategy"] in {c["name"] for c in info["candidates"]}
    r, K = op(params)
    assert K.shape == (9, 9) and r.shape == (9,)
    shifted = [blocks[0][0] + 0.1, blocks[1][0]]
    r2, _ = op(params, shifted)
    assert float(jnp.abs(r2 - r).max()) > 1e-6
    assert op.explain()["first_call_seconds"] > 0.0


def test_demand_routes_between_engines_and_refusals_carry_reasons() -> None:
    params = _init([D, 7, 1], seed=4)
    x = _blocks()[0][0]
    needs_grad = Residual(lambda q: q.grad[:, 0, 0] + q.u[:, 0], ["value", "grad"])
    op = compile_residual(mlp, [(x, needs_grad)], example_params=params, quantity="gram")
    info = op.explain()
    assert info["selected_strategy"] in ("full_streams", "autodiff_rows")
    refused = {r["algorithm"]: r["reason"] for r in info["refused"]}
    assert "projected_streams" in refused and "grad" in refused["projected_streams"]
    r_ref = compile_residual(
        mlp, [(x, needs_grad)], example_params=params, quantity="gram", force="autodiff_rows"
    )(params)[0]
    np.testing.assert_allclose(np.asarray(op(params)[0]), np.asarray(r_ref), rtol=1e-10)


def test_fourth_order_demand_uses_the_projected_engine_or_autodiff() -> None:
    params = _init([D, 6, 1], seed=5)
    x = _blocks()[0][0]
    ks = Residual(
        lambda q: q.bilap[:, 0] + q.lap[:, 0] + q.u[:, 0] * q.grad_sq[:, 0],
        ["value", "grad_sq", "lap", "bilap"],
    )
    got = {}
    for algorithm in ("projected_streams", "autodiff_rows"):
        got[algorithm] = compile_residual(
            mlp, [(x, ks)], example_params=params, quantity="gram", force=algorithm
        )(params)
    np.testing.assert_allclose(
        np.asarray(got["projected_streams"][1]),
        np.asarray(got["autodiff_rows"][1]),
        rtol=1e-8,
        atol=1e-9,
    )
    default = compile_residual(mlp, [(x, ks)], example_params=params, quantity="gram")
    assert any(r["algorithm"] == "full_streams" for r in default.explain()["refused"])


def test_selected_coordinates_diffusion_and_drift_agree_with_autodiff() -> None:
    params = _init([D, 6, 1], seed=6)
    x = _blocks()[0][0]
    drift = np.random.default_rng(7).standard_normal((D, 2))
    res = Residual(
        lambda q: q.lap[:, 0]
        + q.directional[:, 0, 0]
        + 0.3 * q.directional[:, 1, 0]
        - q.grad_sq[:, 0],
        ["value", "grad_sq", "lap", "directional"],
    )
    kw = dict(
        coordinates=[0, 2, 3],
        diffusion=np.array([0.5, 1.0, 1.5]),
        directions=drift,
        quantity="gram",
    )
    a = compile_residual(mlp, [(x, res)], example_params=params, force="projected_streams", **kw)(
        params
    )
    b = compile_residual(mlp, [(x, res)], example_params=params, force="autodiff_rows", **kw)(
        params
    )
    np.testing.assert_allclose(np.asarray(a[1]), np.asarray(b[1]), rtol=1e-8, atol=1e-9)
    full = compile_residual(mlp, [(x, res)], example_params=params, **kw).explain()
    assert any(r["algorithm"] == "full_streams" for r in full["refused"])


def test_models_that_are_not_chains_fall_back_to_autodiff_rows() -> None:
    rng = np.random.default_rng(8)
    params = (jnp.asarray(rng.standard_normal((D, 6))), jnp.asarray(rng.standard_normal(6)))

    def fourier(p, z):
        return jnp.sum(
            jnp.concatenate([jnp.sin(z @ p[0]), jnp.cos(z @ p[0])], axis=-1) * p[1].sum(), axis=-1
        )

    x = _blocks()[0][0]
    res = Residual(lambda q: q.lap[:, 0] - q.u[:, 0], ["value", "lap"])
    op = compile_residual(fourier, [(x, res)], example_params=params, quantity="gauss_newton")
    info = op.explain()
    assert info["selected_strategy"] == "autodiff_rows" and info["matched_patterns"] == []
    assert info["fallback_reason"]
    r, K, step = op(params, damping=1e-3)
    assert jax.tree_util.tree_structure(step) == jax.tree_util.tree_structure(params)
    assert K.shape == (6, 6) and bool(jnp.all(jnp.isfinite(r)))
    with pytest.raises(ValueError, match="only autodiff_rows"):
        compile_residual(fourier, [(x, res)], example_params=params, force="projected_streams")


def test_layer_list_models_return_the_layer_step() -> None:
    params = _init([D, 6, 1], seed=9)
    layers = [(W.T, b, "tanh") for W, b in params[:-1]] + [(params[-1][0].T, params[-1][1], None)]
    x = _blocks()[0][0]
    res = Residual(lambda q: q.lap[:, 0] + q.u[:, 0], ["value", "lap"])
    op = compile_residual(layers, [(x, res)], quantity="gauss_newton", force="projected_streams")
    r, K, step = op(layers, damping=1e-2)
    assert (
        len(step) == 2
        and step[0][0].shape == layers[0][0].shape
        and step[0][1].shape == layers[0][1].shape
    )


def test_input_validation() -> None:
    params = _init([D, 6, 1], seed=10)
    x = _blocks()[0][0]
    with pytest.raises(TypeError, match="Residual"):
        compile_residual(mlp, [(x, lambda *a: a[0])], example_params=params)
    with pytest.raises(ValueError, match="example_params"):
        compile_residual(mlp, [(x, Residual(lambda q: q.u[:, 0], ["value"]))])
    with pytest.raises(ValueError, match="quantity"):
        compile_residual(
            mlp,
            [(x, Residual(lambda q: q.u[:, 0], ["value"]))],
            example_params=params,
            quantity="hessian",
        )
    op = compile_residual(
        mlp, [(x, Residual(lambda q: q.u[:, 0], ["value"]))], example_params=params
    )
    with pytest.raises(ValueError, match="damping"):
        op(params)
    wrapped = legacy_projected(lambda u, gs, lap, dv, z: u[:, 0], order=0)
    assert compile_residual(
        mlp, [(x, wrapped)], example_params=params, quantity="gram"
    ).algorithm in (
        "projected_streams",
        "full_streams",
        "autodiff_rows",
    )

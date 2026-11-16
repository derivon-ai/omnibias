# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Models that are not dense chains: literal-scale folding (SIREN) and the input-product subspace programs."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import compile_operator, match_dense_chain  # noqa: E402

D = 24
H = 5


def _x(batch=4, dim=D, seed=1):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((batch, dim)))


def _reference(fn, params, x, power=1, A=None):
    """``tr(A hess)`` (``power`` times) per point by nested dense autodiff."""

    def point(z):
        return fn(params, z[None])[0]

    g = point
    for _ in range(power):

        def lap(z, g=g):
            H_ = jax.hessian(g)(z)
            return jnp.trace(H_ if A is None else A @ H_)

        g = lap
    return jax.vmap(g)(x)


# ------------------------------------------------------------------------------------------------- models


def _siren(w0=7.0):
    def model(params, x):
        a = x
        for k, (W, b) in enumerate(params[:-1]):
            z = a @ W + b
            a = jnp.sin(w0 * z) if k == 0 else jnp.sin(z)
        W, b = params[-1]
        return (a @ W + b)[..., 0]

    return model


def _siren_params(widths, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (
            jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
            jnp.asarray(0.1 * rng.standard_normal(b)),
        )
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    ]


def _residual(params, x):
    (W1, b1), *blocks, (Wo, bo) = params
    h = jnp.tanh(x @ W1 + b1)
    for W, b in blocks:
        h = h + jnp.tanh(h @ W + b)
    return (h @ Wo + bo)[..., 0]


def _residual_params(seed=0, d=D, h=H, blocks=2):
    rng = np.random.default_rng(seed)
    mk = lambda a, b: (  # noqa: E731
        jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
        jnp.asarray(0.1 * rng.standard_normal(b)),
    )
    return [mk(d, h), *[mk(h, h) for _ in range(blocks)], mk(h, 1)]


def _gated(params, x):
    """PirateNet-style: one Fourier embedding feeds two gates and a stack of gated blocks."""
    B, (Wu, bu), (Wv, bv), (W1, b1), (W2, b2), (Wo, bo) = params
    phi = jnp.concatenate([jnp.sin(x @ B), jnp.cos(x @ B)], axis=-1)
    u, v = jnp.tanh(phi @ Wu + bu), jnp.tanh(phi @ Wv + bv)
    h = phi
    for W, b in ((W1, b1), (W2, b2)):
        f = jnp.tanh(h @ W + b)
        h = f * u + (1.0 - f) * v
    return (h @ Wo + bo)[..., 0]


def _gated_params(seed=0, d=D, m=4, h=6):
    rng = np.random.default_rng(seed)
    mk = lambda a, b: (  # noqa: E731
        jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
        jnp.asarray(0.1 * rng.standard_normal(b)),
    )
    return (
        jnp.asarray(rng.standard_normal((d, m))),
        mk(2 * m, h),
        mk(2 * m, h),
        mk(2 * m, h),
        mk(h, h),
        mk(h, 1),
    )


# ------------------------------------------------------------------------------- literal scale folding


def test_siren_scale_is_folded_into_the_chain() -> None:
    params, x = _siren_params([D, 6, 6, 1]), _x()
    match, why = match_dense_chain(_siren(), (params, x))
    assert why is None and match is not None
    assert [lm.weight_scale for lm in match.layers] == [7.0, 1.0, 1.0]
    assert [lm.bias_scale for lm in match.layers] == [7.0, 1.0, 1.0]
    op = compile_operator(_siren(), (params, x), operator="laplacian")
    assert op.matched and op.algorithm in {"projected", "forward_laplacian"}
    np.testing.assert_allclose(
        np.asarray(op(params, x)),
        np.asarray(_reference(_siren(), params, x)),
        rtol=1e-9,
        atol=1e-9,
    )


def test_scale_between_product_and_bias_scales_only_the_weight() -> None:
    def model(params, x):
        (W1, b1), (Wo, bo) = params
        return (jnp.sin(3.0 * (x @ W1) + b1) @ Wo + bo)[..., 0]

    params, x = _siren_params([D, 6, 1]), _x()
    match, why = match_dense_chain(model, (params, x))
    assert why is None and match is not None
    assert match.layers[0].weight_scale == 3.0 and match.layers[0].bias_scale == 1.0
    op = compile_operator(model, (params, x), operator="laplacian")
    np.testing.assert_allclose(
        np.asarray(op(params, x)), np.asarray(_reference(model, params, x)), rtol=1e-9, atol=1e-9
    )


def test_scale_that_is_not_a_literal_is_refused() -> None:
    def model(params, x):
        (W1, b1), (Wo, bo) = params
        return (jnp.tanh((x @ W1 + b1) * jnp.sum(b1)) @ Wo + bo)[..., 0]

    params, x = _siren_params([D, 6, 1]), _x()
    match, why = match_dense_chain(model, (params, x))
    assert match is None and why is not None


# --------------------------------------------------------------------------------- input-product subspace


@pytest.mark.parametrize("model, make", [(_residual, _residual_params), (_gated, _gated_params)])
def test_non_chain_models_reduce_the_directions(model, make) -> None:
    params, x = make(), _x()
    op = compile_operator(model, (params, x), operator="laplacian")
    assert not op.matched
    assert op.algorithm == "subspace_jet"
    info = op.explain()
    assert info["input_products"] is not None and info["input_products"]["rows"] < D
    assert [c["name"] for c in info["candidates"]][:2] == ["subspace_jet", "subspace_nested"]
    ref = _reference(model, params, x)
    np.testing.assert_allclose(np.asarray(op(params, x)), np.asarray(ref), rtol=1e-9, atol=1e-9)
    forced = compile_operator(model, (params, x), operator="laplacian", force="taylor_jet")
    np.testing.assert_allclose(np.asarray(forced(params, x)), np.asarray(ref), rtol=1e-9, atol=1e-9)


def test_subspace_nested_serves_the_biharmonic() -> None:
    params, x = _residual_params(d=10, h=3, blocks=1), _x(batch=2, dim=10)
    op = compile_operator(_residual, (params, x), operator="laplacian", power=2)
    assert op.algorithm == "subspace_nested"
    np.testing.assert_allclose(
        np.asarray(op(params, x)),
        np.asarray(_reference(_residual, params, x, power=2)),
        rtol=1e-8,
        atol=1e-8,
    )


def test_subspace_with_a_metric_factor() -> None:
    params, x = _residual_params(), _x()
    S = jnp.asarray(np.random.default_rng(3).standard_normal((D, 40)) / 6.0)
    op = compile_operator(_residual, (params, x), operator="laplacian", diffusion=S)
    assert op.algorithm == "subspace_jet"
    np.testing.assert_allclose(
        np.asarray(op(params, x)),
        np.asarray(_reference(_residual, params, x, A=S @ S.T)),
        rtol=1e-9,
        atol=1e-9,
    )


def test_subspace_parameter_gradient_matches_the_full_trace() -> None:
    params, x = _residual_params(), _x()
    op = compile_operator(_residual, (params, x), operator="laplacian")
    ref_op = compile_operator(_residual, (params, x), operator="laplacian", force="taylor_jet")

    def loss(p, operator):
        return jnp.sum(operator(p, x) ** 2)

    got = jax.grad(lambda p: loss(p, op))(params)
    want = jax.grad(lambda p: loss(p, ref_op))(params)
    for (gW, gb), (wW, wb) in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(gW), np.asarray(wW), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(np.asarray(gb), np.asarray(wb), rtol=1e-8, atol=1e-10)


def test_subspace_is_refused_when_the_input_is_read_elsewhere() -> None:
    def model(params, x):
        return _residual(params, x) + 0.3 * x[..., 0] ** 2

    params, x = _residual_params(), _x()
    op = compile_operator(model, (params, x), operator="laplacian")
    assert op.algorithm == "taylor_jet"
    refused = {r["algorithm"]: r["reason"] for r in op.explain()["refused"]}
    assert "subspace" in refused and "read by" in refused["subspace"]
    np.testing.assert_allclose(
        np.asarray(op(params, x)), np.asarray(_reference(model, params, x)), rtol=1e-9, atol=1e-9
    )


def test_subspace_is_refused_without_a_reduction() -> None:
    params, x = _residual_params(d=4, h=6), _x(dim=4)
    op = compile_operator(_residual, (params, x), operator="laplacian")
    assert op.algorithm == "taylor_jet"
    refused = {r["algorithm"]: r["reason"] for r in op.explain()["refused"]}
    assert "no reduction" in refused["subspace"]


def test_a_bound_model_reduces_too() -> None:
    params = _residual_params()
    x = _x()

    def bound(z):
        return _residual(params, z)

    op = compile_operator(bound, (x,), operator="laplacian", params_argnum=None)
    assert op.algorithm == "subspace_jet"
    np.testing.assert_allclose(
        np.asarray(op(x)), np.asarray(_reference(_residual, params, x)), rtol=1e-9, atol=1e-9
    )


# ------------------------------------------------------------------------------------------ forcing and retrying


def test_a_forced_program_that_cannot_run_is_an_error_and_an_unforced_one_is_retried(monkeypatch) -> None:
    from omnibias.jax.experimental import compiler

    def broken(ctx):
        raise RuntimeError("out of memory")

    params, x = _residual_params(), _x()
    monkeypatch.setitem(compiler._EXECUTORS, ("laplacian", "taylor_jet"), broken)
    with pytest.raises(ValueError, match=r"no executor could run this model.*taylor_jet"):
        compile_operator(_residual, (params, x), operator="laplacian", force="taylor_jet")

    monkeypatch.setitem(compiler._EXECUTORS, ("laplacian", "subspace_jet"), broken)
    op = compile_operator(_residual, (params, x), operator="laplacian")
    assert op.algorithm == "subspace_nested" and op.explain()["mode"].endswith("+retry")

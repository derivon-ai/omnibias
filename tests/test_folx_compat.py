# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``omnibias.jax.folx_compat.forward_laplacian``: the fields, layout and values of ``folx.forward_laplacian``."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from omnibias.jax.folx_compat import FwdJacobian, FwdLaplArray, forward_laplacian  # noqa: E402


def _mlp(D=10, H=7, seed=0):
    rng = np.random.default_rng(seed)
    W1 = jnp.asarray(rng.standard_normal((D, H)) / 3)
    b1 = jnp.asarray(rng.standard_normal(H) * 0.1)
    W2 = jnp.asarray(rng.standard_normal((H, H)) / 3)
    w3 = jnp.asarray(rng.standard_normal(H) / 3)

    def f(x):
        h = jnp.tanh(x @ W1 + b1)
        return jnp.tanh(h @ W2) @ w3

    return f, jnp.asarray(rng.standard_normal(D))


def _dense(f, x):
    return f(x), jax.grad(f)(x), jnp.trace(jax.hessian(f)(x))


def test_scalar_chain_matches_value_gradient_and_dense_laplacian_and_is_planned():
    f, x = _mlp()
    wrapped = forward_laplacian(f)
    out = wrapped(x)
    value, grad, lap = _dense(f, x)
    assert isinstance(out, FwdLaplArray) and isinstance(out.jacobian, FwdJacobian)
    np.testing.assert_allclose(out.x, value, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(out.dense_jacobian, grad, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(out.laplacian, lap, rtol=1e-11, atol=1e-12)
    assert out.dense_jacobian.shape == (x.size,) and not out.is_jacobian_weak
    plan = wrapped.explain(x)
    assert plan is not None and plan["matched_patterns"] == ["dense_chain"]


def test_several_arguments_pytree_outputs_and_weights_follow_the_folx_convention():
    def g(a, b):
        return {"v": jnp.stack([jnp.sum(jnp.sin(a) * b), jnp.sum(a**3)]), "s": jnp.sum(b**2)}

    rng = np.random.default_rng(1)
    a, b = jnp.asarray(rng.standard_normal(3)), jnp.asarray(rng.standard_normal(3))
    weights = (2.0, 0.5)
    out = forward_laplacian(g)(a, b, weights=weights)
    wvec = jnp.concatenate([jnp.full(3, 2.0), jnp.full(3, 0.5)])

    def flat(z):
        res = g(a + wvec[:3] * z[:3], b + wvec[3:] * z[3:])
        return res

    z0 = jnp.zeros(6)
    jac = jax.jacfwd(flat)(z0)
    hess = jax.hessian(flat)(z0)
    for key in ("v", "s"):
        np.testing.assert_allclose(out[key].x, g(a, b)[key], rtol=1e-13)
        np.testing.assert_allclose(
            out[key].dense_jacobian, jnp.moveaxis(jac[key], -1, 0), rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            out[key].laplacian, jnp.trace(hess[key], axis1=-2, axis2=-1), rtol=1e-12
        )
    assert out["v"].dense_jacobian.shape == (6, 2)
    assert forward_laplacian(g).explain(a, b) is None  # several arguments take the generic path


def test_the_laplacian_is_differentiable_through_a_closed_over_parameter_and_under_jit():
    f, x = _mlp()

    def lap(scale):
        return forward_laplacian(lambda z: f(z) * scale)(x).laplacian

    want = jnp.trace(jax.hessian(f)(x))
    np.testing.assert_allclose(lap(1.3), 1.3 * want, rtol=1e-11)
    np.testing.assert_allclose(jax.grad(lap)(1.3), want, rtol=1e-11)
    np.testing.assert_allclose(jax.jit(lap)(1.3), 1.3 * want, rtol=1e-11)


def test_vmapped_calls_are_planned_too_and_match_the_dense_laplacian():
    f, x = _mlp()
    X = jnp.stack([x, 0.5 * x, -x])
    wrapped = forward_laplacian(f)
    out = jax.vmap(wrapped)(X)
    want = jax.vmap(lambda z: jnp.trace(jax.hessian(f)(z)))(X)
    np.testing.assert_allclose(out.laplacian, want, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(out.dense_jacobian, jax.vmap(jax.grad(f))(X), rtol=1e-12, atol=1e-13)
    assert out.dense_jacobian.shape == (3, x.size)  # vmap puts the batch first, as for folx
    assert wrapped.explain(x)["matched_patterns"] == ["dense_chain"]
    np.testing.assert_allclose(
        jax.jit(jax.vmap(wrapped))(X).laplacian, want, rtol=1e-11, atol=1e-12
    )


def test_vmap_under_jit_builds_one_plan_for_the_whole_batch(monkeypatch):
    import omnibias.jax.experimental.compiler as compiler

    built = []
    real = compiler.compile_operator

    def spy(model, example_args, **kwargs):
        built.append(tuple(example_args[0].shape))
        return real(model, example_args, **kwargs)

    monkeypatch.setattr(compiler, "compile_operator", spy)
    f, x = _mlp()
    X = jnp.stack([x, 0.5 * x, -x, 2.0 * x])
    out = jax.jit(jax.vmap(forward_laplacian(f)))(X)
    want = jax.vmap(lambda z: jnp.trace(jax.hessian(f)(z)))(X)
    np.testing.assert_allclose(out.laplacian, want, rtol=1e-11, atol=1e-12)
    assert (
        4,
        x.size,
    ) in built, built  # a plan for the batch the enclosing vmap carries, not only one per point
    assert (1, x.size) not in built, built  # and no plan of one point that nothing runs


def test_vmapped_calls_fall_back_to_the_plan_of_one_point_when_the_batch_cannot_be_planned(
    monkeypatch,
):
    import omnibias.jax.experimental.compiler as compiler

    built = []
    real = compiler.compile_operator

    def spy(model, example_args, **kwargs):
        shape = tuple(example_args[0].shape)
        built.append(shape)
        if shape[0] > 1:
            raise ValueError("no plan for a batch")
        return real(model, example_args, **kwargs)

    monkeypatch.setattr(compiler, "compile_operator", spy)
    f, x = _mlp()
    X = jnp.stack([x, 0.5 * x, -x])
    wrapped = forward_laplacian(f)
    out = jax.vmap(wrapped)(X)
    want = jax.vmap(lambda z: jnp.trace(jax.hessian(f)(z)))(X)
    np.testing.assert_allclose(out.laplacian, want, rtol=1e-11, atol=1e-12)
    assert built == [(3, x.size), (1, x.size)], built


def test_explain_after_vmapped_calls_names_the_plan_without_a_second_call_path():
    f, x = _mlp()
    wrapped = forward_laplacian(f)
    jax.vmap(wrapped)(jnp.stack([x, -x]))
    assert wrapped.explain(x)["matched_patterns"] == ["dense_chain"]


def test_non_scalar_output_signature_arguments_and_disable_jit_are_accepted():
    f, x = _mlp()
    vec = lambda z: jnp.stack([f(z), 2.0 * f(z)])  # noqa: E731
    out = forward_laplacian(vec, sparsity_threshold=0.6, disable_jit=True)(x)
    lap = jnp.trace(jax.hessian(f)(x))
    np.testing.assert_allclose(out.laplacian, jnp.stack([lap, 2.0 * lap]), rtol=1e-11)
    assert out.dense_jacobian.shape == (x.size, 2)
    scalar = forward_laplacian(f, disable_jit=True)(x)
    np.testing.assert_allclose(scalar.laplacian, lap, rtol=1e-11)


def test_matches_folx_when_it_is_installed():
    folx = pytest.importorskip("folx")
    f, x = _mlp()
    ours, theirs = forward_laplacian(f)(x), folx.forward_laplacian(f)(x)
    np.testing.assert_allclose(ours.x, theirs.x, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(ours.dense_jacobian, theirs.dense_jacobian, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(ours.laplacian, theirs.laplacian, rtol=1e-11, atol=1e-12)

    def g(a, b):
        return jnp.stack([jnp.sum(jnp.sin(a) * b), jnp.sum(a**3)]), jnp.sum(b**2)

    rng = np.random.default_rng(2)
    a, b = jnp.asarray(rng.standard_normal(3)), jnp.asarray(rng.standard_normal(3))
    mine = forward_laplacian(g)(a, b, weights=(2.0, 0.5))
    ref = folx.forward_laplacian(g)(a, b, weights=(2.0, 0.5))
    for u, v in zip(mine, ref, strict=True):
        np.testing.assert_allclose(u.x, v.x, rtol=1e-13)
        np.testing.assert_allclose(u.dense_jacobian, v.dense_jacobian, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(u.laplacian, v.laplacian, rtol=1e-12, atol=1e-13)

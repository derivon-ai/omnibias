# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.jax.experimental.fused_closed_ridge import (  # noqa: E402
    fused_closed_ridge_residual_loss,
    fused_closed_ridge_value_and_grad,
)
from omnibias.jax.experimental.tanh_ridge import RidgeTerm  # noqa: E402

# Compiled Pallas Triton kernels on a GPU backend, the Pallas interpreter everywhere else.
INTERPRET = jax.default_backend() != "gpu"
ACTS = {"sin": jnp.sin, "cos": jnp.cos, "exp": jnp.exp}
TERMS = [
    (RidgeTerm("laplace", 1),),
    (
        RidgeTerm("laplace", 2, 0.5),
        RidgeTerm("coordinate", 3, 1.0, 0),
        RidgeTerm("coordinate", 1, 0.3, 1),
    ),
    (RidgeTerm("coordinate", 8, 1.0, 1),),
]
# (D, H, B): one-pass kernel, one-pass with several unit blocks, split kernels with ragged sizes
SHAPES = [(2, 7, 9), (3, 40, 37), (6, 21, 70), (6, 5, 3)]


def _data(D: int, H: int, B: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.standard_normal((H, D)) * 0.6)
    b = jnp.asarray(rng.standard_normal(H) * 0.3)
    c = jnp.asarray(rng.standard_normal(H) / np.sqrt(H))
    x = jnp.asarray(rng.standard_normal((B, D)) * 0.5)
    f = jnp.asarray(rng.standard_normal(B))
    return W, b, c, x, f


def _reference_loss(act, terms):
    """Direct sum over units with sigma^(m) from nested autodiff of the scalar activation."""

    def deriv(m):
        g = ACTS[act]
        for _ in range(m):
            g = jax.grad(g)
        return jnp.vectorize(g)

    def loss(W, b, c, x, f):
        z = x @ W.T + b
        out = -f
        for kind, order, coeff, axis in terms:
            if kind == "laplace":
                g, m = coeff * jnp.sum(W * W, axis=1) ** order, 2 * order
            else:
                g, m = coeff * W[:, axis] ** order, order
            out = out + deriv(m)(z) @ (g * c)
        return 0.5 * jnp.mean(out * out)

    return loss


@pytest.mark.parametrize("act", sorted(ACTS))
@pytest.mark.parametrize("terms", TERMS)
@pytest.mark.parametrize("shape", SHAPES)
def test_value_and_gradient_match_independent_reference(act, terms, shape):
    W, b, c, x, f = _data(*shape)
    v, grads = fused_closed_ridge_value_and_grad(W, b, c, x, f, terms, act, interpret=INTERPRET)
    vr, ref = jax.value_and_grad(_reference_loss(act, terms), argnums=(0, 1, 2))(W, b, c, x, f)
    np.testing.assert_allclose(float(v), float(vr), rtol=1e-12)
    for got, want in zip(grads, ref, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-9, atol=1e-11)


def test_differentiable_loss_scales_its_cotangent():
    W, b, c, x, f = _data(6, 21, 70)
    terms = TERMS[1]

    def scaled(W, b, c):
        return 3.0 * fused_closed_ridge_residual_loss(W, b, c, x, f, terms, interpret=INTERPRET)

    got = jax.grad(scaled, argnums=(0, 1, 2))(W, b, c)
    ref = jax.grad(
        lambda W, b, c: 3.0 * _reference_loss("sin", terms)(W, b, c, x, f), argnums=(0, 1, 2)
    )(W, b, c)
    for a, r in zip(got, ref, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(r), rtol=1e-9, atol=1e-11)


def test_jit_matches_eager():
    W, b, c, x, f = _data(3, 40, 37)
    terms = TERMS[1]

    def run(W, b, c, x, f):
        return fused_closed_ridge_value_and_grad(W, b, c, x, f, terms, interpret=INTERPRET)

    eager = run(W, b, c, x, f)
    jitted = jax.jit(run)(W, b, c, x, f)
    for a, r in zip(
        jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(jitted), strict=True
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(r), rtol=1e-13, atol=1e-15)


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="refusal applies to non-GPU backends")
def test_non_gpu_backend_without_interpreter_is_refused():
    W, b, c, x, f = _data(2, 7, 9)
    with pytest.raises(NotImplementedError):
        fused_closed_ridge_value_and_grad(W, b, c, x, f, TERMS[0])


def test_float32_is_refused():
    W, b, c, x, f = (a.astype(jnp.float32) for a in _data(2, 7, 9))
    with pytest.raises(NotImplementedError):
        fused_closed_ridge_value_and_grad(W, b, c, x, f, TERMS[0], interpret=INTERPRET)


@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
def test_non_closed_activations_are_refused(act):
    W, b, c, x, f = _data(2, 7, 9)
    with pytest.raises(ValueError):
        fused_closed_ridge_value_and_grad(W, b, c, x, f, TERMS[0], act, interpret=INTERPRET)

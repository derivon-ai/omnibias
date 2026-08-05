# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation suite for the non-elementwise jet primitives (jax).

Twin of ``packages/omnibias-torch/tests/test_jet_attention.py``. ``compose_jet_mv``
only reaches *elementwise* maps; these primitives extend the reachable class to
rational and coupled ones -- ``1/u``, ``exp``, softmax over an axis, and the
attention block built from them -- so each is checked against nested ``jax.jacfwd``
plus the structural invariants (partition of unity at every order, shift
invariance, convex-hull containment) and against
:func:`omnibias.hopfield.jax.ops.attention` for the value.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.multi_index import multi_indices  # noqa: E402
from omnibias.jax.jet_mv import (  # noqa: E402
    identity_jet,
    jet_attention,
    jet_exp,
    jet_multiply,
    jet_partials,
    jet_reciprocal,
    jet_softmax,
    mlp_jet_mv,
)

DIM = 3
ORDER = 3


@pytest.fixture
def memory():
    rng = np.random.default_rng(7)
    return {
        "W": jnp.asarray(rng.normal(scale=0.7, size=(4, DIM))),
        "b": jnp.asarray(rng.normal(scale=0.3, size=(4,))),
        "K": jnp.asarray(rng.normal(scale=0.8, size=(5, 4))),
        "V": jnp.asarray(rng.normal(scale=1.1, size=(5, 2))),
        "x0": jnp.asarray(rng.normal(size=(DIM,))),
    }


def _ad_partials(f, x0, order: int = ORDER):
    """``{alpha: D^alpha f(x0)}`` by nested forward-mode AD."""
    out = {}
    derivs = [f]
    for _ in range(order):
        derivs.append(jax.jacfwd(derivs[-1]))
    for alpha in multi_indices(DIM, order):
        k = sum(alpha)
        if k == 0:
            out[alpha] = f(x0)
            continue
        axes = [i for i, a in enumerate(alpha) for _ in range(a)]
        val = derivs[k](x0)
        for ax in reversed(axes):
            val = val[..., ax]
        out[alpha] = val
    return out


def _close(a, b, tol: float = 1e-11) -> bool:
    return bool(jnp.allclose(a, b, rtol=tol, atol=tol))


# ----------------------------- reciprocal / exp -----------------------------


def test_reciprocal_matches_autodiff(memory) -> None:
    W, b, x0 = memory["W"], memory["b"], memory["x0"]

    def f(x):
        return 1.0 / (2.5 + jnp.tanh(W @ x + b))

    jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    shifted = jnp.concatenate([(jet[0] + 2.5)[None], jet[1:]], axis=0)
    got = jet_partials(jet_reciprocal(shifted, DIM, ORDER), DIM, ORDER)
    for alpha, value in _ad_partials(f, x0).items():
        assert _close(got[alpha], value), alpha


def test_reciprocal_times_original_is_one(memory) -> None:
    """The defining identity, at *every* order: ``u * (1/u) = 1``."""
    W, b, x0 = memory["W"], memory["b"], memory["x0"]
    jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    shifted = jnp.concatenate([(jet[0] + 2.5)[None], jet[1:]], axis=0)
    product = jet_multiply(shifted, jet_reciprocal(shifted, DIM, ORDER), DIM, ORDER)
    assert _close(product[0], jnp.ones_like(product[0]), 1e-14)
    assert float(jnp.abs(product[1:]).max()) < 1e-13


def test_exp_matches_autodiff(memory) -> None:
    W, b, x0 = memory["W"], memory["b"], memory["x0"]

    def f(x):
        return jnp.exp(jnp.tanh(W @ x + b))

    jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    got = jet_partials(jet_exp(jet, DIM, ORDER), DIM, ORDER)
    for alpha, value in _ad_partials(f, x0).items():
        assert _close(got[alpha], value), alpha


# --------------------------------- softmax ----------------------------------


def test_softmax_matches_autodiff(memory) -> None:
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]

    def f(x):
        return jax.nn.softmax(K @ jnp.tanh(W @ x + b), axis=-1)

    scores = mlp_jet_mv(x0, [(W, b, "tanh"), (K, None, None)], ORDER)
    got = jet_partials(jet_softmax(scores, DIM, ORDER), DIM, ORDER)
    for alpha, value in _ad_partials(f, x0).items():
        assert _close(got[alpha], value), alpha


def test_softmax_is_a_partition_of_unity_at_every_order(memory) -> None:
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]
    scores = mlp_jet_mv(x0, [(W, b, "tanh"), (K, None, None)], ORDER)
    rows = jnp.sum(jet_softmax(scores, DIM, ORDER), axis=-1)
    assert abs(float(rows[0]) - 1.0) < 1e-14
    assert float(jnp.abs(rows[1:]).max()) < 1e-14


def test_softmax_is_shift_invariant(memory) -> None:
    """A constant added to every score changes nothing -- including the jet."""
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]
    scores = mlp_jet_mv(x0, [(W, b, "tanh"), (K, None, None)], ORDER)
    bumped = jnp.concatenate([(scores[0] + 40.0)[None], scores[1:]], axis=0)
    assert _close(
        jet_softmax(scores, DIM, ORDER), jet_softmax(bumped, DIM, ORDER), 1e-14
    )


def test_softmax_survives_scores_that_overflow_a_naive_exp(memory) -> None:
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]
    scores = mlp_jet_mv(x0, [(W, b, "tanh"), (K * 400.0, None, None)], ORDER)
    p = jet_softmax(scores, DIM, ORDER)
    assert bool(jnp.isfinite(p).all())
    assert abs(float(jnp.sum(p[0])) - 1.0) < 1e-14


def test_softmax_rejects_a_scalar_jet(memory) -> None:
    x0 = memory["x0"]
    with pytest.raises(ValueError, match="trailing softmax axis"):
        jet_softmax(identity_jet(x0, ORDER)[:, 0], DIM, ORDER)


# -------------------------------- attention ---------------------------------


def test_attention_matches_autodiff(memory) -> None:
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))
    beta = 1.7

    def f(x):
        q = jnp.tanh(W @ x + b)
        return jax.nn.softmax(beta * (K @ q), axis=-1) @ V

    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    got = jet_partials(jet_attention(q_jet, K, V, DIM, ORDER, beta=beta), DIM, ORDER)
    for alpha, value in _ad_partials(f, x0).items():
        assert _close(got[alpha], value), alpha


def test_attention_value_matches_omnibias_hopfield(memory) -> None:
    """The block *is* hopfield attention; this module only adds ``d/dx``."""
    hopfield = pytest.importorskip("omnibias.hopfield.jax.ops")
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))
    beta = 0.8
    q = jnp.tanh(W @ x0 + b)
    reference = hopfield.attention(q[None], K, V, beta=beta)[0]
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    got = jet_attention(q_jet, K, V, DIM, ORDER, beta=beta)[0]
    assert _close(got, reference, 1e-14)


def test_attention_output_lies_in_the_convex_hull_of_the_values(memory) -> None:
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    out = jet_attention(q_jet, K, V, DIM, ORDER)[0]
    assert bool((out >= jnp.min(V, axis=0) - 1e-12).all())
    assert bool((out <= jnp.max(V, axis=0) + 1e-12).all())


def test_attention_temperature_is_differentiable(memory) -> None:
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)

    def total(beta):
        return jnp.sum(jet_attention(q_jet, K, V, DIM, ORDER, beta=beta))

    assert abs(float(jax.grad(total)(1.3))) > 0.0


def test_attention_sharpens_towards_one_slot_as_beta_grows(memory) -> None:
    """Temperature collapse (the feasibility sense): the mixture hardens."""
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    eye = jnp.eye(K.shape[0], dtype=K.dtype)
    peaks = [
        float(jnp.max(jet_attention(q_jet, K, eye, DIM, ORDER, beta=beta)[0]))
        for beta in (0.5, 5.0, 50.0)
    ]
    assert peaks[0] < peaks[1] < peaks[2]
    assert peaks[-1] > 0.99


def test_attention_is_jittable(memory) -> None:
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))

    def block(x):
        return jet_attention(
            mlp_jet_mv(x, [(W, b, "tanh")], ORDER), K, V, DIM, ORDER, beta=1.2
        )

    assert _close(jax.jit(block)(x0), block(x0), 1e-14)


@pytest.mark.parametrize(
    ("keys_shape", "values_shape", "match"),
    [
        ((3,), (3, 2), "must be 2-D"),
        ((3, 4), (2, 2), "share the memory axis"),
        ((3, 9), (3, 2), "key width"),
    ],
)
def test_attention_validates_shapes(memory, keys_shape, values_shape, match) -> None:
    W, b, x0 = memory["W"], memory["b"], memory["x0"]
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    with pytest.raises(ValueError, match=match):
        jet_attention(
            q_jet, jnp.zeros(keys_shape), jnp.zeros(values_shape), DIM, ORDER
        )


def test_row_count_is_validated(memory) -> None:
    jet = identity_jet(memory["x0"], ORDER)
    with pytest.raises(ValueError, match="requires"):
        jet_reciprocal(jet, DIM, ORDER + 1)
    with pytest.raises(ValueError, match="requires"):
        jet_softmax(jet, DIM, ORDER + 1)

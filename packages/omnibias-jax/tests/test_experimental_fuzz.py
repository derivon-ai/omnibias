# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Property-based agreement of every ``compile_operator`` executor on random dense chains.

The properties are the ones the planner relies on: whatever it ranks first must compute the
same operator as the generic engines, and each of them must agree with a dense oracle. The
default profile is small and deterministic; set ``OMNIBIAS_FUZZ_EXAMPLES`` and
``OMNIBIAS_FUZZ_NIGHTLY=1`` for a larger randomised run. The multiprecision oracle is marked
``slow``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

jax = pytest.importorskip("jax")
hypothesis = pytest.importorskip("hypothesis")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from omnibias.jax.experimental import compile_operator  # noqa: E402

_EXAMPLES = int(os.environ.get("OMNIBIAS_FUZZ_EXAMPLES", "12"))
_NIGHTLY = os.environ.get("OMNIBIAS_FUZZ_NIGHTLY") is not None
_SETTINGS = settings(
    max_examples=_EXAMPLES,
    deadline=None,
    derandomize=not _NIGHTLY,
    database=None,
    suppress_health_check=list(HealthCheck),
)

_ACTIVATIONS = {
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
    "softplus": jax.nn.softplus,
    "sigmoid": jax.nn.sigmoid,
    "sin": jnp.sin,
}


# ``jax.experimental.jet`` raises ``UnexpectedTracerError`` on ``jnp.logaddexp`` (a ``custom_jvp``, which ``jax.nn.softplus`` calls),
# standalone and without omnibias (JAX 0.10.0 and 0.10.2; reproducer in ``omnibias_upstream/jax-jet-logaddexp``).  The planner lists
# ``taylor_jet`` for such a model, refuses it when it cannot trace it, and falls back; only ``force=`` turns the refusal into an error.
_JET_UNTRACEABLE = {"softplus"}


@dataclass(frozen=True)
class Chain:
    dim: int
    hidden: tuple[int, ...]
    activation: str
    batch: int
    seed: int
    weight_scale: float
    bias_scale: float

    def params(self, out: int = 1):
        rng = np.random.default_rng(self.seed)
        widths = [self.dim, *self.hidden, out]
        layers = []
        for a, b in zip(widths[:-1], widths[1:], strict=True):
            layers.append(
                (
                    jnp.asarray(self.weight_scale * rng.standard_normal((a, b)) / np.sqrt(a)),
                    jnp.asarray(self.bias_scale * rng.standard_normal(b)),
                )
            )
        return layers

    def points(self):
        return jnp.asarray(
            np.random.default_rng(self.seed + 1).standard_normal((self.batch, self.dim))
        )

    def model(self, squeeze: bool = True):
        act = _ACTIVATIONS[self.activation]

        def model(params, x):
            a = x
            for W, b in params[:-1]:
                a = act(a @ W + b)
            W, b = params[-1]
            out = a @ W + b
            return out[..., 0] if squeeze and out.shape[-1] == 1 else out

        return model


@st.composite
def chains(draw, max_dim: int = 7):
    return Chain(
        dim=draw(st.integers(1, max_dim)),
        hidden=tuple(draw(st.lists(st.integers(1, 9), min_size=1, max_size=3))),
        activation=draw(st.sampled_from(sorted(_ACTIVATIONS))),
        batch=draw(st.integers(1, 4)),
        seed=draw(st.integers(0, 2**16)),
        weight_scale=draw(st.sampled_from([0.3, 1.0, 1.8])),
        bias_scale=draw(st.sampled_from([0.0, 0.3, 1.0])),
    )


def _dense_laplacian(chain: Chain, params, x):
    model = chain.model()

    def one(xi):
        return jnp.trace(jax.hessian(lambda z: model(params, z[None])[0])(xi))

    return jax.vmap(one)(x)


def _close(got, want, rtol=1e-8):
    got, want = np.asarray(got), np.asarray(want)
    scale = max(1.0, float(np.abs(want).max()))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol * scale)


@_SETTINGS
@given(chains())
def test_every_feasible_laplacian_executor_matches_the_dense_hessian(chain: Chain) -> None:
    params, x = chain.params(), chain.points()
    want = _dense_laplacian(chain, params, x)
    default = compile_operator(chain.model(), (params, x))
    _close(default(params, x), want)
    for cand in default.explain()["candidates"]:
        try:
            op = compile_operator(chain.model(), (params, x), force=cand["name"])
        except ValueError as err:
            # the one tolerated refusal: a jet executor that cannot trace a custom_jvp activation (the default plan above already ran)
            assert cand["name"] == "taylor_jet" and chain.activation in _JET_UNTRACEABLE, err
            assert "executor could not trace this model" in str(err)
            continue
        _close(op(params, x), want)


@_SETTINGS
@given(chains(max_dim=5))
def test_default_dispatch_gradients_match_dense_hessian_gradients(chain: Chain) -> None:
    params, x = chain.params(), chain.points()
    op = compile_operator(chain.model(), (params, x))
    got = jax.grad(lambda p: jnp.sum(op(p, x) ** 2))(params)
    want = jax.grad(lambda p: jnp.sum(_dense_laplacian(chain, p, x) ** 2))(params)
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want), strict=True):
        _close(g, w, rtol=1e-7)


@_SETTINGS
@given(chains(max_dim=4), st.sampled_from([2, 3]))
def test_laplace_powers_agree_between_structured_and_nested_executors(
    chain: Chain, power: int
) -> None:
    params, x = chain.params(), chain.points()
    nested = compile_operator(chain.model(), (params, x), power=power, force="nested_autodiff")
    default = compile_operator(chain.model(), (params, x), power=power)
    _close(default(params, x), nested(params, x), rtol=1e-6)


@_SETTINGS
@given(chains(max_dim=6))
def test_divergence_matches_the_dense_jacobian_trace(chain: Chain) -> None:
    params, x = chain.params(out=chain.dim), chain.points()
    model = chain.model(squeeze=False)
    want = jax.vmap(lambda xi: jnp.trace(jax.jacfwd(lambda z: model(params, z[None])[0])(xi)))(x)
    op = compile_operator(model, (params, x), operator="divergence")
    _close(op(params, x), want, rtol=1e-9)


@_SETTINGS
@given(chains(max_dim=5))
def test_plans_are_stable_and_the_chosen_executor_is_never_a_refused_one(chain: Chain) -> None:
    params, x = chain.params(), chain.points()
    a = compile_operator(chain.model(), (params, x)).explain()
    b = compile_operator(chain.model(), (params, x)).explain()
    assert a["selected_strategy"] == b["selected_strategy"]
    assert a["selected_strategy"] not in {r["algorithm"] for r in a["refused"]}


# ------------------------------------------------------------------------- multiprecision oracle


def _mp_laplacian(chain: Chain, params, x, dps: int = 50) -> np.ndarray:
    """``Delta f`` at each point in 50-digit arithmetic by forward second-order propagation."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = dps

    def act(z):
        if chain.activation == "tanh":
            t = mp.tanh(z)
            return t, 1 - t * t, -2 * t * (1 - t * t)
        if chain.activation == "sin":
            return mp.sin(z), mp.cos(z), -mp.sin(z)
        s = 1 / (1 + mp.exp(-z))
        if chain.activation == "sigmoid":
            d1 = s * (1 - s)
            return s, d1, d1 * (1 - 2 * s)
        if chain.activation == "softplus":
            return mp.log(1 + mp.exp(z)), s, s * (1 - s)
        d1 = s + z * s * (1 - s)  # silu
        d2 = s * (1 - s) * (2 + z * (1 - 2 * s))
        return z * s, d1, d2

    layers = [
        (mp.matrix(np.asarray(W).tolist()), mp.matrix(np.asarray(b).tolist())) for W, b in params
    ]
    out = []
    for xi in np.asarray(x):
        total = mp.mpf(0)
        for i in range(chain.dim):
            a = mp.matrix([mp.mpf(float(v)) for v in xi])
            d1 = mp.matrix([1 if j == i else 0 for j in range(chain.dim)])
            d2 = mp.zeros(chain.dim, 1)
            for k, (W, b) in enumerate(layers):
                z = W.T * a + b
                z1 = W.T * d1
                z2 = W.T * d2
                if k == len(layers) - 1:
                    a, d1, d2 = z, z1, z2
                    break
                n = z.rows
                na, n1, n2 = mp.zeros(n, 1), mp.zeros(n, 1), mp.zeros(n, 1)
                for r in range(n):
                    f0, f1, f2 = act(z[r])
                    na[r], n1[r] = f0, f1 * z1[r]
                    n2[r] = f2 * z1[r] ** 2 + f1 * z2[r]
                a, d1, d2 = na, n1, n2
            total += d2[0]
        out.append(float(total))
    return np.asarray(out)


@pytest.mark.slow
@settings(
    max_examples=max(6, _EXAMPLES // 2),
    deadline=None,
    derandomize=not _NIGHTLY,
    database=None,
    suppress_health_check=list(HealthCheck),
)
@given(chains(max_dim=5))
def test_every_feasible_laplacian_executor_matches_the_multiprecision_oracle(chain: Chain) -> None:
    params, x = chain.params(), chain.points()
    truth = _mp_laplacian(chain, params, x)
    default = compile_operator(chain.model(), (params, x))
    for cand in default.explain()["candidates"]:
        got = compile_operator(chain.model(), (params, x), force=cand["name"])(params, x)
        scale = max(1.0, float(np.abs(truth).max()))
        err = float(np.max(np.abs(np.asarray(got) - truth))) / scale
        assert err < 1e-10, (cand["name"], err)

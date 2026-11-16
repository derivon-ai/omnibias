# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The stream engine serves the extended demand vocabulary; every quantity is checked against nested autodiff."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.demand import Demand  # noqa: E402
from omnibias.jax.experimental import (  # noqa: E402
    QuantitySpec,
    Residual,
    streams_quantities,
    streams_residual,
)

D, C = 3, 2


def _layers(seed=0, hidden=7, depth=2, out=C):
    rng = np.random.default_rng(seed)
    dims = [D] + [hidden] * depth + [out]
    return [
        (
            jnp.asarray(rng.standard_normal((b, a)) / np.sqrt(a)),
            jnp.asarray(0.2 * rng.standard_normal(b)),
            "tanh" if i < depth else None,
        )
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True))
    ]


def _forward(layers, x):
    h = x
    for W, b, spec in layers:
        h = W @ h + b
        if spec is not None:
            h = jnp.tanh(h)
    return h


def _partial(f, alpha):
    """``d^alpha f`` by nested forward-mode derivatives along coordinate axes (the oracle)."""
    for axis, n in enumerate(alpha):
        e = jnp.eye(D)[axis]
        for _ in range(n):
            f = (lambda g, e: lambda x: jax.jvp(g, (x,), (e,))[1])(f, e)
    return f


def _points(B=5, seed=3):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((B, D)) * 0.6)


def _dense(layers, x, fn):
    return jax.vmap(fn)(x)


def test_vocabulary_orders_and_validation() -> None:
    assert Demand.of("hess_form").order == 2
    assert Demand.of("third").order == 3 and Demand.of("lap_grad").order == 3
    assert Demand.of("lap", "fourth").order == 4
    assert Demand.of("lap", "third").to_config() == ["value", "lap", "third"]
    with pytest.raises(ValueError, match="total order 3"):
        QuantitySpec(third=((2, 0, 0),))
    with pytest.raises(ValueError, match="must define"):
        QuantitySpec().check(Demand.of("third"))
    with pytest.raises(ValueError, match="does not read"):
        QuantitySpec(fourth=((4, 0, 0),)).check(Demand.of("lap"))
    with pytest.raises(ValueError, match="length 3"):
        QuantitySpec(mixed=((1, 1),)).check(Demand.of("mixed"), 3)
    assert Demand.of("lap", "bilap").projectable
    assert not Demand.of("third").projectable


def test_old_engines_refuse_new_names_and_point_to_the_stream_engine() -> None:
    from omnibias.jax.experimental import deep_residual_gram, projected_residual_gram

    layers = _layers(1, out=1)
    x = _points(4)
    res = Residual(lambda q: q.third[:, 0, 0], ["third"])
    with pytest.raises(ValueError, match="cannot provide"):
        deep_residual_gram([(x, res)], layers)
    with pytest.raises(ValueError, match="stacked stream engine provides"):
        projected_residual_gram([(x, res)], layers)


def test_hess_form_for_an_indefinite_symmetric_matrix() -> None:
    layers = _layers(2)
    x = _points()
    A = np.array([[1.0, 0.5, -0.3], [0.5, -2.0, 0.7], [-0.3, 0.7, 0.4]])
    res = Residual(lambda q: q.hess_form, ["hess_form"])
    q = streams_quantities(x, layers, res, QuantitySpec(hess_form=A))
    ref = _dense(
        layers,
        x,
        lambda p: jnp.einsum(
            "ij,cij->c", jnp.asarray(A), jax.hessian(lambda z: _forward(layers, z))(p)
        ),
    )
    np.testing.assert_allclose(np.asarray(q.hess_form), np.asarray(ref), rtol=1e-10, atol=1e-12)
    # an A with a zero eigenvalue (rank 2) and the Gram form agree
    A2 = np.outer([1.0, 1.0, 0.0], [1.0, 1.0, 0.0]) - np.outer([0.0, 1.0, 1.0], [0.0, 1.0, 1.0])
    for form in ("auto", "diag", "gram"):
        got = streams_quantities(x, layers, res, QuantitySpec(hess_form=A2), form=form).hess_form
        ref2 = _dense(
            layers,
            x,
            lambda p: jnp.einsum(
                "ij,cij->c", jnp.asarray(A2), jax.hessian(lambda z: _forward(layers, z))(p)
            ),
        )
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref2), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "alphas"),
    [
        ("mixed", ((1, 1, 0), (0, 1, 1), (2, 0, 0))),
        ("third", ((3, 0, 0), (1, 2, 0), (1, 1, 1))),
        ("fourth", ((4, 0, 0), (2, 2, 0), (2, 1, 1))),
    ],
)
def test_partials_match_nested_autodiff(name, alphas) -> None:
    layers = _layers(4)
    x = _points()
    res = Residual(lambda q: getattr(q, name), [name])
    got = getattr(streams_quantities(x, layers, res, QuantitySpec(**{name: alphas})), name)
    assert got.shape == (x.shape[0], len(alphas), C)
    for r, alpha in enumerate(alphas):
        f = _partial(lambda z: _forward(layers, z), alpha)
        ref = jax.vmap(f)(x)
        np.testing.assert_allclose(np.asarray(got[:, r]), np.asarray(ref), rtol=1e-8, atol=1e-10)


def test_laplacian_gradient_and_bilaplacian_share_one_jet_pass() -> None:
    layers = _layers(5)
    x = _points(4)
    res = Residual(lambda q: q.lap_grad.sum(1) + q.bilap, ["lap_grad", "bilap", "third"])
    spec = QuantitySpec(third=((3, 0, 0),))
    q = streams_quantities(x, layers, res, spec)

    def lap(z):
        return jnp.trace(jax.hessian(lambda w: _forward(layers, w))(z), axis1=-2, axis2=-1)

    ref_grad = jax.vmap(jax.jacfwd(lap))(x)  # (B, C, D)
    np.testing.assert_allclose(
        np.asarray(q.lap_grad), np.asarray(jnp.swapaxes(ref_grad, 1, 2)), rtol=1e-8, atol=1e-10
    )
    bilap = jax.vmap(lambda z: jnp.trace(jax.hessian(lap)(z), axis1=-2, axis2=-1))(x)
    np.testing.assert_allclose(np.asarray(q.bilap), np.asarray(bilap), rtol=1e-8, atol=1e-10)
    assert q.third.shape == (4, 1, C)
    assert q.lap is None and q.grad is None  # undeclared quantities stay hidden


def test_grad_metric_and_per_point_drift_directions() -> None:
    layers = _layers(6)
    x = _points(4)
    rng = np.random.default_rng(7)
    V = jnp.asarray(rng.standard_normal((4, D, 2)))  # per-point drift
    res = Residual(lambda q: q.grad_sq + q.lap, ["grad", "grad_sq", "lap", "directional"])
    q = streams_quantities(x, layers, res, directions=V)
    f = lambda z: _forward(layers, z)  # noqa: E731
    jac = jax.vmap(jax.jacfwd(f))(x)  # (B, C, D)
    np.testing.assert_allclose(np.asarray(q.grad), np.asarray(jnp.swapaxes(jac, 1, 2)), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(q.grad_sq), np.asarray((jac**2).sum(-1)), rtol=1e-10)
    lap = jax.vmap(lambda z: jnp.trace(jax.hessian(f)(z), axis1=-2, axis2=-1))(x)
    np.testing.assert_allclose(np.asarray(q.lap), np.asarray(lap), rtol=1e-9, atol=1e-11)
    drift = jnp.einsum("bcd,bdm->bmc", jac, V)
    np.testing.assert_allclose(np.asarray(q.directional), np.asarray(drift), rtol=1e-10)
    # a metric on a coordinate subset: the second-order group is a separate pass
    S = jnp.asarray(rng.standard_normal((2, 2)))
    metric = streams_quantities(
        x, layers, Residual(lambda q: q.lap, ["grad", "lap"]), coordinates=[0, 2], diffusion=S
    )
    A = np.zeros((D, D))
    A[np.ix_([0, 2], [0, 2])] = np.asarray(S @ S.T)
    ref = jax.vmap(lambda z: jnp.einsum("ij,cij->c", jnp.asarray(A), jax.hessian(f)(z)))(x)
    np.testing.assert_allclose(np.asarray(metric.lap), np.asarray(ref), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        np.asarray(metric.grad), np.asarray(jnp.swapaxes(jac, 1, 2)), rtol=1e-10
    )


def test_unbatched_points_return_unbatched_quantities() -> None:
    layers = _layers(8)
    x = _points(1)[0]
    res = Residual(lambda q: q.third, ["third", "lap"])
    q = streams_quantities(x, layers, res, QuantitySpec(third=((3, 0, 0), (0, 2, 1))))
    assert q.u.shape == (C,) and q.lap.shape == (C,) and q.third.shape == (2, C)


def test_kdv_residual_gradients_match_the_dense_oracle() -> None:
    """``u_t + u u_x + 0.0025 u_xxx`` (KdV, x = axis 0, t = axis 1): parameter gradients flow through the streams."""
    base = _layers(9, out=1)
    x = _points(6)
    res = Residual(
        lambda q: q.directional[:, 1, 0]
        + q.u[:, 0] * q.directional[:, 0, 0]
        + 0.0025 * q.third[:, 0, 0],
        ["directional", "third"],
    )
    spec = QuantitySpec(third=((3, 0, 0),))
    V = jnp.eye(D)[:, :2]

    def loss(weights):
        layers = [(W, b, s) for (W, b), (_, _, s) in zip(weights, base, strict=True)]
        r = streams_residual(x, layers, res, spec, directions=V)
        return 0.5 * jnp.sum(r * r)

    def dense_loss(weights):
        layers = [(W, b, s) for (W, b), (_, _, s) in zip(weights, base, strict=True)]
        f = lambda z: _forward(layers, z)[0]  # noqa: E731
        u_xxx = _partial(lambda z: _forward(layers, z), (3, 0, 0))

        def row(z):
            g = jax.grad(f)(z)
            return g[1] + f(z) * g[0] + 0.0025 * u_xxx(z)[0]

        r = jax.vmap(row)(x)
        return 0.5 * jnp.sum(r * r)

    weights = [(W, b) for W, b, _ in base]
    a = jax.value_and_grad(loss)(weights)
    b = jax.value_and_grad(dense_loss)(weights)
    np.testing.assert_allclose(float(a[0]), float(b[0]), rtol=1e-9)
    for (gW, gb), (hW, hb) in zip(a[1], b[1], strict=True):
        np.testing.assert_allclose(np.asarray(gW), np.asarray(hW), rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(np.asarray(gb), np.asarray(hb), rtol=1e-7, atol=1e-10)


def test_stream_engine_block_orders() -> None:
    from omnibias.core.demand import block_order

    assert block_order(Residual(lambda q: q.u, ["third"]), None, "streams") == 3
    assert block_order(Residual(lambda q: q.u, ["hess_form"]), 4, "streams") == 4
    with pytest.raises(ValueError, match="order"):
        block_order(Residual(lambda q: q.u, ["fourth"]), 3, "streams")


# --------------------------------------------------------------------------- skip blocks and Fourier features
def _structured_layers(seed=11, m=4, width=None, out=C):
    from omnibias.jax.experimental import Skip, fourier_feature_layer

    rng = np.random.default_rng(seed)
    width = 2 * m if width is None else width
    freq = jnp.asarray(rng.standard_normal((m, D)))
    phase = jnp.asarray(rng.standard_normal(m))
    first = fourier_feature_layer(freq, phase)
    sq = lambda: (  # noqa: E731
        jnp.asarray(rng.standard_normal((width, width)) / np.sqrt(width)),
        jnp.asarray(0.1 * rng.standard_normal(width)),
    )
    (W1, b1), (W2, b2) = sq(), sq()
    head = (jnp.asarray(rng.standard_normal((out, width)) / np.sqrt(width)), jnp.zeros(out), None)
    return [first, (W1, b1, Skip("tanh")), (W2, b2, Skip("tanh")), head], (freq, phase)


def _forward_structured(layers, x):
    from omnibias.jax.experimental import Skip

    h = x
    for W, b, spec in layers:
        z = W @ h + b
        if isinstance(spec, Skip):
            h = h + jnp.tanh(z)
        elif spec == "sin":
            h = jnp.sin(z)
        elif spec == "tanh":
            h = jnp.tanh(z)
        else:
            h = z
    return h


def test_fourier_feature_layer_is_the_sin_cos_embedding() -> None:
    from omnibias.jax.experimental import fourier_feature_layer

    rng = np.random.default_rng(0)
    freq, phase = jnp.asarray(rng.standard_normal((5, D))), jnp.asarray(rng.standard_normal(5))
    W, b, spec = fourier_feature_layer(freq, phase)
    x = _points(4)
    got = jnp.sin(x @ W.T + b)
    z = x @ freq.T + phase
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=1)), atol=1e-14
    )
    assert spec == "sin" and W.shape == (10, D)
    W0, b0, _ = fourier_feature_layer(freq)
    assert b0.shape == (10,) and float(jnp.max(jnp.abs(b0[:5]))) == 0.0


def test_skip_blocks_and_fourier_features_through_every_stream_program() -> None:
    layers, _ = _structured_layers()
    x = _points(4)
    f = lambda z: _forward_structured(layers, z)  # noqa: E731
    res = Residual(lambda q: q.lap, ["grad", "lap", "third", "hess_form"])
    A = np.array([[1.0, 0.0, 0.4], [0.0, -1.0, 0.0], [0.4, 0.0, 0.5]])
    q = streams_quantities(x, layers, res, QuantitySpec(third=((3, 0, 0), (1, 1, 1)), hess_form=A))
    jac = jax.vmap(jax.jacfwd(f))(x)
    np.testing.assert_allclose(
        np.asarray(q.grad), np.asarray(jnp.swapaxes(jac, 1, 2)), rtol=1e-10, atol=1e-12
    )
    hess = jax.vmap(jax.hessian(f))(x)
    np.testing.assert_allclose(
        np.asarray(q.lap), np.asarray(jnp.trace(hess, axis1=-2, axis2=-1)), rtol=1e-9, atol=1e-11
    )
    np.testing.assert_allclose(
        np.asarray(q.hess_form),
        np.asarray(jnp.einsum("ij,bcij->bc", jnp.asarray(A), hess)),
        rtol=1e-9,
        atol=1e-11,
    )
    for r, alpha in enumerate(((3, 0, 0), (1, 1, 1))):
        ref = jax.vmap(_partial(f, alpha))(x)
        np.testing.assert_allclose(
            np.asarray(q.third[:, r]), np.asarray(ref), rtol=1e-8, atol=1e-10
        )
    # the Gram form of the metric (width > 2 * channels) agrees with the diag form
    gram = streams_quantities(x, layers, Residual(lambda q: q.lap, ["lap"]), form="gram").lap
    np.testing.assert_allclose(np.asarray(gram), np.asarray(q.lap), rtol=1e-9, atol=1e-11)


def test_skip_block_refusals() -> None:
    from omnibias.jax.experimental import Skip

    layers, _ = _structured_layers()
    x = _points(2)
    res = Residual(lambda q: q.lap, ["lap"])
    first = [(layers[0][0], layers[0][1], Skip("sin"))]
    with pytest.raises(ValueError, match="first layer cannot be a Skip"):
        streams_quantities(x, first, res)
    bad = [layers[0], (layers[3][0], layers[3][1], Skip("tanh"))]
    with pytest.raises(ValueError, match="square weight"):
        streams_quantities(x, bad, res)
    with pytest.raises(ValueError, match="wraps an activation"):
        Skip(None)

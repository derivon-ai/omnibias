# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity: torch multi-scale fields vs their JAX twins.

The adaptive slope and the band mixture both change *what* is fed to the closed-form
tower -- a trainable temperature, a scaled first weight matrix -- without changing
the tower itself. Parameterised by identical numpy arrays, the two backends must
therefore still agree to ``rtol=atol=1e-12`` in float64, at every order.

The last test drives the prebuilt ``equations.burgers`` residual on an Mscale field,
proving the multi-scale fields reach the existing PDE surface rather than a private
one.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.jax.architectures.multiscale import (
    AdaptiveActivation as JaxAdaptiveActivation,
)
from omnibias.jax.architectures.multiscale import (
    AdaptiveJetMLP as JaxAdaptiveMLP,
)
from omnibias.jax.architectures.multiscale import (
    MscaleMLP as JaxMscaleMLP,
)
from omnibias.jax.architectures.pinn import JetMLP as JaxJetMLP
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import equations as jeq
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.fields.multiscale import (
    AdaptiveJetMLPVectorField as JaxAdaptiveField,
)
from omnibias.pinn.jax.fields.multiscale import (
    MscaleVectorField as JaxMscaleField,
)
from omnibias.pinn.torch import equations as teq
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import (
    AdaptiveJetMLPVectorField as TorchAdaptiveField,
)
from omnibias.pinn.torch.fields import (
    MscaleVectorField as TorchMscaleField,
)

COORD_SPEC = CoordinateSpec(("x", "y", "t"))
COMP_SPEC = ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")})
JET_ORDER = 3
SLOPE_SCALE = 4.0
BANDS = (0.5, 2.0)


def _allclose(t, j, *, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return np.allclose(t_np, np.asarray(j), rtol=rtol, atol=atol)


def _chain(rng, in_dim: int, hidden: int, out_dim: int, depth: int):
    """Matching ``(weights, biases)`` numpy arrays for a depth-``depth`` MLP."""
    weights, biases = [], []
    prev = in_dim
    for d in [hidden] * depth + [out_dim]:
        weights.append(rng.normal(scale=0.7 / np.sqrt(prev), size=(d, prev)))
        biases.append(rng.normal(scale=0.1, size=(d,)))
        prev = d
    return weights, biases


def _copy_chain(linears, weights, biases) -> None:
    with torch.no_grad():
        for lin, w, b in zip(linears, weights, biases, strict=True):
            lin.weight.copy_(torch.from_numpy(w))
            lin.bias.copy_(torch.from_numpy(b))


# -- adaptive-slope field ----------------------------------------------------- #


@pytest.fixture(params=[1, 2, 3], ids=lambda d: f"depth{d}")
def adaptive_params(request):
    rng = np.random.default_rng(11)
    depth = request.param
    D, C, H = COORD_SPEC.ndim, COMP_SPEC.n_components, 6
    weights, biases = _chain(rng, D, H, C, depth)
    # A per-layer slope well away from 1, so the tempered tower is genuinely exercised.
    slopes = rng.uniform(0.6, 1.8, size=depth)
    return dict(
        weights=weights, biases=biases, slopes=slopes,
        coords=rng.normal(size=(7, D)), hidden=H, depth=depth,
    )


def _torch_adaptive(params, activation: str):
    field = TorchAdaptiveField(
        coordinate_spec=COORD_SPEC,
        components=COMP_SPEC,
        hidden=params["hidden"],
        depth=params["depth"],
        base=activation,
        slope_scale=SLOPE_SCALE,
        jet_order=JET_ORDER,
    )
    _copy_chain(field.net.linears, params["weights"], params["biases"])
    with torch.no_grad():
        for act, s in zip(field.net.activations, params["slopes"], strict=True):
            act.a.fill_(float(s) / SLOPE_SCALE)
    return field, torch.from_numpy(params["coords"])


def _jax_adaptive(params, activation: str):
    spec = jax_get_activation(activation)
    net = JaxAdaptiveMLP(
        weights=tuple(jnp.asarray(w) for w in params["weights"]),
        biases=tuple(jnp.asarray(b) for b in params["biases"]),
        activations=tuple(
            JaxAdaptiveActivation(
                a=jnp.asarray(float(s) / SLOPE_SCALE), base=spec, slope_scale=SLOPE_SCALE
            )
            for s in params["slopes"]
        ),
        in_dim=COORD_SPEC.ndim,
        out_dim=COMP_SPEC.n_components,
    )
    field = JaxAdaptiveField(
        coordinate_spec=COORD_SPEC, components=COMP_SPEC, net=net, jet_order=JET_ORDER,
    )
    return field, jnp.asarray(params["coords"])


def _adaptive_states(params, activation: str):
    tf, tc = _torch_adaptive(params, activation)
    jf, jc = _jax_adaptive(params, activation)
    assert _allclose(
        torch.stack([s.detach() for s in tf.slopes()]),
        jnp.stack(list(jf.slopes())),
    ), "the two backends must start from the same effective slopes"
    return tf(tc), jf(jc)


def test_parity_adaptive_value_and_partials(riccati, adaptive_params):
    ts, js = _adaptive_states(adaptive_params, riccati)
    for n in ("u", "v", "p"):
        assert _allclose(tops.value(ts, n), jops.value(js, n)), f"value {n!r}"
    for axis in ("x", "y", "t"):
        for order in (1, 2, 3):
            assert _allclose(
                tops.derivative(ts, "u", axis=axis, order=order),
                jops.derivative(js, "u", axis=axis, order=order),
            ), f"d^{order} u / d{axis}^{order} under {riccati}"


def test_parity_adaptive_mixed_partials(riccati, adaptive_params):
    ts, js = _adaptive_states(adaptive_params, riccati)
    for axes, orders in [
        (("x", "y"), (1, 1)),
        (("x", "y"), (2, 1)),
        (("x", "y", "t"), (1, 1, 1)),
    ]:
        assert _allclose(
            tops.mixed_partial(ts, "u", axes, orders),
            jops.mixed_partial(js, "u", axes, orders),
        ), f"mixed_partial({axes}, {orders}) under {riccati}"


def test_parity_adaptive_operator_surface(riccati, adaptive_params):
    ts, js = _adaptive_states(adaptive_params, riccati)
    assert _allclose(tops.gradient(ts, "u"), jops.gradient(js, "u"))
    assert _allclose(tops.hessian(ts, "u"), jops.hessian(js, "u"))
    assert _allclose(tops.laplacian(ts, "u"), jops.laplacian(js, "u"))
    assert _allclose(tops.jacobian(ts, ("u", "v")), jops.jacobian(js, ("u", "v")))
    assert _allclose(tops.divergence(ts, ("u", "v")), jops.divergence(js, ("u", "v")))


# -- Mscale band mixture ------------------------------------------------------ #


@pytest.fixture
def mscale_params():
    rng = np.random.default_rng(23)
    D, C, H, depth = COORD_SPEC.ndim, COMP_SPEC.n_components, 4, 2
    bands = [_chain(rng, D, H, C, depth) for _ in BANDS]
    return dict(
        bands=bands, coords=rng.normal(size=(7, D)),
        hidden=H * len(BANDS), depth=depth,
    )


def _torch_mscale(params, activation: str):
    field = TorchMscaleField(
        coordinate_spec=COORD_SPEC,
        components=COMP_SPEC,
        hidden=params["hidden"],
        depth=params["depth"],
        base=activation,
        scales=BANDS,
        jet_order=JET_ORDER,
    )
    for sub, (weights, biases) in zip(field.net.subnets, params["bands"], strict=True):
        _copy_chain(sub.linears, weights, biases)
    return field, torch.from_numpy(params["coords"])


def _jax_mscale(params, activation: str):
    spec = jax_get_activation(activation)
    subnets = tuple(
        JaxJetMLP(
            weights=tuple(jnp.asarray(w) for w in weights),
            biases=tuple(jnp.asarray(b) for b in biases),
            spec=spec,
            in_dim=COORD_SPEC.ndim,
            out_dim=COMP_SPEC.n_components,
        )
        for weights, biases in params["bands"]
    )
    net = JaxMscaleMLP(
        subnets=subnets,
        scales=BANDS,
        in_dim=COORD_SPEC.ndim,
        out_dim=COMP_SPEC.n_components,
    )
    field = JaxMscaleField(
        coordinate_spec=COORD_SPEC, components=COMP_SPEC, net=net, jet_order=JET_ORDER,
    )
    return field, jnp.asarray(params["coords"])


def _mscale_states(params, activation: str):
    tf, tc = _torch_mscale(params, activation)
    jf, jc = _jax_mscale(params, activation)
    assert tf.scales == jf.scales
    return tf(tc), jf(jc)


def test_parity_mscale_value_and_partials(riccati, mscale_params):
    ts, js = _mscale_states(mscale_params, riccati)
    for n in ("u", "v", "p"):
        assert _allclose(tops.value(ts, n), jops.value(js, n)), f"value {n!r}"
    for axis in ("x", "y", "t"):
        for order in (1, 2, 3):
            assert _allclose(
                tops.derivative(ts, "u", axis=axis, order=order),
                jops.derivative(js, "u", axis=axis, order=order),
            ), f"d^{order} u / d{axis}^{order} under {riccati}"


def test_parity_mscale_operator_surface(riccati, mscale_params):
    ts, js = _mscale_states(mscale_params, riccati)
    assert _allclose(tops.gradient(ts, "u"), jops.gradient(js, "u"))
    assert _allclose(tops.hessian(ts, "u"), jops.hessian(js, "u"))
    assert _allclose(tops.laplacian(ts, "u"), jops.laplacian(js, "u"))
    assert _allclose(tops.biharmonic(ts, "u"), jops.biharmonic(js, "u"))
    assert _allclose(
        tops.mixed_partial(ts, "u", ("x", "y"), (2, 1)),
        jops.mixed_partial(js, "u", ("x", "y"), (2, 1)),
    )
    assert _allclose(tops.curl(ts, ("u", "v")), jops.curl(js, ("u", "v")))


# -- the multi-scale fields reach the prebuilt PDE residuals ------------------ #


def test_burgers_residual_on_an_mscale_field_matches_across_backends(mscale_params):
    ts, js = _mscale_states(mscale_params, "tanh")
    t_out = teq.burgers(ts, nu=0.01, form="scalar", component="u")
    j_out = jeq.burgers(js, nu=0.01, form="scalar", component="u")
    scale = float(t_out.residual.detach().abs().max())
    assert _allclose(t_out.residual, j_out.residual, rtol=1e-12, atol=1e-12 * scale)

    t_vec = teq.burgers(ts, nu=0.01, form="vector", velocity=("u", "v"))
    j_vec = jeq.burgers(js, nu=0.01, form="vector", velocity=("u", "v"))
    scale = float(t_vec.residual.detach().abs().max())
    assert _allclose(t_vec.residual, j_vec.residual, rtol=1e-12, atol=1e-12 * scale)


def test_burgers_residual_is_trainable_on_a_multiscale_field(mscale_params, adaptive_params):
    """The residual must backprop into the band weights *and* the trainable slopes."""
    tf, tc = _torch_mscale(mscale_params, "tanh")
    teq.burgers(tf(tc), nu=0.01, form="scalar", component="u").residual.pow(2).mean().backward()
    grads = [p.grad for p in tf.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0 for g in grads)

    af, ac = _torch_adaptive(adaptive_params, "tanh")
    teq.burgers(af(ac), nu=0.01, form="scalar", component="u").residual.pow(2).mean().backward()
    slope_grads = [act.a.grad for act in af.net.activations]
    assert all(g is not None and float(g.abs()) > 0 for g in slope_grads)

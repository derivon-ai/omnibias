# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity: the torch attention field vs its JAX twin.

The block is the same sequence of primitives on both backends -- one encoder jet,
one max-shifted softmax jet, one affine readout -- so a field parameterised by
identical numpy arrays must produce identical values, attention weights and
derivatives in float64. The softmax is the interesting part: a different
max-shift or a different summation order in the denominator would show up here
immediately.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.jax.architectures.attention import AttentionJetMLP as JaxAttentionNet
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.fields import AttentionVectorField as JaxField
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import AttentionVectorField as TorchField
from omnibias.torch.architectures.attention import AttentionJetMLP as TorchAttentionNet

COORD_SPEC = CoordinateSpec(("x", "y", "t"))
COMP_SPEC = ComponentSpec(("u", "v"), groups={"velocity": ("u", "v")})
JET_ORDER = 3
TOL = 1e-12


@pytest.fixture(params=[1, 2], ids=lambda d: f"depth{d}")
def depth(request) -> int:
    return request.param


@pytest.fixture(params=[True, False], ids=["residual", "plain"])
def residual(request) -> bool:
    return request.param


@pytest.fixture
def shared(depth: int, residual: bool):
    rng = np.random.default_rng(23)
    D, C, H, N = COORD_SPEC.ndim, COMP_SPEC.n_components, 6, 5
    d_val = H if residual else 4
    weights, biases = [], []
    prev = D
    for _ in range(depth):
        weights.append(rng.normal(scale=0.7 / np.sqrt(prev), size=(H, prev)))
        biases.append(rng.normal(scale=0.2, size=(H,)))
        prev = H
    return {
        "weights": weights,
        "biases": biases,
        "keys": rng.normal(scale=0.9, size=(N, H)),
        "values": rng.normal(scale=1.1, size=(N, d_val)),
        "readout_w": rng.normal(scale=0.6, size=(C, d_val)),
        "readout_b": rng.normal(scale=0.2, size=(C,)),
        "beta": 1.6,
        "residual": residual,
        "coords": rng.normal(size=(7, D)),
    }


def _torch_field(shared) -> TorchField:
    field = TorchField(
        coordinate_spec=COORD_SPEC,
        components=COMP_SPEC,
        hidden=int(shared["weights"][0].shape[0]),
        depth=len(shared["weights"]),
        memory=int(shared["keys"].shape[0]),
        value_dim=int(shared["values"].shape[-1]),
        beta=shared["beta"],
        residual=shared["residual"],
        jet_order=JET_ORDER,
    )
    net = field.net
    assert isinstance(net, TorchAttentionNet)
    with torch.no_grad():
        for lin, w, b in zip(
            net.encoder, shared["weights"], shared["biases"], strict=True
        ):
            lin.weight.copy_(torch.from_numpy(w))
            lin.bias.copy_(torch.from_numpy(b))
        net.keys.copy_(torch.from_numpy(shared["keys"]))
        net.values.copy_(torch.from_numpy(shared["values"]))
        net.readout.weight.copy_(torch.from_numpy(shared["readout_w"]))
        net.readout.bias.copy_(torch.from_numpy(shared["readout_b"]))
    return field


def _jax_field(shared) -> JaxField:
    net = JaxAttentionNet(
        weights=tuple(jnp.asarray(w) for w in shared["weights"]),
        biases=tuple(jnp.asarray(b) for b in shared["biases"]),
        keys=jnp.asarray(shared["keys"]),
        values=jnp.asarray(shared["values"]),
        readout_weight=jnp.asarray(shared["readout_w"]),
        readout_bias=jnp.asarray(shared["readout_b"]),
        beta=jnp.asarray(shared["beta"]),
        spec=jax_get_activation("tanh"),
        in_dim=COORD_SPEC.ndim,
        out_dim=COMP_SPEC.n_components,
        residual=shared["residual"],
    )
    return JaxField(
        coordinate_spec=COORD_SPEC,
        components=COMP_SPEC,
        net=net,
        jet_order=JET_ORDER,
    )


def _allclose(t, j, *, tol: float = TOL) -> bool:
    t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return np.allclose(t_np, np.asarray(j), rtol=tol, atol=tol)


@pytest.fixture
def states(shared):
    tf, jf = _torch_field(shared), _jax_field(shared)
    coords = shared["coords"]
    return (
        tf,
        jf,
        tf(torch.from_numpy(coords)),
        jf(jnp.asarray(coords)),
    )


def test_parity_value(states) -> None:
    _tf, _jf, ts, js = states
    for name in COMP_SPEC.names:
        assert _allclose(tops.value(ts, name), jops.value(js, name)), name


def test_parity_attention_weights(states, shared) -> None:
    tf, jf, _ts, _js = states
    coords = shared["coords"]
    assert _allclose(
        tf.attention_weights(torch.from_numpy(coords)),
        jf.attention_weights(jnp.asarray(coords)),
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_parity_pure_partials(states, order: int) -> None:
    _tf, _jf, ts, js = states
    for axis in range(COORD_SPEC.ndim):
        assert _allclose(
            tops.derivative(ts, "u", axis=axis, order=order),
            jops.derivative(js, "u", axis=axis, order=order),
        ), (axis, order)


def test_parity_mixed_partials(states) -> None:
    _tf, _jf, ts, js = states
    for axes, orders in (((0, 1), (1, 1)), ((0, 2), (2, 1)), ((1, 2), (1, 2))):
        assert _allclose(
            tops.mixed_partial(ts, "v", axes, orders),
            jops.mixed_partial(js, "v", axes, orders),
        ), (axes, orders)


def test_parity_operator_surface(states) -> None:
    _tf, _jf, ts, js = states
    assert _allclose(tops.gradient(ts, "u"), jops.gradient(js, "u"))
    assert _allclose(tops.hessian(ts, "u"), jops.hessian(js, "u"))
    assert _allclose(tops.laplacian(ts, "u"), jops.laplacian(js, "u"))
    assert _allclose(
        tops.divergence(ts, ("u", "v")), jops.divergence(js, ("u", "v"))
    )


def test_parity_holds_at_a_saturating_temperature(shared) -> None:
    """A sharp mixture is where a mismatched max-shift would show up first.

    The second derivative is compared at a looser tolerance because it is
    genuinely worse conditioned here: ``beta = 25`` multiplies every score
    difference, so the Laplacian entries span ten orders of magnitude and the two
    backends' summation orders differ in the last few digits of the largest ones.
    The value and gradient still agree to ``1e-12``.
    """
    hot = dict(shared, beta=25.0)
    tf, jf = _torch_field(hot), _jax_field(hot)
    coords = hot["coords"]
    ts, js = tf(torch.from_numpy(coords)), jf(jnp.asarray(coords))
    assert _allclose(tops.value(ts, "u"), jops.value(js, "u"))
    assert _allclose(tops.gradient(ts, "u"), jops.gradient(js, "u"))
    assert _allclose(tops.laplacian(ts, "u"), jops.laplacian(js, "u"), tol=1e-10)


def test_parity_of_the_pde_residual(states) -> None:
    """The parity that matters downstream: the same residual on both backends."""
    from omnibias.pinn.jax import equations as jeq
    from omnibias.pinn.torch import equations as teq

    _tf, _jf, ts, js = states
    t_out = teq.burgers(ts, nu=0.03, form="vector", velocity=("u", "v"))
    j_out = jeq.burgers(js, nu=0.03, form="vector", velocity=("u", "v"))
    assert _allclose(t_out.residual, j_out.residual)


def test_the_jax_field_survives_a_pytree_round_trip(states, shared) -> None:
    _tf, jf, _ts, js = states
    leaves, treedef = jax.tree_util.tree_flatten(jf)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert dataclasses.asdict(rebuilt.net).keys() == dataclasses.asdict(jf.net).keys()
    coords = jnp.asarray(shared["coords"])
    assert np.array_equal(rebuilt.net.value(coords), jf.net.value(coords))

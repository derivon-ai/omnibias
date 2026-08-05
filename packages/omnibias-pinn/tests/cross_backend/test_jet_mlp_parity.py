# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity: torch ``jet_mlp`` fields vs their JAX twins.

Both backends drive the same multivariate Faa di Bruno kernel over the same
:func:`omnibias.core.multi_index.multi_indices` row order, so a deep field
parameterised by identical numpy arrays must produce identical derivatives to
``rtol=atol=1e-12`` in float64 -- at every depth, order, and activation.

The last test closes the loop the whole phase exists for: a Fourier-feature field
driving the *prebuilt* ``equations.burgers`` residual on both backends, proving
the new field type reaches the existing PDE surface rather than a private one.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.jax.architectures.pinn import (
    FourierFeatureMLP as JaxFourierMLP,
)
from omnibias.jax.architectures.pinn import (
    JetMLP as JaxJetMLP,
)
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import equations as jeq
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.fields.jet_mlp import (
    FourierFeatureVectorField as JaxFourierField,
)
from omnibias.pinn.jax.fields.jet_mlp import (
    JetMLPVectorField as JaxJetField,
)
from omnibias.pinn.torch import equations as teq
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import (
    FourierFeatureVectorField as TorchFourierField,
)
from omnibias.pinn.torch.fields import (
    JetMLPVectorField as TorchJetField,
)

COORD_SPEC = CoordinateSpec(("x", "y", "t"))
COMP_SPEC = ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")})
JET_ORDER = 4


@pytest.fixture(params=[1, 2, 3], ids=lambda d: f"depth{d}")
def depth(request) -> int:
    return request.param


@pytest.fixture
def deep_params(depth):
    """Matching ``(weights, biases)`` numpy arrays for a depth-``depth`` MLP."""
    rng = np.random.default_rng(11)
    D, C, H = 3, 3, 6
    dims = [H] * depth + [C]
    weights, biases = [], []
    prev = D
    for d in dims:
        weights.append(rng.normal(scale=0.7 / np.sqrt(prev), size=(d, prev)))
        biases.append(rng.normal(scale=0.1, size=(d,)))
        prev = d
    coords = rng.normal(size=(7, D))
    return dict(weights=weights, biases=biases, coords=coords)


def _torch_field(params, activation: str):
    field = TorchJetField(
        coordinate_spec=COORD_SPEC,
        components=COMP_SPEC,
        hidden=int(params["weights"][0].shape[0]),
        depth=len(params["weights"]) - 1,
        base=activation,
        jet_order=JET_ORDER,
    )
    with torch.no_grad():
        for lin, w, b in zip(
            field.net.linears, params["weights"], params["biases"], strict=True
        ):
            lin.weight.copy_(torch.from_numpy(w))
            lin.bias.copy_(torch.from_numpy(b))
    return field, torch.from_numpy(params["coords"])


def _jax_field(params, activation: str):
    net = JaxJetMLP(
        weights=tuple(jnp.asarray(w) for w in params["weights"]),
        biases=tuple(jnp.asarray(b) for b in params["biases"]),
        spec=jax_get_activation(activation),
        in_dim=COORD_SPEC.ndim,
        out_dim=COMP_SPEC.n_components,
    )
    field = JaxJetField(
        coordinate_spec=COORD_SPEC,
        components=COMP_SPEC,
        net=net,
        jet_order=JET_ORDER,
    )
    return field, jnp.asarray(params["coords"])


def _allclose(t, j, *, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return np.allclose(t_np, np.asarray(j), rtol=rtol, atol=atol)


def _states(deep_params, activation: str):
    tf, tc = _torch_field(deep_params, activation)
    jf, jc = _jax_field(deep_params, activation)
    return tf(tc), jf(jc)


def test_parity_value(riccati, deep_params):
    ts, js = _states(deep_params, riccati)
    for n in ("u", "v", "p"):
        assert _allclose(tops.value(ts, n), jops.value(js, n)), f"value {n!r}"


def test_parity_pure_partials(riccati, deep_params):
    ts, js = _states(deep_params, riccati)
    for axis in ("x", "y", "t"):
        for order in (1, 2, 3, 4):
            assert _allclose(
                tops.derivative(ts, "u", axis=axis, order=order),
                jops.derivative(js, "u", axis=axis, order=order),
            ), f"d^{order} u / d{axis}^{order} under {riccati}"


def test_parity_mixed_partials(riccati, deep_params):
    ts, js = _states(deep_params, riccati)
    for axes, orders in [
        (("x", "y"), (1, 1)),
        (("x", "y"), (2, 1)),
        (("x", "t"), (2, 1)),
        (("x", "y", "t"), (1, 1, 1)),
        (("x", "y", "t"), (2, 1, 1)),
    ]:
        assert _allclose(
            tops.mixed_partial(ts, "u", axes, orders),
            jops.mixed_partial(js, "u", axes, orders),
        ), f"mixed_partial({axes}, {orders}) under {riccati}"


def test_parity_gradient_hessian_jacobian(riccati, deep_params):
    ts, js = _states(deep_params, riccati)
    for n in ("u", "v"):
        assert _allclose(tops.gradient(ts, n), jops.gradient(js, n))
        assert _allclose(
            tops.gradient(ts, n, axes=("x", "y", "t")),
            jops.gradient(js, n, axes=("x", "y", "t")),
        )
    assert _allclose(tops.hessian(ts, "u"), jops.hessian(js, "u"))
    assert _allclose(
        tops.hessian(ts, "u", axes=("x", "y")),
        jops.hessian(js, "u", axes=("x", "y")),
    )
    assert _allclose(tops.spatial_hessian(ts, "u"), jops.spatial_hessian(js, "u"))
    assert _allclose(tops.jacobian(ts, ("u", "v")), jops.jacobian(js, ("u", "v")))


def test_parity_laplacian_biharmonic_polylaplacian(riccati, deep_params):
    ts, js = _states(deep_params, riccati)
    for n in ("u", "v"):
        assert _allclose(tops.laplacian(ts, n), jops.laplacian(js, n))
        assert _allclose(tops.biharmonic(ts, n), jops.biharmonic(js, n))
    for k in (1, 2):
        assert _allclose(
            tops.polylaplacian(ts, "u", k=k), jops.polylaplacian(js, "u", k=k)
        ), f"polylap k={k} under {riccati}"


def test_parity_vector_ops(riccati, deep_params):
    ts, js = _states(deep_params, riccati)
    assert _allclose(tops.divergence(ts, ("u", "v")), jops.divergence(js, ("u", "v")))
    assert _allclose(tops.curl(ts, ("u", "v")), jops.curl(js, ("u", "v")))
    assert _allclose(
        tops.advection(ts, velocity=("u", "v")),
        jops.advection(js, velocity=("u", "v")),
    )
    assert _allclose(
        tops.strain_rate(ts, ("u", "v")), jops.strain_rate(js, ("u", "v"))
    )


# -- Fourier-feature field on the prebuilt PDE residuals ---------------------- #


@pytest.fixture
def fourier_params():
    rng = np.random.default_rng(23)
    D, C, F, H = 3, 3, 5, 6
    scales = (0.5, 2.0)
    bands = [
        rng.normal(size=(F, D)) * (2.0 * np.pi * s) for s in scales
    ]
    b_mat = np.concatenate(bands, axis=0)
    f_total = b_mat.shape[0]
    w_ff = np.concatenate([b_mat, b_mat], axis=0)
    b_ff = np.concatenate(
        [np.full((f_total,), 0.5 * np.pi), np.zeros((f_total,))]
    )
    weights, biases = [], []
    prev = 2 * f_total
    for d in (H, C):
        weights.append(rng.normal(scale=0.7 / np.sqrt(prev), size=(d, prev)))
        biases.append(rng.normal(scale=0.1, size=(d,)))
        prev = d
    coords = rng.normal(size=(7, D))
    return dict(
        w_ff=w_ff, b_ff=b_ff, weights=weights, biases=biases,
        coords=coords, scales=scales, num_features=F,
    )


def _torch_fourier(params):
    field = TorchFourierField(
        coordinate_spec=COORD_SPEC,
        components=COMP_SPEC,
        num_features=params["num_features"],
        hidden=int(params["weights"][0].shape[0]),
        depth=1,
        frequency_scale=params["scales"],
    )
    with torch.no_grad():
        field.net.W_ff.copy_(torch.from_numpy(params["w_ff"]))
        field.net.b_ff.copy_(torch.from_numpy(params["b_ff"]))
        for lin, w, b in zip(
            field.net.linears, params["weights"], params["biases"], strict=True
        ):
            lin.weight.copy_(torch.from_numpy(w))
            lin.bias.copy_(torch.from_numpy(b))
    return field, torch.from_numpy(params["coords"])


def _jax_fourier(params):
    net = JaxFourierMLP(
        w_ff=jnp.asarray(params["w_ff"]),
        b_ff=jnp.asarray(params["b_ff"]),
        weights=tuple(jnp.asarray(w) for w in params["weights"]),
        biases=tuple(jnp.asarray(b) for b in params["biases"]),
        base_spec=jax_get_activation("tanh"),
        in_dim=COORD_SPEC.ndim,
        out_dim=COMP_SPEC.n_components,
        num_features=params["num_features"],
        scales=params["scales"],
    )
    field = JaxFourierField(
        coordinate_spec=COORD_SPEC, components=COMP_SPEC, net=net, jet_order=2,
    )
    return field, jnp.asarray(params["coords"])


def test_parity_fourier_feature_field(fourier_params):
    tf, tc = _torch_fourier(fourier_params)
    jf, jc = _jax_fourier(fourier_params)
    assert tf.scales == jf.scales
    assert tf.feature_dim == jf.feature_dim
    ts, js = tf(tc), jf(jc)
    assert _allclose(tops.value(ts, "u"), jops.value(js, "u"))
    # The (2 pi * 2.0) band amplifies second derivatives, so scale the tolerance
    # to the magnitude of the quantity rather than to 1.
    lap_t = tops.laplacian(ts, "u")
    scale = float(lap_t.detach().abs().max())
    assert _allclose(lap_t, jops.laplacian(js, "u"), rtol=1e-12, atol=1e-12 * scale)
    assert _allclose(tops.gradient(ts, "u"), jops.gradient(js, "u"), atol=1e-12)


def test_burgers_residual_on_fourier_field_matches_across_backends(fourier_params):
    """The new field type reaches the prebuilt PDE residual builders, identically."""
    tf, tc = _torch_fourier(fourier_params)
    jf, jc = _jax_fourier(fourier_params)
    ts, js = tf(tc), jf(jc)

    t_out = teq.burgers(ts, nu=0.01, form="scalar", component="u")
    j_out = jeq.burgers(js, nu=0.01, form="scalar", component="u")
    scale = float(t_out.residual.detach().abs().max())
    assert _allclose(t_out.residual, j_out.residual, rtol=1e-12, atol=1e-12 * scale)
    assert t_out.residual.shape == tc.shape[:1]

    t_vec = teq.burgers(ts, nu=0.01, form="vector", velocity=("u", "v"))
    j_vec = jeq.burgers(js, nu=0.01, form="vector", velocity=("u", "v"))
    scale = float(t_vec.residual.detach().abs().max())
    assert _allclose(t_vec.residual, j_vec.residual, rtol=1e-12, atol=1e-12 * scale)


def test_burgers_residual_is_trainable_on_a_fourier_field(fourier_params):
    """A residual on the new field must backprop into the network parameters."""
    tf, tc = _torch_fourier(fourier_params)
    out = teq.burgers(tf(tc), nu=0.01, form="scalar", component="u")
    out.residual.pow(2).mean().backward()
    grads = [p.grad for p in tf.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0 for g in grads)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Joint parameter/input jets against independent AD and six live block roles."""

from __future__ import annotations

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.multi_index import multi_indices
from omnibias.core.realization import LayerSpec, ObservationSpec, ParameterLayout, RealizationSpec
from omnibias.jax import realization as jr
from omnibias.jax.activations import get_activation as jactivation
from omnibias.pinn.operator.jax.parameter_jets import mixed_jet as jmixed
from omnibias.pinn.operator.torch.parameter_jets import ParameterJetSpec
from omnibias.pinn.operator.torch.parameter_jets import mixed_jet as tmixed
from omnibias.torch import realization as tr
from omnibias.torch.activations import get_activation
from omnibias.torch.blocks.operator import OperatorBlock
from omnibias.torch.multipack import multipack_response

jax.config.update("jax_enable_x64", True)


def test_joint_weight_input_mixed_partials_and_live_outer_gradient() -> None:
    spec = RealizationSpec.dense((2, 2, 1))
    theta = torch.linspace(-0.4, 0.6, spec.layout.size, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    basis = torch.linspace(-0.3, 0.4, spec.layout.size * 2, dtype=torch.float64).reshape(-1, 2)
    input_basis = torch.tensor([[0.2, 0.1], [-0.1, 0.4]], dtype=torch.float64)
    jet = tr.parameter_jet(x, theta, spec, basis, 4, input_basis=input_basis)

    def reference(t: torch.Tensor) -> torch.Tensor:
        p, z = theta + basis @ t, x + input_basis @ t
        hidden = torch.tanh(p[:4].reshape(2, 2) @ z + p[4:6])
        return (p[6:8] @ hidden + p[8]).reshape(1)

    origin = torch.zeros(2, dtype=torch.float64)
    for row, alpha in zip(jet, multi_indices(2, 4), strict=True):
        fn = reference
        for axis, n in enumerate(alpha):
            for _ in range(n):
                previous = fn
                def fn(t, f=previous, a=axis):
                    return torch.func.jacfwd(f)(t)[..., a]
        expected = fn(origin) / math.prod(math.factorial(n) for n in alpha)
        torch.testing.assert_close(row, expected, atol=2e-13, rtol=2e-12)
    gradients = torch.autograd.grad(jet.square().sum(), (theta, x))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in gradients)
    jtheta, jx, jb, jib = map(lambda t: jnp.asarray(t.detach().numpy()), (theta, x, basis, input_basis))
    jfn = jax.jit(lambda p: jr.parameter_jet(jx, p, spec, jb, 4, input_basis=jib))
    np.testing.assert_allclose(np.asarray(jfn(jtheta)), jet.detach().numpy(), atol=3e-13, rtol=3e-12)
    np.testing.assert_allclose(np.asarray(jax.grad(lambda p: jnp.sum(jfn(p) ** 2))(jtheta)),
                               gradients[0].numpy(), atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize("op", ["identity", "grad", "laplacian", "derivative", "band", "integral"])
def test_all_six_realization_roles(op: str) -> None:
    window = ("low", "high") if op in {"band", "integral"} else None
    shapes = [("W0", (1, 1)), ("b0", (1,))]
    if window:
        shapes.extend((name, (1,)) for name in window)
    layer = LayerSpec(1, 1, "sigmoid", op=op, derivative_order=3, bias_name="b0", window_names=window)
    spec = RealizationSpec((layer,), ParameterLayout.from_shapes(shapes), 1, 1)
    theta = torch.tensor([0.6, -0.2] + ([-0.3, 0.4] if window else []), dtype=torch.float64)
    direction = torch.linspace(0.2, 0.5, theta.numel(), dtype=torch.float64)
    x = torch.tensor([0.25], dtype=torch.float64)
    jet = tr.parameter_jet(x, theta, spec, direction[:, None], 3)

    def reference(t: torch.Tensor) -> torch.Tensor:
        p = theta + direction * t
        z = p[0] * x + p[1]
        if op == "band":
            return torch.sigmoid(z + p[3]) - torch.sigmoid(z + p[2])
        if op == "integral":
            return torch.nn.functional.softplus(z + p[3]) - torch.nn.functional.softplus(z + p[2])
        fn = torch.sigmoid
        n = {"identity": 0, "grad": 1, "laplacian": 2, "derivative": 3}[op]
        for _ in range(n):
            old = fn
            def fn(z, f=old):
                return torch.func.jacfwd(f)(z).diagonal()
        return fn(z)

    fn = reference
    for n in range(4):
        torch.testing.assert_close(jet[n], fn(torch.tensor(0.0, dtype=torch.float64)) / math.factorial(n),
                                   atol=3e-12, rtol=3e-11)
        fn = torch.func.jacfwd(fn)
    jp = jr.parameter_jet(jnp.asarray(x.numpy()), jnp.asarray(theta.numpy()), spec,
                         jnp.asarray(direction.numpy())[:, None], 3)
    np.testing.assert_allclose(np.asarray(jp), jet.numpy(), atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("op", ["identity", "grad", "laplacian", "derivative", "band", "integral"])
def test_operator_block_adapter_preserves_live_biases_and_signs(op: str) -> None:
    block = OperatorBlock(op, base="tanh", channels=2, learnable_signs=True,
                          derivative_order=3 if op == "derivative" else None).double()
    z = torch.tensor([[0.2, -0.3], [0.1, 0.4]], dtype=torch.float64)
    dz = torch.full_like(z, 0.17)
    zero = torch.zeros_like(z)
    jet = tr.operator_block_jet(block, torch.stack((z, dz, zero)), dim=1, order=2)
    def fn(t):
        return block(z + t * dz)
    torch.testing.assert_close(jet[0], fn(torch.tensor(0.0, dtype=torch.float64)))
    torch.testing.assert_close(jet[1], torch.func.jacfwd(fn)(torch.tensor(0.0, dtype=torch.float64)))
    torch.testing.assert_close(jet[2] * 2, torch.func.jacfwd(torch.func.jacfwd(fn))(torch.tensor(0.0, dtype=torch.float64)))
    grads = torch.autograd.grad(jet.sum(), (block.ombu.biases, block.ombu.signs), allow_unused=True)
    assert grads[0] is not None and torch.isfinite(grads[0]).all()
    assert (grads[1] is None) == (op in {"grad", "laplacian", "derivative"})


def test_integral_adapter_normalization_and_existing_midpoint_policy() -> None:
    for width in (0.8, 1e-7):
        block = OperatorBlock("integral", base="sigmoid", init_delta=width, normalize_integral=True).double()
        z = torch.tensor([[0.3]], dtype=torch.float64)
        jet = tr.operator_block_jet(block, torch.stack((z, torch.ones_like(z))), dim=1, order=1)
        torch.testing.assert_close(jet[0], block(z))
        torch.testing.assert_close(jet[1], torch.func.jacfwd(lambda t, b=block, zz=z: b(zz + t))(torch.tensor(0.0, dtype=torch.float64)))


def test_observations_vjp_and_coefficient_budget() -> None:
    spec = RealizationSpec.dense((1, 2, 1))
    theta = torch.linspace(-0.4, 0.6, spec.layout.size, dtype=torch.float64)
    points = torch.tensor([[0.1], [0.4]], dtype=torch.float64)
    realization = tr.Realization(spec)
    direction = torch.ones_like(theta)
    cotangent = torch.tensor([[0.5], [-0.2]], dtype=torch.float64)
    torch.testing.assert_close((realization.jvp(points, theta, direction) * cotangent).sum(),
                               (realization.vjp(points, theta, cotangent) * direction).sum())
    obs = ObservationSpec(kind="derivatives", derivative_indices=((1,), (2,)))
    derivative = tr.observe(theta, spec, obs, points=points)
    def f(x):
        return tr.evaluate(x, theta, spec)
    expected = torch.cat([torch.func.vmap(torch.func.jacfwd(f))(points).reshape(-1),
                          torch.func.vmap(torch.func.jacfwd(torch.func.jacfwd(f)))(points).reshape(-1)])
    torch.testing.assert_close(derivative, expected)
    integral = tr.observe(theta, spec, ObservationSpec(kind="integrals"), points=points,
                          weights=torch.tensor([0.25, 0.75], dtype=torch.float64))
    torch.testing.assert_close(integral, (f(points).flatten() * torch.tensor([0.25, 0.75])).sum().reshape(1))
    with pytest.raises(ValueError, match="budget"):
        tr.parameter_jet(points, theta, spec, torch.eye(theta.numel(), dtype=torch.float64), 5, max_coefficients=100)


def test_shared_tower_reuses_one_native_base_per_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.sigmoid
    calls = []
    def counted(z: torch.Tensor) -> torch.Tensor:
        calls.append(z)
        return original(z)
    monkeypatch.setattr(torch, "sigmoid", counted)
    spec = get_activation("sigmoid")
    z = torch.tensor([0.1, 0.2], dtype=torch.float64)
    out = multipack_response(z, torch.tensor([0.0, 0.3]), torch.ones(4), (0, 1, 4, 2), spec,
                              mean_index=(0, 0, 0, 1))
    assert len(calls) == 2
    assert torch.isfinite(out).all()
    fallback = replace(spec, tower=None)
    calls.clear()
    reference = multipack_response(z, torch.tensor([0.0, 0.3]), torch.ones(4), (0, 1, 4, 2), fallback,
                                    mean_index=(0, 0, 0, 1))
    assert len(calls) == 4
    assert torch.equal(out, reference)


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus"])
def test_tower_rows_match_existing_fastpaths(name: str) -> None:
    x = torch.tensor([-0.2, 0.0, 0.5], dtype=torch.float64)
    spec, jspec = get_activation(name), jactivation(name)
    assert spec.tower is not None and jspec.tower is not None
    rows = spec.tower(x, 6)
    assert torch.equal(rows, torch.stack([spec.fastpath(x, n) for n in range(7)]))
    jrows = jspec.tower(jnp.asarray(x.numpy()), 6)
    np.testing.assert_allclose(np.asarray(jrows), rows.numpy(), atol=1e-13, rtol=1e-11)
    with pytest.raises(ValueError):
        spec.tower(x, -1)


def test_parameter_api_uses_real_ad_and_retains_graph() -> None:
    cfg = ParameterJetSpec(space_order=1, param_order=2, method="autodiff", mu_in_jet_trunk=False,
                           space_multi_index=(1, 0))
    point = torch.tensor([0.3, 0.2], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    value = tmixed(lambda c, p: p ** 3 * c[0] ** 2 + c[1], point, mu, spec=cfg)
    torch.testing.assert_close(value, 12 * mu * point[0])
    torch.testing.assert_close(torch.autograd.grad(value, mu)[0], 12 * point[0])
    jvalue = jax.jit(lambda p: jmixed(lambda c, m: m ** 3 * c[0] ** 2 + c[1],
                                     jnp.array([0.3, 0.2]), p, spec=cfg))(jnp.array(0.7))
    np.testing.assert_allclose(np.asarray(jvalue), value.detach().numpy(), atol=1e-14)
    with pytest.raises(TypeError, match="mixed_parameter_jet"):
        tmixed(lambda c, p: p * c[0], point, mu)


def test_manufactured_mixed_heat_partial_is_live_and_matches_ad() -> None:
    cfg = ParameterJetSpec(space_order=3, param_order=2, space_multi_index=(1, 2))
    point = torch.tensor([0.3, 0.2], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    actual = tmixed(None, point, mu, spec=cfg, wave=1.3, live=True)
    expected = tmixed(None, point, mu, spec=replace(cfg, method="autodiff"), wave=1.3)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.grad(actual, mu)[0].isfinite()


def test_activation_window_is_distinct_from_domain_and_measure_integrals() -> None:
    layer = LayerSpec(1, 1, "sigmoid", op="integral", window_names=("low", "high"))
    spec = RealizationSpec((layer,), ParameterLayout.from_shapes([("W0", (1, 1)),
                                                                 ("low", (1,)), ("high", (1,))]), 1, 1)
    theta = torch.tensor([1.0, -0.3, 0.4], dtype=torch.float64)
    points = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    activation = ObservationSpec(kind="integrals", integral_kind="activation_window")
    window = tr.observe(theta, spec, activation, points=points)
    expected = torch.nn.functional.softplus(points + 0.4) - torch.nn.functional.softplus(points - 0.3)
    torch.testing.assert_close(window, expected.flatten())
    np.testing.assert_allclose(np.asarray(jr.observe(jnp.asarray(theta.numpy()), spec, activation,
                                                     points=jnp.asarray(points.numpy()))), window.numpy(), atol=2e-16)
    weights = torch.tensor([0.2, 0.8], dtype=torch.float64)
    for kind in (None, "domain_quadrature", "measure"):
        obs = ObservationSpec(kind="integrals", integral_kind=kind, provenance="declared finite weighted rule")
        total = tr.observe(theta, spec, obs, points=points, weights=weights)
        torch.testing.assert_close(total, (window * weights).sum().reshape(1))
    with pytest.raises(ValueError, match="quadrature weights"):
        tr.observe(theta, spec, activation, points=points, weights=weights)
    with pytest.raises(ValueError, match="integral-role"):
        tr.observe(torch.ones(2, dtype=torch.float64), RealizationSpec.dense((1, 1)), activation, points=points)


def test_live_closed_form_field_provider_is_honored() -> None:
    class Provider:
        def mixed_parameter_jet(self, coords, parameters, *, spec):
            return parameters * coords[0] + spec.param_order
    point = torch.tensor([0.3, 0.2], dtype=torch.float64)
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    result = tmixed(Provider(), point, parameter)
    torch.testing.assert_close(result, parameter * point[0] + 1)
    torch.testing.assert_close(torch.autograd.grad(result, parameter)[0], point[0])
    actual = jax.jit(lambda p: jmixed(Provider(), jnp.array([0.3, 0.2]), p))(jnp.array(0.7))
    np.testing.assert_allclose(np.asarray(actual), result.detach().numpy(), atol=1e-16)


def test_manufactured_legacy_scalar_and_explicit_live_opt_in() -> None:
    from omnibias.core.parameter_jets import mixed_jet as core_mixed
    coordinates = (0.300000000000001, 0.200000000000003)
    parameter = 0.700000000000007
    expected = core_mixed(None, coordinates, parameter)
    assert tmixed(None, coordinates, parameter) == expected
    assert jmixed(None, coordinates, parameter) == expected
    point = torch.tensor(coordinates, dtype=torch.float64)
    mu = torch.tensor(parameter, dtype=torch.float64, requires_grad=True)
    legacy = tmixed(None, point, mu)
    assert type(legacy) is float and legacy == expected
    live = tmixed(None, point, mu, live=True)
    assert isinstance(live, torch.Tensor) and live.requires_grad
    live_jax = jax.jit(lambda p: jmixed(None, jnp.asarray(coordinates), p, live=True))(jnp.asarray(parameter))
    np.testing.assert_allclose(np.asarray(live_jax), live.detach().numpy(), atol=1e-16)


def test_disabling_fastpath_does_not_reactivate_inherited_tower() -> None:
    from omnibias.jax.jet import _sigma_tower as jtower
    from omnibias.torch.jet import _sigma_tower as ttower
    with pytest.raises(ValueError, match="fastpath"):
        ttower(replace(get_activation("tanh"), fastpath=None), torch.tensor(0.3), 1)
    with pytest.raises(ValueError, match="fastpath"):
        jtower(replace(jactivation("tanh"), fastpath=None), jnp.asarray(0.3), 1)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation suite for the torch Faà di Bruno jet kernel.

Oracles (float64): single-layer reduction to the closed-form fast path, nested
``torch.func.jacfwd``, the Bell-polynomial decomposition vs the shifted-power
kernel, the dense (pre-reduction) composition kernel as a bit-for-bit value
oracle, and the order-cap error path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.core.bell import faa_di_bruno_terms  # noqa: E402
from omnibias.torch.activations.registry import (  # noqa: E402
    get_activation,
    list_activations,
)
from omnibias.torch.jet import (  # noqa: E402
    _path_jet,
    _sigma_tower,
    affine_jet,
    antiderivative_jet,
    compose_jet,
    compose_jet_riccati,
    derivative_jet,
    jet_to_tower,
    layer_jet,
    mlp_jet,
)
from torch.func import jacfwd  # noqa: E402


def _dense_compose_jet(u_jet, sigma_tower):
    """The dense shifted-power kernel ``compose_jet`` replaced.

    Kept verbatim as the value oracle: it forms every ``(k, n)`` product,
    including the ones that vanish because ``v = u - u_0`` has valuation 1.
    """
    np1 = u_jet.shape[0]
    zero = torch.zeros_like(u_jet[0])
    w = [zero] + [u_jet[j] for j in range(1, np1)]
    p = [torch.ones_like(u_jet[0])] + [zero for _ in range(np1 - 1)]
    result = [sigma_tower[0] * p[0]] + [zero for _ in range(np1 - 1)]
    fact = 1.0
    for k in range(1, np1):
        fact *= k
        new_p = []
        for n in range(np1):
            acc = zero
            for i in range(n + 1):
                acc = acc + p[i] * w[n - i]
            new_p.append(acc)
        p = new_p
        dk = sigma_tower[k] / fact
        for n in range(np1):
            result[n] = result[n] + dk * p[n]
    return torch.stack(result, dim=0)


_RICCATI_NAMES = sorted(
    {name for name in list_activations() if get_activation(name).riccati_polynomial}
)
# Base points that keep tan / cot / coth away from their poles.
_RICCATI_BASE = {"tan": 0.4, "cot": 1.0, "coth": 1.2}


@pytest.fixture(autouse=True)
def _default_float64():
    """Run these float64 oracle tests without leaking the global default dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _build_mlp(seed: int = 0, dims=(3, 5, 4, 2), act: str = "tanh"):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        W = torch.as_tensor(rng.normal(scale=0.7, size=(dout, din)))
        b = torch.as_tensor(rng.normal(scale=0.5, size=(dout,)))
        spec = None if i == len(dims) - 2 else get_activation(act)
        layers.append((W, b, spec))
    x0 = torch.as_tensor(rng.normal(size=(dims[0],)))
    v = torch.as_tensor(rng.normal(size=(dims[0],)))
    return layers, x0, v


def _forward(layers):
    def f(x):
        z = x
        for W, b, spec in layers:
            z = W @ z + b
            if spec is not None:
                z = spec.forward(z)
        return z

    return f


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "gaussian", "cosh"])
def test_single_layer_reduces_to_fastpath(act: str) -> None:
    rng = np.random.default_rng(3)
    D, H = 4, 6
    W = torch.as_tensor(rng.normal(size=(H, D)))
    b = torch.as_tensor(rng.normal(size=(H,)))
    x0 = torch.as_tensor(rng.normal(size=(D,)))
    v = torch.as_tensor(rng.normal(size=(D,)))
    spec = get_activation(act)
    order = 6
    tower = jet_to_tower(layer_jet(_path_jet(x0, v, order), W, b, spec, order))

    z0 = W @ x0 + b
    wv = W @ v
    for k in range(order + 1):
        expected = spec.forward(z0) if k == 0 else spec.fastpath(z0, k) * wv**k
        assert torch.allclose(tower[k], expected, rtol=1e-12, atol=1e-12)


def test_deep_mlp_matches_nested_jacfwd() -> None:
    layers, x0, v = _build_mlp(seed=2)
    f = _forward(layers)
    order = 6

    def g(t):
        return f(x0 + t * v)

    def kth(k):
        fn = g
        for _ in range(k):
            fn = jacfwd(fn)
        return fn(torch.tensor(0.0))

    tower = jet_to_tower(mlp_jet(x0, v, layers, order))
    for k in range(order + 1):
        assert torch.allclose(tower[k], kth(k), rtol=1e-9, atol=1e-9)


def test_compose_jet_matches_bell_oracle() -> None:
    rng = np.random.default_rng(11)
    order = 6
    u_jet = torch.as_tensor(rng.normal(size=(order + 1, 3)))
    sigma_tower = torch.as_tensor(rng.normal(size=(order + 1, 3)))
    out = jet_to_tower(compose_jet(u_jet, sigma_tower)).numpy()

    u_jet_np = u_jet.numpy()
    sigma_np = sigma_tower.numpy()
    u_deriv = np.array([math.factorial(i) * u_jet_np[i] for i in range(order + 1)])
    expected = np.empty_like(sigma_np)
    expected[0] = sigma_np[0]
    for n in range(1, order + 1):
        acc = np.zeros(u_jet_np.shape[1])
        for k, exps, coeff in faa_di_bruno_terms(n):
            prod = np.ones(u_jet_np.shape[1])
            for i, e in enumerate(exps, start=1):
                if e:
                    prod = prod * u_deriv[i] ** e
            acc = acc + coeff * sigma_np[k] * prod
        expected[n] = acc
    assert np.allclose(out, expected, rtol=1e-10, atol=1e-10)


def test_order_cap_raises_value_error() -> None:
    rng = np.random.default_rng(5)
    D, H = 3, 4
    W = torch.as_tensor(rng.normal(size=(H, D)))
    b = torch.as_tensor(rng.normal(size=(H,)))
    x0 = torch.as_tensor(rng.normal(size=(D,)))
    v = torch.as_tensor(rng.normal(size=(D,)))
    # arctan caps at order 2, so an order-3 jet must raise.
    with pytest.raises(ValueError, match="does not support order"):
        layer_jet(_path_jet(x0, v, 3), W, b, "arctan", 3)


def test_affine_jet_is_linear_per_order() -> None:
    rng = np.random.default_rng(7)
    order = 4
    z_jet = torch.as_tensor(rng.normal(size=(order + 1, 5)))
    W = torch.as_tensor(rng.normal(size=(3, 5)))
    b = torch.as_tensor(rng.normal(size=(3,)))
    out = affine_jet(z_jet, W, b)
    assert torch.allclose(out[0], W @ z_jet[0] + b)
    for k in range(1, order + 1):
        assert torch.allclose(out[k], W @ z_jet[k])


# ----- two-sided (integral) tower: antiderivative_jet / derivative_jet -----


def test_antiderivative_jet_ftc_part1_roundtrip() -> None:
    """FTC part 1: d/dt of the antiderivative jet recovers the original vector jet."""
    layers, x0, v = _build_mlp(seed=4)
    order = 6
    jet = mlp_jet(x0, v, layers, order)  # (order+1, C)
    anti = antiderivative_jet(jet, constant=0.3)
    assert anti.shape[0] == jet.shape[0] + 1
    assert torch.allclose(anti[0], torch.full_like(jet[0], 0.3))
    back = derivative_jet(anti)
    assert back.shape == jet.shape
    assert torch.allclose(back, jet, rtol=1e-12, atol=1e-12)


def test_antiderivative_jet_of_mlp_jet_matches_mpmath() -> None:
    mpmath = pytest.importorskip("mpmath")
    a, c = 1.1, -0.3
    order = 6
    jet = mlp_jet(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        [(torch.tensor([[a]]), torch.tensor([c]), "tanh")],
        order,
    )  # net(t) = tanh(a t + c)
    anti = antiderivative_jet(jet)  # F(t) = int_0^t net, F(0) = 0

    def big_f(t: object) -> object:
        return (mpmath.log(mpmath.cosh(a * t + c)) - mpmath.log(mpmath.cosh(c))) / a

    with mpmath.workdps(50):
        taylor_f = mpmath.taylor(big_f, 0.0, order + 1)
    got = anti[:, 0].tolist()
    for k in range(order + 2):
        assert abs(got[k] - float(taylor_f[k])) <= 1e-10


def test_derivative_jet_scalar_matches_manual() -> None:
    jet = torch.tensor([2.0, 3.0, 5.0, 7.0])  # a_0 .. a_3
    d = derivative_jet(jet)  # (k+1) a_{k+1}
    assert torch.allclose(d, torch.tensor([3.0, 10.0, 21.0]))


# ----- valuation-reduced kernel vs the dense kernel it replaced -----


@pytest.mark.parametrize("order", list(range(0, 11)))
def test_compose_jet_matches_dense_kernel_bitwise(order: int) -> None:
    """Skipping the structurally-zero products may not move a single bit."""
    rng = np.random.default_rng(1000 + order)
    for _ in range(4):
        u_jet = torch.as_tensor(rng.normal(size=(order + 1, 4)))
        sigma_tower = torch.as_tensor(rng.normal(size=(order + 1, 4)))
        got = compose_jet(u_jet, sigma_tower)
        want = _dense_compose_jet(u_jet, sigma_tower)
        assert torch.equal(got, want)
        # ... and identical bit patterns, i.e. not even a sign-of-zero drift.
        assert torch.equal(got.view(torch.int64), want.view(torch.int64))


def test_compose_jet_bitwise_on_path_jets_and_float32() -> None:
    """Structural zeros (a line jet) and float32 must also be bit-preserved."""
    rng = np.random.default_rng(17)
    order = 7
    u_jet = torch.zeros(order + 1, 3)
    u_jet[0] = torch.as_tensor(rng.normal(size=3))
    u_jet[1] = torch.as_tensor(rng.normal(size=3))
    tower = torch.as_tensor(rng.normal(size=(order + 1, 3)))
    assert torch.equal(compose_jet(u_jet, tower), _dense_compose_jet(u_jet, tower))

    u32 = u_jet.float()
    t32 = tower.float()
    got32 = compose_jet(u32, t32)
    assert got32.dtype == torch.float32
    assert torch.equal(got32, _dense_compose_jet(u32, t32))


def test_compose_jet_uses_fewer_multiplies_than_dense_kernel() -> None:
    """The saving is a measured constant factor (>= 3x at order 16), not O(N^2)."""
    dispatch = pytest.importorskip("torch.utils._python_dispatch")

    class MulCounter(dispatch.TorchDispatchMode):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            self.n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
            if "mul" in str(func):
                self.n += 1
            return func(*args, **(kwargs or {}))

    rng = np.random.default_rng(23)
    order = 16
    u_jet = torch.as_tensor(rng.normal(size=(order + 1, 2)))
    tower = torch.as_tensor(rng.normal(size=(order + 1, 2)))
    with MulCounter() as dense_count:
        _dense_compose_jet(u_jet, tower)
    with MulCounter() as lean_count:
        compose_jet(u_jet, tower)
    assert lean_count.n * 3 <= dense_count.n, (dense_count.n, lean_count.n)


def test_compose_jet_does_not_propagate_inf_to_lower_orders() -> None:
    """The one documented semantic change: no ``inf * 0`` contamination."""
    rng = np.random.default_rng(29)
    order = 4
    u_jet = torch.as_tensor(rng.normal(size=(order + 1, 1)))
    tower = torch.as_tensor(rng.normal(size=(order + 1, 1)))
    tower[order] = math.inf
    dense = _dense_compose_jet(u_jet, tower)
    lean = compose_jet(u_jet, tower)
    assert torch.isnan(dense[:order]).all()  # inf * 0 -> NaN in the dense form
    assert torch.isfinite(lean[:order]).all()
    assert torch.isinf(lean[order]).all()


# ----- Riccati fastpath: sigma' = P(sigma) -----


def test_riccati_registry_is_not_empty() -> None:
    assert {"sigmoid", "tanh", "exp"} <= set(_RICCATI_NAMES)


@pytest.mark.parametrize("act", _RICCATI_NAMES)
def test_compose_jet_riccati_matches_tower_kernel(act: str) -> None:
    spec = get_activation(act)
    poly = spec.riccati_polynomial
    assert poly is not None
    # tan / cot / coth cap their fastpath at order 3, so the tower oracle
    # (and hence the comparison) is only available that far.
    order = 3 if act in _RICCATI_BASE else 8
    rng = np.random.default_rng(31)
    u_jet = torch.zeros(order + 1, 3)
    u_jet[0] = _RICCATI_BASE.get(act, 0.3) + torch.as_tensor(
        rng.normal(scale=0.05, size=3)
    )
    u_jet[1] = torch.as_tensor(rng.normal(scale=0.3, size=3))
    u_jet[2] = torch.as_tensor(rng.normal(scale=0.2, size=3))
    want = compose_jet(u_jet, _sigma_tower(spec, u_jet[0], order))
    got = compose_jet_riccati(u_jet, spec.forward(u_jet[0]), poly)
    assert got.shape == want.shape
    assert torch.allclose(got, want, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("act", sorted(_RICCATI_BASE))
def test_compose_jet_riccati_passes_the_fastpath_order_cap(act: str) -> None:
    """tan / cot / coth: the tower path raises, the Riccati recurrence does not."""
    spec = get_activation(act)
    poly = spec.riccati_polynomial
    assert poly is not None
    order = 7
    z0 = torch.tensor([_RICCATI_BASE[act]])
    d = torch.tensor([0.35])
    u_jet = torch.zeros(order + 1, 1)
    u_jet[0], u_jet[1] = z0, d
    with pytest.raises(ValueError, match="does not support order"):
        _sigma_tower(spec, z0, order)

    got = jet_to_tower(compose_jet_riccati(u_jet, spec.forward(z0), poly))

    def kth(k: int) -> torch.Tensor:
        fn = lambda t: spec.forward(z0 + t * d)  # noqa: E731
        for _ in range(k):
            fn = jacfwd(fn)
        return fn(torch.tensor(0.0))

    for k in range(order + 1):
        assert torch.allclose(got[k], kth(k), rtol=1e-9, atol=1e-9), k


def test_layer_and_mlp_jet_riccati_flag_agrees_with_default() -> None:
    rng = np.random.default_rng(37)
    D, H, order = 3, 4, 7
    W = torch.as_tensor(rng.normal(scale=0.6, size=(H, D)))
    b = torch.as_tensor(rng.normal(scale=0.4, size=(H,)))
    x0 = torch.as_tensor(rng.normal(size=(D,)))
    v = torch.as_tensor(rng.normal(size=(D,)))
    z_jet = _path_jet(x0, v, order)
    assert torch.allclose(
        layer_jet(z_jet, W, b, "tanh", order, riccati=True),
        layer_jet(z_jet, W, b, "tanh", order),
        rtol=1e-11,
        atol=1e-12,
    )
    layers = [
        (W, b, "tanh"),
        (torch.as_tensor(rng.normal(scale=0.6, size=(2, H))), None, "sigmoid"),
        (torch.as_tensor(rng.normal(size=(1, 2))), None, None),
    ]
    assert torch.allclose(
        mlp_jet(x0, v, layers, order, riccati=True),
        mlp_jet(x0, v, layers, order),
        rtol=1e-11,
        atol=1e-12,
    )


def test_riccati_fastpath_backpropagates_like_the_tower_path() -> None:
    rng = np.random.default_rng(41)
    D, H, order = 3, 4, 5
    x0 = torch.as_tensor(rng.normal(size=(D,)))
    v = torch.as_tensor(rng.normal(size=(D,)))
    b = torch.as_tensor(rng.normal(scale=0.4, size=(H,)))
    raw = torch.as_tensor(rng.normal(scale=0.6, size=(H, D)))

    def grad(riccati: bool) -> torch.Tensor:
        W = raw.clone().requires_grad_(True)
        out = mlp_jet(x0, v, [(W, b, "tanh")], order, riccati=riccati).sum()
        return torch.autograd.grad(out, W)[0]

    assert torch.allclose(grad(True), grad(False), rtol=1e-10, atol=1e-12)


def test_riccati_order_zero_and_constant_polynomial() -> None:
    u_jet = torch.tensor([[0.3], [0.5], [0.25], [0.1]])
    # P constant: sigma' = 3 everywhere, so sigma(u) = sigma(u0) + 3 (u - u0).
    got = compose_jet_riccati(u_jet, torch.tensor([2.0]), (3.0,))
    assert torch.allclose(got, torch.tensor([[2.0], [1.5], [0.75], [0.3]]))
    scalar = compose_jet_riccati(u_jet[:1], torch.tensor([2.0]), (0.0, 1.0, -1.0))
    assert scalar.shape == (1, 1)
    assert torch.equal(scalar[0], torch.tensor([2.0]))


def test_riccati_error_paths() -> None:
    u_jet = torch.tensor([[0.3], [0.5]])
    with pytest.raises(ValueError, match="at least one coefficient"):
        compose_jet_riccati(u_jet, torch.tensor([1.0]), ())
    rng = np.random.default_rng(43)
    W = torch.as_tensor(rng.normal(size=(2, 1)))
    with pytest.raises(ValueError, match="not in the Riccati class"):
        layer_jet(u_jet, W, None, "relu", 1, riccati=True)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Non-periodic spectral fractional operators + trainable fractional layers.

Covers the two-sided spectral fractional Laplacian ``(-Delta)^{alpha/2}`` on a
bounded interval (DST-I Dirichlet / DCT-II Neumann), the windowed-FFT operator,
and the ``nn.Module`` (torch) / pytree (jax) layer wrappers. Exactness is checked
on the sine / cosine basis modes (eigenfunctions), the ``alpha=2`` reduction to
``-u''``, the semigroup property ``(-Delta)^{a/2}(-Delta)^{a/2} = (-Delta)^{a}``,
torch<->jax parity, and autograd in the (learnable) order.

float64 throughout (jax x64 enabled in ``conftest``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fractional.jax import layers as jl
from omnibias.fractional.jax.ops import spectral as jsp
from omnibias.fractional.torch import layers as tl
from omnibias.fractional.torch.ops import fractional as tfr
from omnibias.fractional.torch.ops import spectral as tsp

F = torch.float64


def _dirichlet_grid(n: int, length: float) -> torch.Tensor:
    h = length / (n + 1)
    return torch.arange(1, n + 1, dtype=F) * h


def _neumann_grid(n: int, length: float) -> torch.Tensor:
    h = length / n
    return (torch.arange(n, dtype=F) + 0.5) * h


# ===================== spectral fractional Laplacian ========================


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("alpha", [2.0, 1.3, 0.5])
def test_dirichlet_eigenmode_exact(m: int, alpha: float) -> None:
    n, length = 40, 1.0
    x = _dirichlet_grid(n, length)
    f = torch.sin(m * torch.pi * x / length)
    out = tsp.spectral_fractional_laplacian(f, alpha=alpha, length=length, bc="dirichlet")
    eig = (m * torch.pi / length) ** alpha
    assert torch.allclose(out, eig * f, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("m", [0, 1, 2, 3])
@pytest.mark.parametrize("alpha", [2.0, 1.4])
def test_neumann_eigenmode_exact(m: int, alpha: float) -> None:
    n, length = 40, 1.3
    x = _neumann_grid(n, length)
    f = torch.cos(m * torch.pi * x / length)
    out = tsp.spectral_fractional_laplacian(f, alpha=alpha, length=length, bc="neumann")
    eig = (m * torch.pi / length) ** alpha  # m=0 -> 0
    assert torch.allclose(out, eig * f, rtol=1e-10, atol=1e-10)


def test_dirichlet_alpha_two_recovers_neg_second_derivative() -> None:
    # A superposition of Dirichlet modes; (-Delta)^1 = -u''.
    n, length = 64, 2.0
    x = _dirichlet_grid(n, length)
    coeffs = {1: 1.0, 3: -0.5, 5: 0.25}
    f = sum(c * torch.sin(m * torch.pi * x / length) for m, c in coeffs.items())
    neg_upp = sum(
        c * (m * torch.pi / length) ** 2 * torch.sin(m * torch.pi * x / length)
        for m, c in coeffs.items()
    )
    out = tsp.spectral_fractional_laplacian(f, alpha=2.0, length=length, bc="dirichlet")
    assert torch.allclose(out, neg_upp, rtol=1e-9, atol=1e-9)


def test_semigroup_property_dirichlet() -> None:
    # (-Delta)^{a/2} (-Delta)^{a/2} f = (-Delta)^{a} f  (symbol xi^a . xi^a = xi^{2a}).
    n, length, a = 48, 1.0, 0.8
    x = _dirichlet_grid(n, length)
    rng = np.random.default_rng(0)
    f = torch.zeros(n, dtype=F)
    for m in range(1, 8):
        f = f + float(rng.normal()) * torch.sin(m * torch.pi * x / length)
    once = tsp.spectral_fractional_laplacian(f, alpha=a, length=length, bc="dirichlet")
    twice = tsp.spectral_fractional_laplacian(once, alpha=a, length=length, bc="dirichlet")
    direct = tsp.spectral_fractional_laplacian(f, alpha=2 * a, length=length, bc="dirichlet")
    assert torch.allclose(twice, direct, rtol=1e-9, atol=1e-9)


def test_laplacian_order_gradient_matches_finite_difference() -> None:
    n, length = 32, 1.0
    x = _dirichlet_grid(n, length)
    f = torch.sin(torch.pi * x / length) + 0.3 * torch.sin(3 * torch.pi * x / length)
    a0 = 1.1
    alpha = torch.tensor(a0, dtype=F, requires_grad=True)
    tsp.spectral_fractional_laplacian(f, alpha=alpha, length=length).sum().backward()
    eps = 1e-6
    fp = float(tsp.spectral_fractional_laplacian(f, alpha=a0 + eps, length=length).sum())
    fm = float(tsp.spectral_fractional_laplacian(f, alpha=a0 - eps, length=length).sum())
    assert abs(float(alpha.grad) - (fp - fm) / (2 * eps)) < 1e-6


def test_laplacian_torch_jax_parity_value_and_grad() -> None:
    n, length, a0 = 40, 1.0, 1.2
    x_np = np.arange(1, n + 1) * (length / (n + 1))
    f_np = np.sin(np.pi * x_np / length) + 0.4 * np.sin(4 * np.pi * x_np / length)

    vt = tsp.spectral_fractional_laplacian(
        torch.as_tensor(f_np, dtype=F), alpha=a0, length=length
    ).numpy()
    vj = np.asarray(
        jsp.spectral_fractional_laplacian(jnp.asarray(f_np), alpha=a0, length=length)
    )
    assert np.allclose(vt, vj, rtol=1e-10, atol=1e-11)

    at = torch.tensor(a0, dtype=F, requires_grad=True)
    tsp.spectral_fractional_laplacian(
        torch.as_tensor(f_np, dtype=F), alpha=at, length=length
    ).sum().backward()
    gj = jax.grad(
        lambda al: jsp.spectral_fractional_laplacian(
            jnp.asarray(f_np), alpha=al, length=length
        ).sum()
    )(a0)
    assert np.allclose(float(at.grad), float(gj), rtol=1e-6, atol=1e-8)


def test_laplacian_error_paths() -> None:
    f = torch.ones(8, dtype=F)
    with pytest.raises(ValueError, match="length must be"):
        tsp.spectral_fractional_laplacian(f, alpha=1.0, length=0.0)
    with pytest.raises(ValueError, match="bc must be"):
        tsp.spectral_fractional_laplacian(f, alpha=1.0, length=1.0, bc="periodic")
    with pytest.raises(ValueError, match="f must be 1-D"):
        tsp.spectral_fractional_laplacian(torch.ones(4, 4, dtype=F), alpha=1.0, length=1.0)


# ============================ windowed FFT ==================================


def test_tukey_window_edge_cases() -> None:
    assert torch.allclose(tsp.tukey_window(16, 0.0, F, torch.device("cpu")), torch.ones(16, dtype=F))
    hann = tsp.tukey_window(32, 1.0, F, torch.device("cpu"))
    assert float(hann[0]) < 1e-12 and float(hann[-1]) < 1e-12
    assert torch.all((hann >= 0.0) & (hann <= 1.0 + 1e-12))
    assert torch.allclose(hann, hann.flip(0), atol=1e-12)  # symmetric
    with pytest.raises(ValueError, match="taper must be"):
        tsp.tukey_window(16, 1.5, F, torch.device("cpu"))


def test_windowed_taper_zero_is_plain_spectral() -> None:
    n, length = 64, 2.0
    x = torch.linspace(0.0, length, n, dtype=F)
    f = torch.exp(-((x - 1.0) ** 2) / 0.1)
    a = tsp.windowed_spectral_fractional(f, alpha=1.0, length=length, taper=0.0)
    b = tfr.spectral_fractional(f, alpha=1.0, length=length)
    assert torch.allclose(a, b, rtol=1e-12, atol=1e-12)


def test_windowed_matches_analytic_derivative_interior() -> None:
    # A Gaussian bump that decays to ~0 at both ends; alpha=1 -> first derivative.
    n, length, c, w = 512, 4.0, 2.0, 0.35
    x = torch.linspace(0.0, length, n, dtype=F)
    f = torch.exp(-((x - c) ** 2) / (2 * w**2))
    dfdx = -(x - c) / w**2 * f
    out = tsp.windowed_spectral_fractional(f, alpha=1.0, length=length, taper=0.2).real
    sl = slice(n // 4, 3 * n // 4)
    rel = torch.max(torch.abs(out[sl] - dfdx[sl])) / torch.max(torch.abs(dfdx))
    assert float(rel) < 5e-3


def test_windowed_torch_jax_parity() -> None:
    n, length = 128, 3.0
    x_np = np.linspace(0.0, length, n)
    f_np = np.exp(-((x_np - 1.5) ** 2) / 0.2)
    vt = tsp.windowed_spectral_fractional(
        torch.as_tensor(f_np, dtype=F), alpha=1.4, length=length, taper=0.15
    ).numpy()
    vj = np.asarray(
        jsp.windowed_spectral_fractional(jnp.asarray(f_np), alpha=1.4, length=length, taper=0.15)
    )
    assert np.allclose(vt, vj, rtol=1e-9, atol=1e-11)


# ============================ torch layers ==================================


def test_torch_laplacian_layer_forward_and_train_step() -> None:
    n, length = 40, 1.0
    x = _dirichlet_grid(n, length)
    f = torch.sin(torch.pi * x / length)
    layer = tl.SpectralFractionalLaplacianLayer(
        length=length, bc="dirichlet", order=1.3, learnable_order=True
    )
    out = layer(f)
    eig = (torch.pi / length) ** 1.3
    assert torch.allclose(out, eig * f, rtol=1e-4, atol=1e-4)

    # one training step moves the order.
    target = (torch.pi / length) ** 0.7 * f
    opt = torch.optim.SGD(layer.parameters(), lr=0.5)
    before = float(layer.alpha.detach())
    for _ in range(20):
        opt.zero_grad()
        loss = (layer(f) - target).pow(2).sum()
        loss.backward()
        assert layer.order_module.raw.grad is not None
        opt.step()
    assert abs(float(layer.alpha.detach()) - before) > 1e-4


def test_torch_spectral_layer_matches_op_and_fixed_order() -> None:
    n, length = 64, 2.0
    x = torch.linspace(0.0, length, n, dtype=F)
    f = torch.sin(2 * torch.pi * x / length)
    layer = tl.SpectralFractionalLayer(
        length=length, order=1.0, learnable_order=False, real=True, dtype=F
    )
    assert torch.allclose(layer.alpha, torch.tensor(1.0, dtype=F))
    got = layer(f)
    ref = tfr.spectral_fractional(f, alpha=layer.alpha, length=length).real
    assert torch.allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_torch_gl_layer_matches_op() -> None:
    n, h = 50, 0.05
    f = torch.linspace(0.0, 1.0, n, dtype=F) ** 2
    layer = tl.GrunwaldLetnikovLayer(h=h, order=0.5, learnable_order=False, dtype=F)
    got = layer(f)
    ref = tfr.grunwald_letnikov(f, alpha=layer.alpha, h=h)
    assert torch.allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_torch_layer_error_paths() -> None:
    with pytest.raises(ValueError, match="grid spacing h"):
        tl.GrunwaldLetnikovLayer(h=0.0)
    with pytest.raises(ValueError, match="length must be"):
        tl.SpectralFractionalLayer(length=-1.0)
    with pytest.raises(ValueError, match="bc must be"):
        tl.SpectralFractionalLaplacianLayer(length=1.0, bc="periodic")


# ============================= jax layers ===================================


def test_jax_laplacian_layer_forward_and_grad() -> None:
    n, length = 40, 1.0
    x_np = np.arange(1, n + 1) * (length / (n + 1))
    f = jnp.asarray(np.sin(np.pi * x_np / length))
    layer = jl.SpectralFractionalLaplacianLayer.from_order(1.3, length=length, bc="dirichlet")
    out = layer(f)
    eig = (np.pi / length) ** 1.3
    assert np.allclose(np.asarray(out), eig * np.asarray(f), rtol=1e-6, atol=1e-7)

    target = jnp.asarray((np.pi / length) ** 0.7 * np.sin(np.pi * x_np / length))

    def loss(lyr: jl.SpectralFractionalLaplacianLayer) -> jax.Array:
        return jnp.sum((lyr(f) - target) ** 2)

    grad_layer = jax.grad(loss)(layer)
    assert np.isfinite(float(grad_layer.raw_order)) and abs(float(grad_layer.raw_order)) > 1e-6


def test_jax_layer_pytree_roundtrip() -> None:
    layer = jl.SpectralFractionalLayer.from_order(1.2, length=2.0, real=True)
    leaves, treedef = jax.tree_util.tree_flatten(layer)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.allclose(float(rebuilt.alpha), float(layer.alpha))
    assert rebuilt.length == layer.length and rebuilt.real == layer.real


def test_jax_gl_layer_matches_op() -> None:
    from omnibias.fractional.jax.ops import fractional as jfr

    n, h = 40, 0.05
    f = jnp.asarray(np.linspace(0.0, 1.0, n) ** 2)
    layer = jl.GrunwaldLetnikovLayer.from_order(0.5, h=h)
    got = np.asarray(layer(f))
    ref = np.asarray(jfr.grunwald_letnikov(f, alpha=layer.alpha, h=h))
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-7)

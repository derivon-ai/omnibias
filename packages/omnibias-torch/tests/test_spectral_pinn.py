# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Spectral-bias-mitigating omnibias PINNs: ``FourierFeatureMLP`` and ``make_siren``.

Two things are checked:

1. *Exactness*: the closed-form multivariate-jet derivatives of the sin-encoded
   Fourier-feature net and of the SIREN match ``torch.func`` autograd to ~machine
   precision (float64). The whole point of routing the encoding through a single
   omnibias ``sin`` layer is that ``D^alpha u(x)`` stays exact to arbitrary order.
2. *Spectral bias*: on a high-frequency target both constructs reach a far lower
   fit error than a plain ``tanh`` :class:`JetMLP` in the same training budget --
   the measured evidence that the construct does what it claims.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.architectures import FourierFeatureMLP, JetMLP, make_siren


def _val_fn(net: FourierFeatureMLP | JetMLP):
    def val(xi: torch.Tensor) -> torch.Tensor:
        return net.value(xi).reshape(())

    return val


# --- FourierFeatureMLP: exact derivatives ---------------------------------


def test_fourier_value_equals_jet_row0() -> None:
    net = FourierFeatureMLP(2, num_features=8, hidden=16, depth=2, seed=1).double()
    x = torch.randn(6, 2, dtype=torch.float64)
    jet = net.jet(x, 2)
    assert torch.allclose(jet[:, 0, :], net.value(x), atol=1e-13)


def test_fourier_encoding_is_cos_sin() -> None:
    """First layer output must be exactly ``[cos(B x), sin(B x)]``."""
    net = FourierFeatureMLP(3, num_features=5, hidden=8, depth=1, seed=2).double()
    x = torch.randn(4, 3, dtype=torch.float64)
    f_total = net.num_features * len(net.scales)
    b_mat = net.W_ff[:f_total]  # W_ff = [B; B]
    z = x @ b_mat.t()
    feats = torch.sin(x @ net.W_ff.t() + net.b_ff)
    assert torch.allclose(feats[:, :f_total], torch.cos(z), atol=1e-13)
    assert torch.allclose(feats[:, f_total:], torch.sin(z), atol=1e-13)


def test_fourier_gradient_matches_autograd() -> None:
    net = FourierFeatureMLP(2, num_features=8, hidden=16, depth=2, seed=3).double()
    x = torch.randn(7, 2, dtype=torch.float64)
    g = net.gradient(x).squeeze(-1)
    g_ad = torch.func.vmap(torch.func.jacrev(_val_fn(net)))(x)
    assert torch.allclose(g, g_ad, atol=1e-10)


def test_fourier_hessian_matches_autograd() -> None:
    net = FourierFeatureMLP(2, num_features=8, hidden=16, depth=2, seed=4).double()
    x = torch.randn(5, 2, dtype=torch.float64)
    h = net.hessian(x).squeeze(-1)
    h_ad = torch.func.vmap(torch.func.hessian(_val_fn(net)))(x)
    assert torch.allclose(h, h_ad, atol=1e-9)
    assert torch.allclose(h, h.transpose(1, 2), atol=1e-11)


def test_fourier_third_order_matches_autograd() -> None:
    net = FourierFeatureMLP(2, num_features=6, hidden=12, depth=2, seed=5).double()
    x = torch.randn(4, 2, dtype=torch.float64)
    parts = net.partials(x, 3)
    jf = torch.func.jacfwd
    t3 = torch.func.vmap(jf(jf(jf(_val_fn(net)))))(x)
    assert torch.allclose(parts[(3, 0)].squeeze(-1), t3[:, 0, 0, 0], atol=1e-8)
    assert torch.allclose(parts[(2, 1)].squeeze(-1), t3[:, 0, 0, 1], atol=1e-8)
    assert torch.allclose(parts[(0, 3)].squeeze(-1), t3[:, 1, 1, 1], atol=1e-8)


@pytest.mark.parametrize("scales", [1.0, (0.5, 2.0), (0.25, 1.0, 4.0)])
def test_fourier_multiscale_shapes_and_grad(scales: float | tuple[float, ...]) -> None:
    net = FourierFeatureMLP(
        2, num_features=4, hidden=10, depth=2, frequency_scale=scales, seed=6
    ).double()
    n_bands = 1 if isinstance(scales, float) else len(scales)
    assert net.feature_dim == 2 * 4 * n_bands
    x = torch.randn(5, 2, dtype=torch.float64)
    g = net.gradient(x).squeeze(-1)
    g_ad = torch.func.vmap(torch.func.jacrev(_val_fn(net)))(x)
    assert torch.allclose(g, g_ad, atol=1e-10)


def test_fourier_depth0_pure_rff_matches_autograd() -> None:
    """``depth=0`` is a pure random-Fourier-feature model with a linear readout."""
    net = FourierFeatureMLP(1, num_features=16, depth=0, frequency_scale=3.0, seed=7).double()
    x = torch.randn(6, 1, dtype=torch.float64)
    g = net.gradient(x).reshape(-1)
    g_ad = torch.func.vmap(torch.func.jacrev(_val_fn(net)))(x).reshape(-1)
    assert torch.allclose(g, g_ad, atol=1e-9)


def test_fourier_features_buffer_by_default() -> None:
    net = FourierFeatureMLP(2, num_features=4, hidden=8, depth=1, seed=8)
    names = dict(net.named_buffers())
    assert "W_ff" in names and "b_ff" in names
    assert "W_ff" not in dict(net.named_parameters())


def test_fourier_trainable_features_grad_flows() -> None:
    net = FourierFeatureMLP(
        2, num_features=4, hidden=8, depth=1, trainable_features=True, seed=9
    )
    assert "W_ff" in dict(net.named_parameters())
    x = torch.randn(5, 2)
    loss = net.value(x).pow(2).mean()
    loss.backward()
    assert net.W_ff.grad is not None and torch.isfinite(net.W_ff.grad).all()


def test_fourier_base_without_fastpath_raises() -> None:
    nofp = dataclasses.replace(get_activation("tanh"), name="tanh_nofp", fastpath=None)
    net = FourierFeatureMLP(2, num_features=4, hidden=8, depth=2, base=nofp)
    with pytest.raises(ValueError, match="closed-form derivative"):
        net.gradient(torch.randn(3, 2))


def test_fourier_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        FourierFeatureMLP(0, num_features=4)
    with pytest.raises(ValueError):
        FourierFeatureMLP(2, num_features=0)
    with pytest.raises(ValueError):
        FourierFeatureMLP(2, num_features=4, depth=-1)
    with pytest.raises(ValueError):
        FourierFeatureMLP(2, num_features=4, frequency_scale=-1.0)
    with pytest.raises(ValueError):
        FourierFeatureMLP(2, num_features=4, frequency_scale=())


# --- SIREN: exact derivatives ---------------------------------------------


def test_siren_gradient_matches_autograd() -> None:
    net = make_siren(1, hidden=24, depth=3, omega_0=10.0, seed=2).double()
    x = torch.randn(6, 1, dtype=torch.float64)
    g = net.gradient(x).reshape(-1)
    g_ad = torch.func.vmap(torch.func.jacrev(_val_fn(net)))(x).reshape(-1)
    assert torch.allclose(g, g_ad, atol=1e-10)


def test_siren_high_order_matches_autograd() -> None:
    net = make_siren(2, hidden=16, depth=2, omega_0=8.0, seed=3).double()
    x = torch.randn(4, 2, dtype=torch.float64)
    parts = net.partials(x, 4)
    jf = torch.func.jacfwd
    t4 = torch.func.vmap(jf(jf(jf(jf(_val_fn(net))))))(x)
    assert torch.allclose(parts[(4, 0)].squeeze(-1), t4[:, 0, 0, 0, 0], atol=1e-7)
    assert torch.allclose(parts[(2, 2)].squeeze(-1), t4[:, 0, 0, 1, 1], atol=1e-7)


def test_siren_first_layer_scaled_by_omega0() -> None:
    """The omega_0 fold makes the first-layer weights ~omega_0x larger than hidden ones."""
    net = make_siren(1, hidden=64, depth=3, omega_0=30.0, seed=0)
    first = net.linears[0].weight.abs().mean().item()
    hidden = net.linears[1].weight.abs().mean().item()
    assert first > 5.0 * hidden  # omega_0=30 fold dominates the sqrt(6/n)/omega_0 hidden init


def test_make_siren_invalid_omega_raises() -> None:
    with pytest.raises(ValueError):
        make_siren(1, hidden=8, depth=2, omega_0=0.0)


# --- Spectral-bias mitigation: measured evidence --------------------------


def _target_high_freq(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * math.pi * 5.0 * x) + 0.3 * torch.sin(2 * math.pi * 1.0 * x)


def _fit_value(model: torch.nn.Module, *, steps: int = 400, lr: float = 5e-3, seed: int = 0) -> float:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_tr = torch.linspace(0.0, 1.0, 128).unsqueeze(-1)
    y_tr = _target_high_freq(x_tr)
    for _ in range(steps):
        opt.zero_grad()
        loss = (model(x_tr) - y_tr).pow(2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x_te = torch.linspace(0.0, 1.0, 512).unsqueeze(-1)
        return (model(x_te) - _target_high_freq(x_te)).pow(2).mean().item()


def test_fourier_features_beat_tanh_on_high_frequency() -> None:
    """Random Fourier features fit a 5-cycle target the plain tanh MLP cannot."""
    mse_tanh = _fit_value(JetMLP(1, 48, 1, depth=3, base="tanh"))
    mse_ff = _fit_value(
        FourierFeatureMLP(1, num_features=48, hidden=48, depth=2, frequency_scale=5.0, seed=0)
    )
    assert mse_tanh > 5e-2  # tanh is stuck on the low-frequency component
    assert mse_ff < 5e-3
    assert mse_ff < 0.1 * mse_tanh


def test_siren_beats_tanh_on_high_frequency() -> None:
    mse_tanh = _fit_value(JetMLP(1, 48, 1, depth=3, base="tanh"))
    mse_siren = _fit_value(make_siren(1, 48, 1, depth=3, omega_0=15.0, seed=0))
    assert mse_tanh > 5e-2
    assert mse_siren < 5e-3
    assert mse_siren < 0.1 * mse_tanh

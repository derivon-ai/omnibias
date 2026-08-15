# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Transmission interface field tests (theory 02-05)."""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.pinn.interface import (
    Interface,
    TransmissionInterface,
    order_for_condition,
    smoothing_error_bound,
)
from omnibias.pinn.interface.torch import MultiInterfaceField


def _affine(a: float, b: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return a + b * x[..., 0]

    return fn


def test_order_for_condition_and_alias() -> None:
    assert order_for_condition("value") == 0
    assert order_for_condition("flux") == 1
    assert order_for_condition("curvature") == 2
    assert TransmissionInterface is Interface


def test_g1_sharp_limit_piecewise_linear() -> None:
    torch.set_default_dtype(torch.float64)
    alpha = 1.0e8
    c = -0.3
    b = 0.5
    a = 0.8 - abs(c) * math.log(2.0) / alpha
    iface = Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=alpha, jump=-0.6)
    field = MultiInterfaceField(_affine(a, b), [iface], hard=True, dtype=torch.float64)
    xs = torch.linspace(-1.0, 1.0, 81, dtype=torch.float64).reshape(-1, 1)
    pred = field(xs).detach().numpy().reshape(-1)
    x = xs.numpy().reshape(-1)
    exact = np.where(x <= 0.0, 0.8 * (x + 1.0), 0.8 + 0.2 * x)
    far = np.abs(x) > 1.0e-3
    rel = float(np.linalg.norm(pred[far] - exact[far]) / np.linalg.norm(exact[far]))
    assert rel <= 1e-10


def test_g2_smoothing_rate_and_bound() -> None:
    torch.set_default_dtype(torch.float64)
    c = -0.3
    b = 0.5
    alphas = (20.0, 40.0, 80.0, 160.0)
    errs = []
    for alpha in alphas:
        a = 0.8 - abs(c) * math.log(2.0) / alpha
        iface = Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=alpha, jump=-0.6)
        field = MultiInterfaceField(_affine(a, b), [iface], hard=True, dtype=torch.float64)
        u0 = float(field(torch.zeros(1, 1, dtype=torch.float64)))
        err = abs(u0 - 0.8)
        bound = smoothing_error_bound(iface, coeff=c)
        assert err <= float(bound.hi) + 1e-15
        errs.append(err)
    # First-order in 1/alpha: err*alpha ~ constant
    scaled = [e * al for e, al in zip(errs, alphas, strict=True)]
    assert max(scaled) / min(scaled) < 1.2


def test_g4_hard_residuals_without_interface_loss() -> None:
    torch.set_default_dtype(torch.float64)
    iface = Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=50.0, jump=-0.6)
    field = MultiInterfaceField(_affine(0.8, 0.5), [iface], hard=True, dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64).reshape(-1, 1)
    res = field.interface_residuals(x)
    assert 0 in res
    # tanh(8) is 1 to many digits; residual is the smoothing floor.
    assert abs(float(res[0])) < 1e-6


def test_g5_torch_jax_parity() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.pinn.interface.jax import hard_coeffs, multi_interface_apply

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    iface = Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=40.0, jump=-0.6)
    a, b = 0.79, 0.5
    field = MultiInterfaceField(_affine(a, b), [iface], hard=True, dtype=torch.float64)
    x_np = np.linspace(-1.0, 1.0, 11, dtype=np.float64).reshape(-1, 1)
    t = field(torch.as_tensor(x_np)).detach().numpy()

    def base(x: jnp.ndarray) -> jnp.ndarray:
        return a + b * x[..., 0]

    j = np.asarray(
        multi_interface_apply(
            jnp.asarray(x_np),
            base=base,
            interfaces=[iface],
            coeffs=hard_coeffs([iface]),
        )
    )
    assert np.allclose(t, j, rtol=0, atol=1e-12)


def test_mixed_condition_orders() -> None:
    v = Interface(normal=(1.0,), offset=-0.5, condition="value", jump=0.2)
    f = Interface(normal=(1.0,), offset=0.0, condition="flux", jump=-0.6)
    c = Interface(normal=(1.0,), offset=0.5, condition="curvature", jump=0.4)
    assert v.order == 0 and f.order == 1 and c.order == 2
    from omnibias.pinn.interface._core import profile

    xs = np.linspace(-1.0, 1.0, 9)
    mat = np.stack(
        [
            np.array([profile(int(v.order or 0), float(x + 0.5), 20.0) for x in xs]),
            np.array([profile(int(f.order or 1), float(x), 20.0) for x in xs]),
            np.array([profile(int(c.order or 2), float(x - 0.5), 20.0) for x in xs]),
        ]
    )
    rank = int(np.linalg.matrix_rank(mat, tol=1e-8))
    assert rank == 3

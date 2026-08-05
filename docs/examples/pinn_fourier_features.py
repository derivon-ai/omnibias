# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Beating spectral bias with Fourier-feature PINN fields -- omnibias.pinn.

Run:

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_fourier_features.py

Standard PINNs suffer **spectral bias**: gradient descent fits the low-frequency
part of a solution long before the high-frequency part, so a target with a thin
layer or an oscillatory component converges painfully slowly. The cure (Tancik et
al. 2020) is to lift the input through a random Fourier-feature encoding
``gamma(x) = [cos(B x), sin(B x)]`` before the network body.

The omnibias twist is that the cure is **free in the differential operator**.
Because ``cos(z) = sin(z + pi/2)``, the encoding is a single ``sin`` layer, and
``sin^{(n)}(z) = sin(z + n pi/2)`` is closed form at every order. So the PDE
residual of a :class:`FourierFeatureVectorField` still comes from one exact
multivariate jet -- no ``torch.autograd.grad`` anywhere in the operator, at any
depth or order.

The problem: 1D Poisson with a two-scale exact solution,

    u''(x) = f(x)  on (0, 1),   u(0) = u(1) = 0,
    u*(x)  = sin(2 pi x) + 0.2 sin(14 pi x),

so ``f(x) = -(2 pi)^2 sin(2 pi x) - 0.2 (14 pi)^2 sin(14 pi x)``. Three fields are
trained identically:

* ``OneLayerVectorField``     -- the incumbent single-hidden-layer field.
* ``JetMLPVectorField``       -- a deep tanh MLP, same closed-form jet path.
* ``FourierFeatureVectorField`` -- the same body behind a two-band encoding.

Projecting each solution onto ``sin(2 pi x)`` and ``sin(14 pi x)`` separates the
two scales. The single-layer field shows the textbook spectral-bias signature --
it recovers the smooth band and about 5% of the oscillation. Depth helps: the deep
MLP does find the oscillation's amplitude, but its solution is still three orders
of magnitude less accurate overall. Only the Fourier encoding gets both bands and
the shape right.

Honesty: the derivatives are exact (closed-form jets); the *solution* is trained,
not certified. The reported numbers are training outcomes on a fixed seed, not
bounds.
"""

from __future__ import annotations

import math

import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.fields import (
    FourierFeatureVectorField,
    JetMLPVectorField,
    OneLayerVectorField,
)

DTYPE = torch.float64
LOW, HIGH, AMP = 2.0, 14.0, 0.2


def exact(x: torch.Tensor) -> torch.Tensor:
    """``u*(x) = sin(2 pi x) + AMP sin(14 pi x)``, evaluated on ``(B, 1)`` coords."""
    xs = x[:, 0]
    return torch.sin(LOW * math.pi * xs) + AMP * torch.sin(HIGH * math.pi * xs)


def source(x: torch.Tensor) -> torch.Tensor:
    """``f = u*''``."""
    xs = x[:, 0]
    return -((LOW * math.pi) ** 2) * torch.sin(LOW * math.pi * xs) - AMP * (
        HIGH * math.pi
    ) ** 2 * torch.sin(HIGH * math.pi * xs)


def train(field: torch.nn.Module, *, steps: int = 3000, lr: float = 3e-3) -> None:
    """Adam on ``MSE((u'' - f) / |f|_max) + MSE(u on the boundary)``.

    The residual is non-dimensionalised by ``|f|_max ~ (14 pi)^2``; without that
    the loss scale depends on the highest frequency in the source and no single
    learning rate serves all three architectures fairly.
    """
    x_int = torch.linspace(0.0, 1.0, 129, dtype=DTYPE).reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
    f_int = source(x_int)
    scale = float(f_int.abs().max())
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pde = ((ops.laplacian(field(x_int), "u") - f_int) / scale).pow(2).mean()
        bc = ops.value(field(x_bc), "u").pow(2).mean()
        (pde + bc).backward()
        opt.step()


def report(field: torch.nn.Module) -> tuple[float, float, float]:
    """Relative L2 error plus the learned amplitude in each of the two bands.

    The two projections ``2 <u, sin(k pi x)>`` are what separate "learned the
    smooth part" from "learned the oscillation"; the targets are ``1.0`` and
    ``AMP`` respectively.
    """
    xg = torch.linspace(0.0, 1.0, 401, dtype=DTYPE).reshape(-1, 1)
    with torch.no_grad():
        u = ops.value(field(xg), "u")
        target = exact(xg)
        rel_l2 = float((u - target).pow(2).mean().sqrt() / target.pow(2).mean().sqrt())
        low = float(2.0 * (u * torch.sin(LOW * math.pi * xg[:, 0])).mean())
        high = float(2.0 * (u * torch.sin(HIGH * math.pi * xg[:, 0])).mean())
    return rel_l2, low, high


def main() -> None:
    cs, comp = CoordinateSpec(("x",)), ComponentSpec(("u",))
    print("=== spectral bias on 1D Poisson: u* = sin(2 pi x) + 0.2 sin(14 pi x) ===")
    results: dict[str, tuple[float, float, float]] = {}

    torch.manual_seed(0)
    one = OneLayerVectorField(
        coordinate_spec=cs, components=comp, hidden=64, base="tanh", dtype=DTYPE
    )
    train(one)
    results["OneLayerVectorField"] = report(one)

    torch.manual_seed(0)
    deep = JetMLPVectorField(
        coordinate_spec=cs, components=comp, hidden=32, depth=3, base="tanh"
    )
    train(deep)
    results["JetMLPVectorField"] = report(deep)

    torch.manual_seed(0)
    fourier = FourierFeatureVectorField(
        coordinate_spec=cs,
        components=comp,
        num_features=16,
        hidden=32,
        depth=2,
        frequency_scale=(1.0, 8.0),
        seed=0,
    )
    train(fourier)
    results["FourierFeatureVectorField"] = report(fourier)

    print(f"    target amplitudes: 2-pi band = 1.000, {HIGH:.0f}-pi band = {AMP:.3f}")
    for name, (rel, low, high) in results.items():
        print(
            f"    {name:26s}: rel L2 = {rel:7.4f}   "
            f"2-pi = {low:+.3f}   {HIGH:.0f}-pi = {high:+.4f}"
        )

    ff_rel, _, ff_high = results["FourierFeatureVectorField"]
    deep_rel, _, deep_high = results["JetMLPVectorField"]
    one_rel, _, one_high = results["OneLayerVectorField"]
    print(
        f"\n    Single layer: smooth band recovered, only "
        f"{100 * one_high / AMP:.0f}% of the oscillation -- spectral bias."
        f"\n    Depth alone: {100 * deep_high / AMP:.0f}% of the oscillation, but still "
        f"{deep_rel / ff_rel:.0f}x the error of the encoded field."
        f"\n    Fourier features: {100 * ff_high / AMP:.0f}% of the oscillation and "
        f"rel L2 = {ff_rel:.1e}."
    )

    assert ff_rel < 0.1 * min(deep_rel, one_rel), (ff_rel, deep_rel, one_rel)
    assert abs(ff_high - AMP) < 0.05 * AMP, (ff_high, AMP)
    assert one_high < 0.25 * AMP, (one_high, AMP)

    # The point of the field type: all of this went through the *exact* jet, and a
    # second-order residual costs exactly one jet evaluation per collocation batch.
    x = torch.linspace(0.0, 1.0, 5, dtype=DTYPE).reshape(-1, 1)
    state = fourier(x)
    ops.value(state, "u")
    ops.gradient(state, "u")
    ops.laplacian(state, "u")
    cached = state.extra["_jet_mlp_jets"]
    print(f"    value + gradient + laplacian used {len(cached)} jet (orders {sorted(cached)}).")
    assert sorted(cached) == [2]

    # ... and the closed-form jet agrees with autograd to machine precision.
    xg = torch.linspace(0.05, 0.95, 33, dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
    u = fourier.forward_values(xg)[:, 0]
    du = torch.autograd.grad(u.sum(), xg, create_graph=True)[0][:, 0]
    d2u = torch.autograd.grad(du.sum(), xg)[0][:, 0].detach()
    with torch.no_grad():
        closed = ops.laplacian(fourier(xg.detach()), "u")
        err = float((closed - d2u).abs().max() / d2u.abs().max())
    print(f"    closed-form u'' vs autograd u'': max relative error {err:.2e}")
    assert err < 1e-12

    print("\nSpectral bias broken with the derivative tower intact; all checks passed.")


if __name__ == "__main__":
    main()

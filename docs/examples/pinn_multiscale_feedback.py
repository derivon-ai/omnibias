# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Measuring the frequency bands instead of guessing them -- omnibias.pinn.

Run:

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_multiscale_feedback.py

Every multi-scale PINN construct -- the Fourier encoding, the MscaleDNN band
mixture -- takes a tuple of *band scales*, and in the literature that tuple is a
hyperparameter you guess, almost always the ladder ``1, 2, 4, 8, ...``. This
example shows the guess failing on a problem it cannot reach, and closes the loop
instead: measure where the solution keeps its energy with
``suggest_frequency_bands``, and hand the measurement to the field.

The problem: 1D Poisson with a two-scale solution whose scales are twenty times
apart,

    u''(x) = f(x)  on (0, 1),   u*(x) = sin(2 pi x) + 0.5 sin(40 pi x),

trained against the source plus interior samples of ``u*``. Four fields, same
budget and optimiser:

* ``JetMLPVectorField``           -- plain deep MLP; the spectral-bias baseline.
* ``MscaleVectorField``, ladder   -- bands guessed as ``geometric_bands(2) = (1, 2)``.
* ``MscaleVectorField``, measured -- bands read off the data by ``suggest_frequency_bands``.
* ``AdaptiveJetMLPVectorField``   -- no bands at all; a trainable slope per layer.

Projecting each solution onto the two modes separates "learned the smooth part"
from "learned the oscillation". The ladder buys nothing here -- its top band, 2,
is nowhere near 20 -- so it scores like the plain MLP. The measured bands find the
oscillation. The adaptive field, which needs no spectrum at all, does best, and
its learned slopes *report* the frequency it had to reach.

Honesty: the derivatives are exact (closed-form jets, no ``autograd`` in the
operator); the *solutions* are trained, not certified, and the numbers below are
training outcomes on a fixed seed, not bounds.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.diagnostics import geometric_bands, suggest_frequency_bands
from omnibias.pinn.torch.fields import (
    AdaptiveJetMLPVectorField,
    JetMLPVectorField,
    MscaleVectorField,
)

DTYPE = torch.float64
LOW, HIGH, AMP = 1.0, 20.0, 0.5


def exact(x: torch.Tensor) -> torch.Tensor:
    """``u*(x) = sin(2 pi x) + 0.5 sin(40 pi x)`` on ``(B, 1)`` coordinates."""
    xs = x[:, 0]
    return torch.sin(2 * math.pi * LOW * xs) + AMP * torch.sin(2 * math.pi * HIGH * xs)


def source(x: torch.Tensor) -> torch.Tensor:
    """``f = u*''``; its high-frequency term is ``(40 pi)^2 ~ 1.6e4`` times larger."""
    xs = x[:, 0]
    return -((2 * math.pi * LOW) ** 2) * torch.sin(2 * math.pi * LOW * xs) - AMP * (
        2 * math.pi * HIGH
    ) ** 2 * torch.sin(2 * math.pi * HIGH * xs)


def train(field: torch.nn.Module, *, steps: int = 3000, lr: float = 5e-3) -> None:
    """Adam on ``MSE((u'' - f) / |f|_max) + MSE(u - u*)`` over fixed grids.

    The residual is non-dimensionalised by ``|f|_max``; without that the loss
    scale is set by the highest frequency in the source and no single learning
    rate serves all four architectures fairly. The interior samples of ``u*``
    stand in for measured data -- the same samples the spectrum is read from.
    """
    x_int = torch.linspace(0.0, 1.0, 129, dtype=DTYPE).reshape(-1, 1)
    x_dat = torch.linspace(0.0, 1.0, 81, dtype=DTYPE).reshape(-1, 1)
    f_int, u_dat = source(x_int), exact(x_dat)
    scale = float(f_int.abs().max())
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pde = ((ops.laplacian(field(x_int), "u") - f_int) / scale).pow(2).mean()
        data = (ops.value(field(x_dat), "u") - u_dat).pow(2).mean()
        (pde + data).backward()
        opt.step()


def report(field: torch.nn.Module) -> tuple[float, float, float]:
    """Relative L2 error plus the learned amplitude in each of the two modes.

    The projections ``2 <u, sin(2 pi k x)>`` have targets ``1.0`` and ``AMP``.
    """
    xg = torch.linspace(0.0, 1.0, 401, dtype=DTYPE).reshape(-1, 1)
    with torch.no_grad():
        u = ops.value(field(xg), "u")
        target = exact(xg)
        rel_l2 = float((u - target).pow(2).mean().sqrt() / target.pow(2).mean().sqrt())
        low = float(2.0 * (u * torch.sin(2 * math.pi * LOW * xg[:, 0])).mean())
        high = float(2.0 * (u * torch.sin(2 * math.pi * HIGH * xg[:, 0])).mean())
    return rel_l2, low, high


def main() -> None:
    cs, comp = CoordinateSpec(("x",)), ComponentSpec(("u",))
    print("=== two-scale 1D Poisson: u* = sin(2 pi x) + 0.5 sin(40 pi x) ===")

    # --- the feedback loop -------------------------------------------------
    # Sample the data on a uniform grid and read the bands off its spectrum.
    # power_spectrum_per_d bins by integer wavenumber, so on L = 1 the scales
    # come back in the same units as LOW / HIGH.
    xs = np.linspace(0.0, 1.0, 128, endpoint=False)
    u_samples = np.sin(2 * np.pi * LOW * xs) + AMP * np.sin(2 * np.pi * HIGH * xs)
    measured = suggest_frequency_bands(u_samples[None, :], L=1.0, n_bands=2)
    ladder = geometric_bands(2)
    print(f"    true modes:     ({LOW:.0f}, {HIGH:.0f})")
    print(f"    guessed ladder: {tuple(round(b, 2) for b in ladder)}")
    print(f"    measured bands: {tuple(round(b, 2) for b in measured)}")
    assert abs(measured[0] - LOW) < 0.5 and abs(measured[-1] - HIGH) < 0.5, measured

    # --- four fields, same budget -----------------------------------------
    results: dict[str, tuple[float, float, float]] = {}

    torch.manual_seed(0)
    deep = JetMLPVectorField(coordinate_spec=cs, components=comp, hidden=64, depth=3)
    train(deep)
    results["JetMLPVectorField (plain)"] = report(deep)

    guessed_label = f"Mscale, guessed {tuple(round(b) for b in ladder)}"
    torch.manual_seed(0)
    guessed = MscaleVectorField(
        coordinate_spec=cs, components=comp, hidden=64, depth=3, scales=ladder
    )
    train(guessed)
    results[guessed_label] = report(guessed)

    measured_label = f"Mscale, measured {tuple(round(b) for b in measured)}"
    torch.manual_seed(0)
    informed = MscaleVectorField(
        coordinate_spec=cs, components=comp, hidden=64, depth=3, scales=measured
    )
    train(informed)
    results[measured_label] = report(informed)

    torch.manual_seed(0)
    adaptive = AdaptiveJetMLPVectorField(
        coordinate_spec=cs, components=comp, hidden=64, depth=3, slope_scale=10.0
    )
    train(adaptive)
    results["Adaptive slopes (no bands)"] = report(adaptive)

    print(f"\n    target amplitudes: low = 1.000, high = {AMP:.3f}")
    for name, (rel, low, high) in results.items():
        print(
            f"    {name:32s}: rel L2 = {rel:6.4f}   "
            f"low = {low:+.3f}   high = {high:+.4f}"
        )

    plain_high = results["JetMLPVectorField (plain)"][2]
    guessed_high = results[guessed_label][2]
    informed_high = results[measured_label][2]
    adaptive_high = results["Adaptive slopes (no bands)"][2]
    slopes = [float(s.detach().mean()) for s in adaptive.slopes()]
    print(
        f"\n    The ladder tops out at {ladder[-1]:.0f}, a factor {HIGH / ladder[-1]:.0f} "
        f"short of the mode that is actually there, so it recovers "
        f"{100 * guessed_high / AMP:.0f}% of the oscillation -- the plain MLP's "
        f"{100 * plain_high / AMP:.0f}%, unhelped."
        f"\n    Measured bands: {100 * informed_high / AMP:.0f}% of the oscillation at "
        "the same width and depth; the only change is the tuple."
        f"\n    Adaptive slopes: {100 * adaptive_high / AMP:.0f}% with no spectrum at "
        f"all; the learned slopes {[round(s, 2) for s in slopes]} start at 1.0, so the "
        "field is reporting the frequency it had to reach."
    )

    # The ladder is stuck where the plain MLP is; the measurement is what moves.
    assert abs(guessed_high) < 0.1 * AMP, (guessed_high, AMP)
    assert informed_high > 0.15 * AMP, (informed_high, AMP)
    assert adaptive_high > 0.5 * AMP, (adaptive_high, AMP)
    assert slopes[0] > 2.0, slopes

    # --- the tower is still exact -----------------------------------------
    # A trainable frequency is exactly the case that "obviously" needs autodiff:
    # the activation changes every optimiser step. It does not -- the slope is
    # the temperature of a tempered() spec, so sigma^(k) still comes from the
    # closed-form tower, and the Mscale mixture's jet is the sum of band jets.
    xg = torch.linspace(0.05, 0.95, 33, dtype=DTYPE).reshape(-1, 1)
    for name, field in (("adaptive", adaptive), ("mscale", informed)):
        xr = xg.clone().requires_grad_(True)
        u = field.forward_values(xr)[:, 0]
        du = torch.autograd.grad(u.sum(), xr, create_graph=True)[0][:, 0]
        d2u = torch.autograd.grad(du.sum(), xr)[0][:, 0].detach()
        with torch.no_grad():
            closed = ops.laplacian(field(xg), "u")
            err = float((closed - d2u).abs().max() / d2u.abs().max())
        print(f"    {name}: closed-form u'' vs autograd u'', max rel error {err:.2e}")
        assert err < 1e-12, (name, err)

    # And the jet cache still holds: value + gradient + laplacian is one jet.
    state = informed(torch.linspace(0.0, 1.0, 5, dtype=DTYPE).reshape(-1, 1))
    ops.value(state, "u")
    ops.gradient(state, "u")
    ops.laplacian(state, "u")
    cached = state.extra["_jet_mlp_jets"]
    print(f"    mscale: value + gradient + laplacian used {len(cached)} jet order.")
    assert sorted(cached) == [2]

    print("\nBands measured, not guessed; tower exact throughout; all checks passed.")


if __name__ == "__main__":
    main()

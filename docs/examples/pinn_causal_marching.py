# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Balancing the loss and respecting causality -- omnibias.pinn.

Run:

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_causal_marching.py

Two things sink more PINNs than any architecture problem, and neither is cured
by a bigger network:

* **The gradient pathology.** ``L = L_pde + lambda L_ic`` needs a ``lambda`` you
  cannot know before training. Below, one decade in that single constant is the
  difference between a useless model and a good one.
  :class:`GradNormWeighter` derives it from the measured gradients instead.
* **Causality violation.** Trained on a whole time interval at once, a PINN will
  fit late times before the early ones that determine them.
  :class:`TimeMarcher` solves a short window at a time and warm-starts the next
  from it.

The problem is Krishnapriyan et al.'s reaction benchmark,

    u_t = rho u (1 - u)  on  x in [0, 2 pi), t in [0, 1],   rho = 12,

with a Gaussian initial bump, whose exact solution is the logistic
``u = h e^{rho t} / (h e^{rho t} + 1 - h)``, ``h = u(x, 0)``. It carries no
spatial derivative at all: ``x`` enters only through the initial condition, so
every ``x`` is an independent trajectory and the *only* coupling is the shared
network. That is what makes it a clean test -- nothing spatial can be blamed.

Every run goes through one ``solve()`` and differs by one argument.
``TimeWindowSchedule(n_windows=1)`` is by construction the plain whole-interval
causal solve, so "marched" versus "not marched" is one knob at a fixed total
step budget, not two different programs.

Honesty: the derivatives are exact (closed-form jets, no ``autograd`` in the
operator); the *solutions* are trained, not certified, and the numbers below are
training outcomes on a fixed seed, not bounds. The marching result in particular
is regime-dependent -- see the note at the end of part 2.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.fields import JetMLPVectorField
from omnibias.pinn.torch.losses import (
    ConstantWeighter,
    GradNormWeighter,
    LossWeighter,
    SelfAdaptiveWeights,
    TimeMarcher,
    TimeWindowSchedule,
    causal_residual_loss,
    grad_stats,
)

DTYPE = torch.float64
RHO = 12.0
LENGTH = 2.0 * math.pi
N_BINS, PER_BIN, N_SLICE = 16, 64, 96
BUDGET, LR = 2000, 3e-3

CS = CoordinateSpec(("x", "t"), domain=((0.0, LENGTH), (0.0, 1.0)), time_axis="t")
COMP = ComponentSpec(("u",))
T_AXIS = CS.axis_index("t")


def u0(x: np.ndarray) -> np.ndarray:
    """Gaussian bump ``exp(-(x - pi)^2 / (2 (pi/4)^2))``."""
    return np.exp(-((x - math.pi) ** 2) / (2.0 * (math.pi / 4.0) ** 2))


def exact(points: np.ndarray) -> np.ndarray:
    """Logistic growth of the initial bump, on ``(N, 2)`` coordinates."""
    h = u0(points[:, 0])
    growth = np.exp(RHO * points[:, 1])
    return h * growth / (h * growth + 1.0 - h)


def residual(field: torch.nn.Module, coords: torch.Tensor) -> torch.Tensor:
    """``u_t - rho u (1 - u)``, pointwise, exactly as the equation is written.

    Deliberately *not* non-dimensionalised by ``rho``. Dividing through would
    shrink the per-bin losses by ``rho^2``, and since the causal weights are
    ``exp(-epsilon sum_j L_j)`` -- a function of the residual's absolute
    magnitude, not its shape -- that quietly drives every weight to 1 and
    switches the causal filter off. ``epsilon`` has to be scaled to the residual
    you actually have.
    """
    state = field(coords)
    u = ops.value(state, "u")
    return ops.derivative(state, "u", axis=T_AXIS) - RHO * u * (1.0 - u)


def relative_error(field: torch.nn.Module) -> float:
    """Relative L2 error against the exact solution on a space-time grid."""
    xs = np.linspace(0.0, LENGTH, 64, endpoint=False)
    ts = np.linspace(0.0, 1.0, 51)
    grid = np.stack(np.meshgrid(xs, ts, indexing="ij"), axis=-1).reshape(-1, 2)
    with torch.no_grad():
        u = ops.value(field(torch.tensor(grid, dtype=DTYPE)), "u").numpy()
    truth = exact(grid)
    return float(np.linalg.norm(u - truth) / np.linalg.norm(truth))


def solve(
    weighter: LossWeighter, *, n_windows: int = 1, seed: int = 0
) -> tuple[torch.nn.Module, TimeMarcher, list[float]]:
    """Train the field, marching across ``n_windows`` windows.

    The total optimiser budget is ``BUDGET`` steps however many windows it is
    split across, so more windows never buys more compute. Returns the field,
    the marcher (for its convergence record), and the max handoff error per
    window -- the error actually inherited across each seam.
    """
    torch.manual_seed(seed)
    field = JetMLPVectorField(
        coordinate_spec=CS, components=COMP, hidden=48, depth=3, jet_order=1
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=n_windows, n_time_bins=N_BINS, epsilon=0.5, tolerance=0.1
    )
    marcher = TimeMarcher(CS, schedule, per_bin=PER_BIN, n_slice=N_SLICE, seed=0)
    marcher.set_initial(u0(marcher.initial_points()[:, 0]))
    opt = torch.optim.Adam(field.parameters(), lr=LR)
    # GradNormWeighter is the only weighter here that consumes a measurement,
    # and only on its own cadence -- update() ignores `stats` otherwise, so
    # measuring every step would be pure waste.
    measured = isinstance(weighter, GradNormWeighter)
    steps = BUDGET // n_windows
    handoff_error: list[float] = []

    while not marcher.done:
        coords = torch.tensor(marcher.collocation().reshape(-1, CS.ndim), dtype=DTYPE)
        ic_x = torch.tensor(marcher.initial_points(), dtype=DTYPE)
        ic_u = torch.tensor(marcher.initial_values, dtype=DTYPE)
        epsilon = marcher.epsilon
        weights = None
        for _ in range(steps):
            opt.zero_grad()
            r = residual(field, coords)
            pde, weights = causal_residual_loss(
                r.reshape(N_BINS, PER_BIN), epsilon=epsilon, return_weights=True
            )
            terms = {
                "pde": pde,
                "ic": (ops.value(field(ic_x), "u") - ic_u).pow(2).mean(),
            }
            stats = (
                grad_stats(terms, field.parameters())
                if measured and weighter.step % weighter.every == 0
                else {}
            )
            weighter.update(stats)
            weighter.combine(terms).backward()
            opt.step()

        assert weights is not None
        marcher.observe(weights.detach().numpy())
        seam = marcher.handoff_points()
        with torch.no_grad():
            handoff = ops.value(field(torch.tensor(seam, dtype=DTYPE)), "u").numpy()
        handoff_error.append(float(np.abs(handoff - exact(seam)).max()))
        marcher.advance(handoff)
    return field, marcher, handoff_error


def main() -> None:
    print(f"=== reaction u_t = {RHO:.0f} u (1 - u), logistic front ===")
    results: dict[str, float] = {}

    # --- 1. one constant, two outcomes -------------------------------------
    print("\n  [1] whole interval; the only change is the weight on L_ic")
    for lam in (1.0, 10.0):
        field, _, _ = solve(ConstantWeighter({"pde": 1.0, "ic": lam}))
        results[f"whole, fixed ic={lam:g}"] = relative_error(field)
        print(f"    fixed ic={lam:<5g}: rel L2 = {results[f'whole, fixed ic={lam:g}']:.4f}")

    # The weighter measures the imbalance instead of being told it. Its first
    # target for 'ic' is max|dL_pde/dtheta| / mean|dL_ic/dtheta|; the EMA then
    # tracks it, and `ceiling` bounds the drift once L_ic is nearly satisfied
    # and its mean gradient heads for zero.
    gradnorm = GradNormWeighter(
        ["pde", "ic"], reference="pde", alpha=0.9, every=20, ceiling=1e4
    )
    field_gn, _, _ = solve(gradnorm)
    results["whole, GradNormWeighter"] = relative_error(field_gn)
    print(f"    GradNormWeighter : rel L2 = {results['whole, GradNormWeighter']:.4f}")
    print(
        f"    ic = 1 is not a slightly worse choice, it is a failed run "
        f"({results['whole, fixed ic=1']:.2f} vs "
        f"{results['whole, fixed ic=10']:.2f}), and nothing about the problem\n"
        f"    tells you which side of that cliff you are on. The weighter lands "
        f"at {results['whole, GradNormWeighter']:.2f}\n"
        f"    with no sweep -- not the best hand-tuned value, but on the right "
        f"side of the cliff\n    by measurement rather than by luck."
    )
    assert results["whole, fixed ic=1"] > 0.5, results
    assert results["whole, fixed ic=10"] < 0.25, results
    assert results["whole, GradNormWeighter"] < 0.25, results

    # --- 2. causality: same budget, split into windows ----------------------
    print("\n  [2] same solver, same budget, marched instead of solved at once")
    marched, marcher, handoff = solve(
        ConstantWeighter({"pde": 1.0, "ic": 10.0}), n_windows=5
    )
    results["marched x5, fixed ic=10"] = relative_error(marched)
    print(f"    whole interval  (1 window,  {BUDGET} steps): "
          f"rel L2 = {results['whole, fixed ic=10']:.4f}")
    print(f"    marched         (5 windows, {BUDGET // 5} steps each): "
          f"rel L2 = {results['marched x5, fixed ic=10']:.4f}")
    print(f"    max handoff error per seam: {[round(e, 3) for e in handoff]}")
    print(f"    advance criterion met per window: {marcher.converged}")
    print(
        "    Marching is the better of the two here, at identical budget and "
        "identical weights.\n    It is not a universal win: the handoff error "
        "above is real error inherited by the\n    next window, so when a "
        "window ends badly marching compounds it where a\n    whole-interval "
        "solve would not. `converged` and the seam errors are the "
        "diagnostics\n    that tell you which regime you are in -- report them, "
        "do not assume."
    )
    assert results["marched x5, fixed ic=10"] < results["whole, fixed ic=10"], results

    # --- 3. self-adaptive pointwise weights ---------------------------------
    # A different question from "how much does this term matter": *which points*
    # matter. Freezing the trained field isolates the mechanism -- the weights
    # are trained by gradient ascent, so they climb fastest where the residual
    # is largest, and rank the collocation set by difficulty.
    print("\n  [3] self-adaptive pointwise weights: ranking points by residual")
    coords = torch.tensor(
        np.stack(
            np.meshgrid(
                np.linspace(0.0, LENGTH, PER_BIN, endpoint=False),
                np.linspace(0.0, 1.0, N_BINS),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2),
        dtype=DTYPE,
    )
    with torch.no_grad():
        frozen = residual(marched, coords)
    attention = SelfAdaptiveWeights(frozen.shape[0], dtype=DTYPE)
    # Plain SGD, because the ascent rate is *proportional to* r^2 -- which is
    # the whole mechanism. Adam would preserve the ordering but normalise the
    # rates, moving every weight at roughly the same speed. The learning rate is
    # scaled by the largest residual so the demonstration is scale-free.
    rate = 0.12 * frozen.shape[0] / float(frozen.pow(2).max())
    opt = torch.optim.SGD(attention.parameters(), lr=rate)
    for _ in range(200):
        opt.zero_grad()
        attention(frozen).backward()
        opt.step()

    att = attention.attention()
    order = torch.argsort(frozen.abs())
    n_tenth = max(1, att.shape[0] // 10)
    easy = float(att[order[:n_tenth]].mean())
    hard = float(att[order[-n_tenth:]].mean())
    rank_corr = float(
        np.corrcoef(
            torch.argsort(torch.argsort(att)).numpy(),
            torch.argsort(torch.argsort(frozen.abs())).numpy(),
        )[0, 1]
    )
    print(f"    attention on the easiest 10% of points: {easy:.3f}")
    print(f"    attention on the hardest 10% of points: {hard:.3f}")
    print(f"    rank correlation with |residual|: {rank_corr:+.4f}")
    print(
        "    Every weight started at the neutral 0.5. The ones over points the "
        "field already\n    fits are still there; the ones over points it "
        "cannot fit climbed on their own, with\n    no supervision beyond the "
        "residual they are trying to defeat. Feeding that back\n    into "
        "training is what makes the loss concentrate where the solution is stiff."
    )
    assert easy < 0.6, easy
    assert hard > 0.8, hard
    assert rank_corr > 0.99, rank_corr

    print("\n  summary")
    for name, err in results.items():
        print(f"    {name:26s}: rel L2 = {err:.4f}")

    # --- the tower is still exact -------------------------------------------
    # All of the above is weighting and scheduling; none of it touches the
    # operator, which is still the closed-form jet rather than autodiff.
    probe = torch.tensor(
        np.stack([np.linspace(0.1, LENGTH - 0.1, 17), np.linspace(0.05, 0.95, 17)], -1),
        dtype=DTYPE,
    )
    xr = probe.clone().requires_grad_(True)
    u = marched.forward_values(xr)[:, 0]
    du = torch.autograd.grad(u.sum(), xr)[0][:, T_AXIS].detach()
    with torch.no_grad():
        closed = ops.derivative(marched(probe), "u", axis=T_AXIS)
    err = float((closed - du).abs().max() / du.abs().max())
    print(f"\n    closed-form u_t vs autograd u_t, max rel error {err:.2e}")
    assert err < 1e-12, err

    print("\nWeights measured, not guessed; causality marched; tower exact throughout.")


if __name__ == "__main__":
    main()

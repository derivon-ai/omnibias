# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Does residual-adaptive refinement actually buy accuracy? Budget-matched gate.

The claim RAR makes is about the *worst* point, not the average one, so the metric
is the held-out dense-grid residual **max-norm**. The comparison is budget-matched
by construction: the RAR arm starts at ``N_START`` random interior points and grows
to exactly ``N_FINAL``; the uniform arm draws ``N_FINAL`` up front. Same field
init, same warmup, same optimiser, same iteration count -- the only difference is
*where* the points are.

What the measurements support (a GPU-cluster sweep at this reduced budget across 16
seeds, cross-checked at the larger budget in ``docs/benchmarks.md`` across 8):

* **Sharp front** (stiff bistable pair, interface width ~sqrt(D/k)): a per-seed
  win, 16/16 seeds for both strategies, median 2.1x (proportional) and 3.0x
  (greedy), worst seed 1.5x. This is gated per seed below.
* **Low-viscosity Burgers**: a *median* win only -- 1.18x greedy (14/16 seeds),
  1.07x proportional (11/16). Gated on the median, which is the honest form of
  the claim; asserting a per-seed win here would be a flaky test asserting
  something false.

The budget is deliberately small -- the effect is larger at the benchmark budget,
not smaller -- but twenty solves is still minutes, so the whole module is marked
``slow`` and runs under ``-m slow`` rather than on every invocation. The cheap
mechanical guarantees (schedule, cap, which points get chosen) are the fast suite
in ``test_refinement.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from omnibias.pinn.solver.torch.assemble import (  # noqa: E402
    interior_residual,
    to_tensor,
)

pytestmark = pytest.mark.slow

SEEDS = range(5)
N_START, N_FINAL, ROUNDS = 100, 220, 4
ITERS, ADAM_ITERS, HIDDEN = 40, 150, 24
HOLDOUT = 80


def _burgers(viscosity: float) -> pde.System:
    """u_t + u u_x = nu u_xx, u(x,0) = -sin(pi x): steepens toward a shock at x=0."""

    def initial(c):
        xp = pde.array_namespace(c)
        return -xp.sin(math.pi * c[:, 0])

    dom = pde.Domain(("x", "t"), ((-1.0, 1.0), (0.0, 1.0)), time_axis="t")
    return pde.burgers(dom, viscosity=viscosity, initial=initial)


def _sharp_front(diffusivity: float, rate: float) -> pde.System:
    """A stiff bistable pair; ``u`` carries an interface of width ~sqrt(D/k)."""

    def reaction(u: Any, v: Any) -> tuple[Any, Any]:
        return rate * u * (1.0 - u * u), rate * (u - v)

    def initial_u(c):
        xp = pde.array_namespace(c)
        return xp.tanh(c[:, 0] / 0.05)

    dom = pde.Domain(("x", "t"), ((-1.0, 1.0), (0.0, 1.0)), time_axis="t")
    return pde.reaction_diffusion(
        dom,
        diffusivities=(diffusivity, diffusivity),
        reaction=reaction,
        initial=(initial_u, 0.0),
    )


def _holdout_max_norm(sol: Any) -> float:
    """Max |interior residual| on a dense grid that neither arm trained on."""
    field, system = sol.field, sol.system
    axes = [
        np.linspace(lo + 1e-3, hi - 1e-3, HOLDOUT) for (lo, hi) in system.domain.bounds
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=-1)
    with torch.no_grad():
        return float(interior_residual(field, system, to_tensor(pts, field)).abs().max())


def _solve(system: pde.System, *, seed: int, strategy: str | None) -> Any:
    """One arm. ``strategy=None`` is the uniform baseline at the final budget."""
    n_interior = N_FINAL if strategy is None else N_START
    spec = pde.CollocationSpec(
        n_interior=n_interior, n_boundary=12, method="random", seed=seed
    )
    refinement = (
        None
        if strategy is None
        else pde.RefinementSpec(
            every=max(1, ITERS // (ROUNDS + 1)),
            n_candidates=500,
            n_add=(N_FINAL - N_START) // ROUNDS,
            strategy=strategy,
            power=2.0,
            max_points=N_FINAL,
            seed=1000 + seed,
        )
    )
    return pt.solve_optimize(
        system,
        hidden=HIDDEN,
        seed=seed,
        collocation=spec,
        iters=ITERS,
        adam_iters=ADAM_ITERS,
        condition_weight=50.0,
        refinement=refinement,
    )


def _ratios(system: pde.System, strategy: str) -> list[float]:
    """Per-seed ``uniform / rar`` max-norm ratios (>1 means RAR is better)."""
    out: list[float] = []
    for seed in SEEDS:
        uniform = _solve(system, seed=seed, strategy=None)
        refined = _solve(system, seed=seed, strategy=strategy)
        assert refined.diagnostics["n_interior_final"] == N_FINAL
        assert int(uniform.diagnostics.get("n_interior_final", N_FINAL)) == N_FINAL
        out.append(_holdout_max_norm(uniform) / _holdout_max_norm(refined))
    return out


@pytest.mark.parametrize("strategy", ["greedy", "proportional"])
def test_sharp_front_is_a_per_seed_win(strategy: str) -> None:
    """Thresholds sit ~1.5x below the measured worst case, not at it.

    Over 16 seeds the weakest arm (proportional) has worst 1.50x and median 2.10x;
    on these five seeds it is worst 2.07x and median 2.29x. Gating at the measured
    edge would make this a coin flip on an unseen seed.
    """
    ratios = _ratios(_sharp_front(0.002, 20.0), strategy)
    assert min(ratios) > 1.3, f"{strategy} lost a seed on the sharp front: {ratios}"
    assert float(np.median(ratios)) > 1.8, f"{strategy} median too small: {ratios}"


def test_low_viscosity_burgers_is_a_median_win() -> None:
    """Honest form of the Burgers claim: the median improves, not every seed.

    Moving points onto the shock costs resolution elsewhere, so an individual seed
    can come out worse -- 2 of 16 do here. The median win survives at the larger
    benchmark budget (1.13x greedy, 1.23x proportional), where *proportional* is
    the arm that reaches 8/8 seeds while greedy still loses 3; ``greedy`` is what
    this test uses because it is the stronger arm at this reduced budget, which is
    the opposite of the ordering at the benchmark budget.
    """
    ratios = _ratios(_burgers(0.003), "greedy")
    assert float(np.median(ratios)) > 1.0, f"no median improvement: {ratios}"


def test_the_uniform_baseline_really_is_budget_matched() -> None:
    """Guard the comparison itself: both arms must end on the same point count."""
    system = _sharp_front(0.002, 20.0)
    uniform = _solve(system, seed=0, strategy=None)
    refined = _solve(system, seed=0, strategy="greedy")
    assert "n_interior_final" not in uniform.diagnostics
    assert refined.diagnostics["n_interior_uniform"] == N_START
    assert refined.diagnostics["n_interior_final"] == N_FINAL
    assert refined.diagnostics["n_refinement_rounds"] == ROUNDS

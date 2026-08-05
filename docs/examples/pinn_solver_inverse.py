# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Inverse problems and residual-adaptive collocation on the omnibias-pinn solver.

Run:

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_solver_inverse.py

Two capabilities that share the one optimisation driver:

1. **Inverse problems.** Wrap a PDE coefficient in ``Unknown`` and ``solve_inverse``
   recovers it from measurements of the solution, jointly with the field. Here the
   heat equation's diffusivity is recovered from a 3x-wrong initial guess against
   the analytic solution ``exp(-D pi^2 t) sin(pi x)``.
2. **Residual-adaptive refinement (RAR).** ``RefinementSpec`` periodically scores
   fresh candidate points by ``|residual|`` and keeps the worst ones, so collocation
   concentrates where the solution is hard. Shown on a sharp reaction front at an
   *equal final point budget* against uniform sampling, which is the only comparison
   that means anything.

Why the coefficient is reachable at all: the residual reads it through a resolver
bound to a live tensor, so a single frozen ``System`` serves both forward and
inverse modes, and each coefficient is parameterised by an unconstrained variable
through a transform -- positivity holds by construction, with no clipping step.

Honesty labels: the differential operators are closed form; the parameter and
coefficient curvature is autodiff-exact; the collocation integral is numerical.
Identifiability is the caller's problem and fails silently by nature -- a
coefficient the data cannot see does not error, it just stops moving.
"""

from __future__ import annotations

import math

import numpy as np
import omnibias.pinn.solver as pde
import omnibias.pinn.solver.torch as pt
import torch
from omnibias.pinn.solver._core.sampling import candidate_points, select_refinement_points
from omnibias.pinn.solver.torch.assemble import interior_residual, to_tensor

torch.set_default_dtype(torch.float64)

TRUE_D = 0.35
N_OBS = 48
BUDGET = {
    "hidden": 16,
    "collocation": pde.CollocationSpec(n_interior=10, n_boundary=6),
    "iters": 40,
    "adam_iters": 150,
    "condition_weight": 20.0,
    "data_weight": 20.0,
}


def heat_system(diffusivity: float | pde.Unknown) -> pde.System:
    def initial(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 0])

    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.2)), time_axis="t")
    return pde.heat(dom, diffusivity=diffusivity, initial=initial, boundary=0.0)


def heat_exact(pts: np.ndarray) -> np.ndarray:
    return np.exp(-TRUE_D * math.pi**2 * pts[:, 1]) * np.sin(math.pi * pts[:, 0])


def sharp_front(diffusivity: float = 0.002, rate: float = 20.0) -> pde.System:
    """A stiff bistable pair whose ``u`` carries an interface of width ~sqrt(D/k)."""

    def reaction(u, v):
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


def holdout_max_norm(solution, n: int = 60) -> float:
    """Max |interior residual| on a dense grid neither arm trained on."""
    axes = [
        np.linspace(lo + 1e-3, hi - 1e-3, n)
        for (lo, hi) in solution.system.domain.bounds
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=-1)
    with torch.no_grad():
        rows = interior_residual(solution.field, solution.system, to_tensor(pts, solution.field))
        return float(rows.abs().max())


def check(name: str, ok: bool, detail: str) -> None:
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        raise SystemExit(f"validation failed: {name}")


def observation_points(system: pde.System, seed: int = 4242) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(lo, hi, N_OBS) for (lo, hi) in system.domain.bounds], axis=-1
    )


def demo_inverse() -> None:
    print("1. recovering an unknown diffusivity from 48 point measurements")
    coords = observation_points(heat_system(TRUE_D))
    values = heat_exact(coords)

    guess = pde.Unknown("D", initial=3.0 * TRUE_D, transform="positive")
    system = heat_system(guess)
    print(f"     truth={TRUE_D}  initial guess={guess.initial} (3x wrong, held positive)")

    # A forward driver refuses this system rather than solving the wrong PDE.
    try:
        pt.solve_optimize(system, hidden=8, iters=1, adam_iters=1)
    except ValueError as exc:
        print(f"     forward driver correctly refuses it: {str(exc).splitlines()[0]}")

    sol = pt.solve_inverse(
        system, [pde.Observations("u", coords, values)], seed=0, **BUDGET
    )
    error = abs(sol.recovered["D"] - TRUE_D) / TRUE_D
    print(f"     recovered D={sol.recovered['D']:.6f}  rel error={error:.3%}")
    print(f"     data misfit (RMS)={sol.data_misfit:.3e}  residual={sol.residual_norm:.3e}")
    check("diffusivity recovered to better than 1%", error < 0.01, f"{error:.3%}")
    check("the positive transform held", sol.recovered["D"] > 0.0, f"{sol.recovered['D']:.6f}")

    # The optimiser matters more here than it does for a forward solve: one scalar
    # coefficient and a few hundred weights have curvature on different scales.
    lbfgs = pt.solve_inverse(
        heat_system(pde.Unknown("D", initial=3.0 * TRUE_D, transform="positive")),
        [pde.Observations("u", coords, values)],
        seed=0,
        optimizer="lbfgs",
        **BUDGET,
    )
    lbfgs_error = abs(lbfgs.recovered["D"] - TRUE_D) / TRUE_D
    print(f"     same budget with L-BFGS: rel error={lbfgs_error:.1%}")
    check(
        "exact-curvature recovery beats L-BFGS by an order of magnitude",
        error < 0.1 * lbfgs_error,
        f"{error:.3%} vs {lbfgs_error:.1%}",
    )

    print("\n2. noisy data degrades the recovery, it does not break it")
    rng = np.random.default_rng(9000)
    noisy = values + rng.normal(0.0, 0.05 * float(np.abs(values).max()), values.shape)
    noisy_sol = pt.solve_inverse(
        heat_system(pde.Unknown("D", initial=3.0 * TRUE_D, transform="positive")),
        [pde.Observations("u", coords, noisy)],
        seed=0,
        **BUDGET,
    )
    noisy_error = abs(noisy_sol.recovered["D"] - TRUE_D) / TRUE_D
    print(f"     5% observation noise -> D={noisy_sol.recovered['D']:.6f}, rel error={noisy_error:.2%}")
    check("still within 10% at 5% noise", noisy_error < 0.10, f"{noisy_error:.2%}")


def demo_refinement() -> None:
    print("\n3. residual-adaptive refinement on a sharp front (equal point budget)")
    system = sharp_front()
    n_start, n_final, rounds = 60, 120, 3
    common = {"hidden": 20, "seed": 0, "iters": 24, "adam_iters": 100,
              "condition_weight": 50.0}

    uniform = pt.solve_optimize(
        system,
        collocation=pde.CollocationSpec(n_interior=n_final, n_boundary=10, method="random", seed=0),
        **common,
    )
    refined = pt.solve_optimize(
        system,
        collocation=pde.CollocationSpec(n_interior=n_start, n_boundary=10, method="random", seed=0),
        refinement=pde.RefinementSpec(
            every=max(1, common["iters"] // (rounds + 1)),
            n_candidates=400,
            n_add=(n_final - n_start) // rounds,
            max_points=n_final,
            seed=1000,
        ),
        **common,
    )
    print(f"     uniform : {n_final} interior points from the start")
    print(
        f"     RAR     : {refined.diagnostics['n_interior_uniform']} -> "
        f"{refined.diagnostics['n_interior_final']} over "
        f"{refined.diagnostics['n_refinement_rounds']} rounds"
    )
    check(
        "the two arms end on the same point budget",
        refined.diagnostics["n_interior_final"] == n_final,
        f"{refined.diagnostics['n_interior_final']} == {n_final}",
    )
    u_norm, r_norm = holdout_max_norm(uniform), holdout_max_norm(refined)
    print(f"     held-out residual max-norm: uniform={u_norm:.4e}  RAR={r_norm:.4e}")
    print(f"     improvement: {u_norm / r_norm:.2f}x")

    print("\n4. the added points really do track the residual")
    field = pt.build_field(system, hidden=20, seed=0)
    spec = pde.CollocationSpec(n_interior=60, n_boundary=10)
    ref = pde.RefinementSpec(n_candidates=400, n_add=40, strategy="greedy")
    candidates = candidate_points(system.domain, spec, ref, round_index=1)
    with torch.no_grad():
        rows = interior_residual(field, system, to_tensor(candidates, field))
        scores = rows.reshape(-1, candidates.shape[0]).abs().amax(dim=0).numpy()
    kept = select_refinement_points(candidates, scores, ref, n_existing=0)
    with torch.no_grad():
        kept_rows = interior_residual(field, system, to_tensor(kept, field))
        kept_scores = kept_rows.reshape(-1, kept.shape[0]).abs().amax(dim=0).numpy()
    print(f"     all candidates : mean |r| = {scores.mean():.4e}")
    print(f"     kept points    : mean |r| = {kept_scores.mean():.4e}")
    check(
        "every kept point beats the candidate median",
        float(kept_scores.min()) >= float(np.median(scores)),
        f"min kept {kept_scores.min():.3e} >= median {np.median(scores):.3e}",
    )


def main() -> None:
    print("omnibias-pinn :: inverse problems + adaptive collocation\n")
    demo_inverse()
    demo_refinement()
    print("\nall checks passed")


if __name__ == "__main__":
    main()

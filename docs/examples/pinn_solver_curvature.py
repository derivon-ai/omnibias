# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Second-order PINN training on the omnibias-pinn solver.

Run:

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_solver_curvature.py

Solves the 2-D Poisson problem ``Delta u = -2 pi^2 sin(pi x) sin(pi y)`` with the
mesh-free solver and compares the first-order / quasi-Newton baselines against the
exact-curvature optimisers of ``omnibias.torch.optim``.

Why this is possible: every differential operator in the residual is a **closed-form**
``sigma``-tower reduction, so the residual stays smooth and cheap to re-evaluate. The
curvature optimisers then get exact Hessian / Gauss-Newton products from matrix-free
double-backward passes -- no nested autodiff through a finite-difference operator, and
no learning rate to tune.

Honesty labels: the differential operators are closed form; the parameter curvature is
autodiff-exact; the collocation integral is numerical.
"""

from __future__ import annotations

import math
import time

import numpy as np
import omnibias.pinn.solver as pde
import omnibias.pinn.solver.torch as pt
import torch
from omnibias.pinn.solver.torch.assemble import (
    condition_residual,
    default_interior,
    interior_residual,
)
from omnibias.pinn.solver.torch.steady import _weighted_rows

torch.set_default_dtype(torch.float64)

SPEC = pde.CollocationSpec(n_interior=10, n_boundary=10)
CONDITION_WEIGHT = 20.0
HIDDEN = 20
WARMUP = 80
BUDGET = {
    "hidden": HIDDEN,
    "collocation": SPEC,
    "adam_iters": WARMUP,
    "condition_weight": CONDITION_WEIGHT,
}
ITERS = 10


def poisson_system() -> pde.System:
    def source(c):
        xp = pde.array_namespace(c)
        return -2.0 * math.pi**2 * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    return pde.poisson(dom, source=source, boundary=0.0)


def held_out_grid() -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.02, 0.98, 30)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    exact = np.sin(math.pi * pts[:, 0]) * np.sin(math.pi * pts[:, 1])
    return pts, exact


def objective(solution) -> float:
    """The fused training objective, read back off a returned solution."""
    field, system = solution.field, solution.system
    coords = default_interior(field, system, SPEC)
    with torch.no_grad():
        rows = _weighted_rows(field, system, coords, SPEC, CONDITION_WEIGHT)
        return float(torch.sum(rows**2))


def rel_l2(solution, pts: np.ndarray, exact: np.ndarray) -> float:
    u = solution.evaluate(pts, "u").detach().numpy()
    return float(np.linalg.norm(u - exact) / np.linalg.norm(exact))


def check(name: str, ok: bool, detail: str) -> None:
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        raise SystemExit(f"validation failed: {name}")


def main() -> None:
    print("omnibias-pinn :: second-order solver training (Poisson 2-D)\n")
    system = poisson_system()
    pts, exact = held_out_grid()
    seed = 0

    # 1. The objective the residual-vector optimisers see is the scalar loss.
    field = pt.build_field(system, hidden=16, seed=seed)
    coords = default_interior(field, system, SPEC)
    with torch.no_grad():
        rows = _weighted_rows(field, system, coords, SPEC, CONDITION_WEIGHT)
        fused = torch.mean(interior_residual(field, system, coords) ** 2)
        fused = fused + CONDITION_WEIGHT * torch.mean(
            condition_residual(field, system, SPEC) ** 2
        )
    print("1. residual-vector <-> scalar-loss identity")
    check(
        "sum(weighted rows^2) == fused loss",
        abs(float(torch.sum(rows**2)) - float(fused)) <= 1e-12 * abs(float(fused)),
        f"{float(torch.sum(rows ** 2)):.12e}",
    )

    # 2. Warmup-only baseline: every optimiser must descend from here.
    warm = pt.solve_optimize(system, optimizer="adam", iters=0, seed=seed, **BUDGET)
    warm_obj = objective(warm)
    print(f"\n2. Adam warmup only ({BUDGET['adam_iters']} iters)")
    print(f"     objective={warm_obj:.6e}  relL2={rel_l2(warm, pts, exact):.4f}")

    # 3. Equal-epoch comparison after the same warmup.
    cases: list[tuple[str, dict[str, object] | None]] = [
        ("adam", None),
        ("lbfgs", None),
        # A small Krylov subspace is plenty for this problem size and keeps the
        # example inside a few seconds per optimiser.
        ("cubic_newton", {"krylov_dim": 8}),
        ("cubic_gauss_newton", {"krylov_dim": 8}),
        ("gauss_newton", {"solver": "qr", "damping_strategy": "nielsen"}),
        ("natural_gradient", None),
    ]
    print(f"\n3. equal-epoch ({ITERS} iters each, seed={seed})")
    print(f"     {'optimizer':24s} {'objective':>12s} {'relL2':>9s} {'seconds':>8s}")
    results: dict[str, tuple[float, float, float]] = {}
    for name, okw in cases:
        t0 = time.perf_counter()
        sol = pt.solve_optimize(
            system, optimizer=name, optimizer_kwargs=okw, iters=ITERS,
            seed=seed, **BUDGET,
        )
        secs = time.perf_counter() - t0
        obj, rel = objective(sol), rel_l2(sol, pts, exact)
        results[name] = (obj, rel, secs)
        print(f"     {name:24s} {obj:12.4e} {rel:9.4f} {secs:8.2f}")

    print("\n4. guarantees")
    for name, (obj, _, _) in results.items():
        check(
            f"{name} descends from the warmup objective",
            obj <= warm_obj * (1.0 + 1e-9),
            f"{warm_obj:.4e} -> {obj:.4e}",
        )
    best_second = min(
        results[n][1] for n in ("cubic_gauss_newton", "gauss_newton", "natural_gradient")
    )
    check(
        "best Gauss-Newton-metric method beats L-BFGS on held-out accuracy",
        best_second < results["lbfgs"][1],
        f"relL2 {best_second:.4f} < {results['lbfgs'][1]:.4f}",
    )

    # 5. Gradient-norm balancing replaces the hand-tuned condition weight.
    balanced = pt.solve_optimize(
        system, optimizer="cubic_newton", iters=ITERS, seed=seed,
        loss_balancing="grad_norm", balance_every=5,
        hidden=HIDDEN, collocation=SPEC, adam_iters=WARMUP, condition_weight=1.0,
    )
    weights = balanced.diagnostics["balance_weights"]
    print("\n5. grad-norm loss balancing (condition_weight=1.0)")
    print(f"     learned term weights: {tuple(round(w, 3) for w in weights)}")
    check(
        "the condition term is reweighted away from 1",
        abs(weights[1] - 1.0) > 1e-3,
        f"lambda_condition={weights[1]:.3f}",
    )
    print(f"     relL2={rel_l2(balanced, pts, exact):.4f}")

    print("\nall checks passed")


if __name__ == "__main__":
    main()

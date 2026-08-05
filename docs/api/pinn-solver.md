# omnibias.pinn.solver — mesh-free PDE solver

Mesh-free solver for **coupled systems of PDEs** on the omnibias closed-form
derivative tower, shipped inside `omnibias-pinn` as the `omnibias.pinn.solver`
submodule (folded from the former standalone `omnibias-pde`). A `System` bundles a
`Domain`, one or more `Field`s, a tuple of coupled residual closures over
`state.ops.*`, and boundary / initial conditions; solver drivers then fit a
one-layer omnibias field (or march a spectral grid) so the residual goes to zero.

The real edge is the closed-form arbitrary-order derivative tower: one forward
pass yields *every* mixed partial exactly, so high-order / mixed operators
(biharmonic, 4th-order) are cheap and mesh-free with **no nested autograd**. It
does not aim to beat mature solvers (FEM / PETSc / Dedalus) on raw throughput;
the edges are exact high-order operators in one pass, mesh-free / high-dimensional
collocation, fast convergence on smooth solutions, and an optional certified
solve.

## Honest method labels

Every path is labelled: closed-form (the `sigma`-tower operators), autodiff
(only the *parameter* Jacobian in the residual-minimisation driver), numerical
(RK4 / implicit steps, least-squares solves), spectral (FFT spatial derivatives),
and high-order (the jet-Taylor time integrator, local truncation `O(dt^{N+1})`).
The solver never asserts a `unproven_claim` / continuum-regularity claim.

## Core schemas

::: omnibias.pinn.solver
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - System
        - Field
        - Domain
        - BoundaryCondition
        - InitialCondition
        - CollocationSpec
        - Classification
        - PDEType
        - Linearity
        - ProblemKind
        - Arity
        - honesty_labels

## Canonical problems

Six neutral builders span the taxonomy cross-product (type x linearity x kind x
arity): `poisson`, `heat`, `wave`, `burgers`, `reaction_diffusion` (coupled), and
`advection_diffusion` (coupled).

::: omnibias.pinn.solver
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - poisson
        - heat
        - wave
        - burgers
        - reaction_diffusion
        - advection_diffusion

## Solver drivers (torch)

::: omnibias.pinn.solver.torch
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - solve_steady
        - solve_least_squares
        - solve_optimize
        - solve_evolution
        - method_of_lines
        - SpectralGrid1D

### Second-order training

`solve_optimize(optimizer=...)` accepts, beyond the `"lbfgs"` default and `"adam"`,
the exact-curvature optimisers of
[`omnibias.torch.optim`](torch.md): `"cubic_newton"`,
`"cubic_gauss_newton"`, `"gauss_newton"`, `"trust_region_newton_cg"`,
`"jet_subspace_tensor"` and `"natural_gradient"` (the full set is
`omnibias.pinn.solver.torch.OPTIMIZERS`). Because every differential operator in the
residual is a closed-form `sigma`-tower reduction, the Hessian / Gauss-Newton
products are matrix-free double-backward passes over a smooth residual -- there is
no nested autodiff through a difference operator, and the ARC / trust-region
methods need no learning rate.

For a least-squares PINN residual the Gauss-Newton metric is the right one:
`"gauss_newton"` (Levenberg-Marquardt, bridged through
`omnibias.torch.optim.functional_residual_fn`; prefer
`optimizer_kwargs={"solver": "qr"}` or `"cgls"` so a stiff operator's conditioning
is not squared) and `"cubic_gauss_newton"` are the recommended choices.
`"natural_gradient"` defaults to the closed-form Gauss-Newton Fisher of this
residual. `loss_balancing="grad_norm"` drives a
`omnibias.torch.optim.GradNormBalancer` over `[interior_mse, condition_mse]` so the
weighted per-term gradient norms match, which is the principled replacement for a
hand-tuned `condition_weight`.

Measured results, honest boundaries (default unchanged, torch-only, `KFAC`
deliberately excluded) and the equal-wall-clock comparison are on the
[benchmarks page](../benchmarks.md#curvature-optimizers-pinn-operator-scaling-llm);
[`docs/examples/pinn_solver_curvature.py`](../examples/pinn_solver_curvature.py) is
the runnable version.

### Residual-adaptive collocation

Uniform collocation spends the same number of points on a flat region as on a
shock. Passing a `RefinementSpec` to `solve_optimize(refinement=...)` turns on
residual-adaptive refinement (RAR): every `every` iterations the driver draws
`n_candidates` fresh interior points, scores them by `abs(residual)` under
`no_grad`, and concatenates `n_add` of them onto the interior set until
`max_points` is reached.

Two selection rules. `strategy="proportional"` (the default) samples without
replacement with probability proportional to `score ** power`, and
`strategy="greedy"` keeps the top `n_add` outright. Greedy is the sharper
instrument and proportional the more robust one; which wins depends on the budget,
so both are measured on the [benchmarks page](../benchmarks.md#residual-adaptive-collocation-rar).

Three boundaries worth stating:

- **`solve_optimize` only.** `solve_least_squares` caches a `CollocationPlan`, so a
  growing point set would invalidate it; RAR is not wired there.
- **No refinement during the Adam warmup.** An early-stage residual is dominated by
  initialisation, not by the solution's structure, so refining against it would
  spend the budget on noise.
- **Equal-budget comparisons only.** RAR that ends with more points than the
  baseline is not a fair win. The gate in
  `packages/omnibias-pinn/tests/solver/test_refinement_accuracy.py` matches the
  final point count by construction.

### Inverse problems

`solve_inverse` recovers unknown PDE coefficients from measurements of the
solution, jointly with the field. Wrap a coefficient in `Unknown` and every
canonical builder accepts it wherever it accepts a float:

```python
import numpy as np
import omnibias.pinn.solver as pde
import omnibias.pinn.solver.torch as pt

domain = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.2)), time_axis="t")

def u0(x):
    return np.sin(np.pi * x[..., 0])

# a few interior measurements of the true D = 0.1 solution
coords = np.stack([np.linspace(0.1, 0.9, 8), np.full(8, 0.1)], axis=-1)
values = np.sin(np.pi * coords[:, 0]) * np.exp(-0.1 * np.pi**2 * 0.1)

unknown = pde.Unknown("D", initial=1.0, transform="positive")
system = pde.heat(domain, diffusivity=unknown, initial=u0, boundary=0.0)
solution = pt.solve_inverse(
    system, [pde.Observations("u", coords, values)],
    hidden=16, iters=5, adam_iters=20,     # tiny budget, so this page stays fast
)
solution.recovered["D"]
```

The objective gains a third term,
`mean(r_pde^2) + condition_weight * mean(r_bc^2) + data_weight * mean(r_data^2)`.
That data term is what makes the coefficient identifiable at all: the PDE residual
alone is satisfied by many `(field, coefficient)` pairs.

A coefficient is reachable by a gradient because the residual reads it through a
resolver bound to a live tensor, so one frozen `System` serves both modes. Each
coefficient is parameterised by an *unconstrained* variable through its
`transform` -- `"positive"` (softplus) or `"bounded"` (scaled sigmoid) -- so the
constraint holds by construction, with no projection step and no clipping.

The default optimiser is `"cubic_gauss_newton"`, not the forward driver's
`"lbfgs"`, and the gap is large rather than marginal; see the
[benchmarks page](../benchmarks.md#inverse-problems-coefficient-recovery). RAR and
`loss_balancing="grad_norm"` (which now balances three terms) compose here, because
both drivers run the same loop.

Honest boundaries:

- **torch-only**, like the rest of the second-order surface.
- **Identifiability is the caller's responsibility and fails silently.** A
  coefficient the data cannot see does not raise; it simply stops moving.
  Structurally, `wave` sees only `speed ** 2` (the sign is unrecoverable), and a
  coefficient multiplying a term that vanishes on the observed region has no
  gradient. Validate against a synthetic study before trusting real data.
- **A forward driver refuses a system with unbound unknowns** rather than quietly
  solving at the initial guess. Pin them with `bind_unknowns({...})` to solve the
  same `System` forward -- which is exactly how synthetic observations are made.

[`docs/examples/pinn_solver_inverse.py`](../examples/pinn_solver_inverse.py) is the
runnable version of both this and RAR.

## JAX twin

The JAX backend (`omnibias.pinn.solver.jax`) is the bit-identical twin: given the
same ansatz parameters, `omnibias.pinn.solver.jax.assemble` reproduces the torch
residual rows to double-precision round-off (the parity test runs jax in x64). It
provides the linear-collocation `solve_least_squares`, the spectral
method-of-lines, and the same integrators (`rk4_step`, `linear_jet_step`,
`burgers_jet_step`, `implicit_linear_step`). The nonlinear residual-minimisation
`solve_optimize` driver is torch-only in v1.

## Optional certified mode

`omnibias.pinn.solver.verify` wraps `omnibias.core.verified.pde_certificate` for a
solved field on the linear, steady canonical problems, sealing an a-posteriori
sup-norm error certificate (`unproven_claim: False`). The interior / boundary
residuals are rigorous interval enclosures; the well-posedness stability constants
are the caller's recorded obligation. Scope is deliberately modest-scale and
linear.

::: omnibias.pinn.solver.verify
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - extract_layers
        - certify_poisson
        - certify_linear_bvp

Status: Alpha submodule (`omnibias.pinn.solver`) of the Beta `omnibias-pinn`
package (folded from the former `omnibias-pde` `0.1.0a1`).

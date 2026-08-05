# omnibias.score.flow — continuous normalizing flows

> **Folded package.** This is the alpha `omnibias.score.flow` submodule of
> `omnibias-score` (formerly the standalone `omnibias-flow` package). Imports use
> `omnibias.score.flow`; the numerics are unchanged.

Continuous normalizing flows (CNF) with an **exact** trace-of-Jacobian. For an
ODE `dx/dt = v(t, x)` the log-density evolves as
`d log p / dt = -tr(d v / d x) = -div_x v`, and `exact_trace_jacobian` computes
that divergence *deterministically* through the `omnibias-fields` divergence
operator (zero variance), in contrast to FFJORD's stochastic **Hutchinson**
estimator `εᵀJε`. Both are available -- `hutchinson_trace_jacobian` provides the
one-VJP baseline, and `cnf_dynamics` / `integrate_cnf` / `log_prob` take a
`trace_fn` hook so either flows through the same augmented integrator (default
`None` = exact). It uses the same `omnibias-fields` field-composition pattern as
the score / SDE ops in this package.

The exact trace costs `O(d)` VJPs vs Hutchinson's one; it is the right choice in
low dimension and noise-limited regimes, while Hutchinson's unbiased noise averages
out on well-conditioned, adequately-trained objectives (see the honest
`flow_experiments/` study in the separate `omnibias_experiments` project).

## Ops (torch)

::: omnibias.score.flow.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.score.flow.jax.ops`) mirrors the torch ops via the
`omnibias.fields.jax` divergence; `exact_trace_jacobian` and `integrate_cnf`
agree across backends to `rtol=1e-7` (integration accumulates floating-point
operations, so this is looser than the bit-identical kernels).

Status: Alpha submodule (`omnibias.score.flow`) of the alpha `omnibias-score`
package (folded from the former `omnibias-flow` `0.1.0a1`).

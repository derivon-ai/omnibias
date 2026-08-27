# Inequality engine (09-30)

A **front door** for systems of inequalities. One `Conjecture` kind
(`inequality_system`) dispatches to linear / polynomial / Boolean /
finite-CSP backends. The optimizer proposes; an exact `Q` / GF(2) /
enclosure / tiny enum proves. Soft RMSE is never an `ExactCheck`.

This is **not** a new LP algorithm, **not** a complete CSP solver, and
**not** a P vs NP claim. `InfeasibleProblemError` is `BLOCKED`, not
emptiness. `theorem_prover_verified` stays false unless a genuine
`lake build` earned it.

Home: `omnibias.core.proof.inequality`. Adapters register at import.
Cookbook: [Inequality engine](../cookbook/inequality-engine.md).

## API

::: omnibias.core.proof.inequality
    options:
      show_root_heading: false
      heading_level: 3

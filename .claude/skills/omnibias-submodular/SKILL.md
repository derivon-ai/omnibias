---
name: omnibias-submodular
description: Maximize submodular set functions with a differentiable multilinear extension, continuous greedy, pipage rounding, and a min-of-bounds gap certificate, plus exact Lovász minimization. Use when inventing a new set-function family or a certified greedy layer.
---

# omnibias-submodular

Differentiable + certified submodular optimization for omnibias: the
multilinear-extension relaxation solved by continuous greedy (Frank-Wolfe over a matroid
polytope), pipage / swap rounding, accelerated (lazy / stochastic) and knapsack greedy,
non-monotone (double / measured continuous) greedy, streaming, and a certified (1 - 1/e)
/ curvature-sharpened approximation guarantee plus a min-of-bounds optimality-gap
sandwich; coverage / facility-location / budget-additive / log-det-DPP / graph-cut
families + a submodular algebra over uniform / partition / laminar / graphic /
transversal / intersection matroids and knapsack budgets (jax + torch twins), and honest
P-class exact submodular minimization (Lovász extension + Fujishige-Wolfe
min-norm-point).

## Why nested AD fails

Greedy set selection has no gradient. Nested AD cannot evaluate the multilinear
extension's exact partials. Generic submodular libraries do not sandwich f(S) against a
certified U(S) with torch/jax twins.

## What only this tower unlocks

Differentiable + certified submodular optimization for omnibias: the
multilinear-extension relaxation solved by continuous greedy (Frank-Wolfe over a matroid
polytope), pipage / swap rounding, accelerated (lazy / stochastic) and knapsack greedy,
non-monotone (double / measured continuous) greedy, streaming, and a certified (1 - 1/e)
/ curvature-sharpened approximation guarantee plus a min-of-bounds optimality-gap
sandwich; coverage / facility-location / budget-additive / log-det-DPP / graph-cut
families + a submodular algebra over uniform / partition / laminar / graphic /
transversal / intersection matroids and knapsack budgets (jax + torch twins), and honest
P-class exact submodular minimization (Lovász extension + Fujishige-Wolfe
min-norm-point).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Exact P-class minimization (Lovász + Fujishige-Wolfe) ships as a solver,
not a complexity-class statement.

| You want | Import |
| --- | --- |
| Multilinear extension / greedy | `omnibias.submodular` |
| Coverage / facility / graph-cut | `omnibias.submodular` families |

## Extend

- Source: [`packages/omnibias-submodular`](../../../packages/omnibias-submodular).
- Namespace: `omnibias.submodular`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-submodular/tests -q`.
- Compose with `omnibias-discrete`, `omnibias-combinatorics`, `omnibias-measure` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A coverage instance whose continuous-greedy path, pipage rounding, and min-of-bounds
certificate recover OPT against brute_force_max on a CI-size set.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)

---
name: omnibias-symbolic
description: Discover ODE / PDE identities from exact neural jets — library-free SINDy, field-law recovery, piecewise automata, and AutoML surrogates. Use when inventing a new discovery loop, reading a governing equation off a fitted field, or when the user mentions neural-jet discovery or PDE identification.
---

# Neural-jet equation discovery

`omnibias.symbolic` fits a smooth neural field, reads its **exact closed-form
jet**, and recovers the governing ODE / PDE. Names below re-export from the
package root: `from omnibias.symbolic import <name>`.

## Why nested AD fails

Finite-difference SINDy on autodiff fields is noise-dominated at the orders
that distinguish heat from wave from Burgers. Nested AD jets are truncated,
backend-split, and expensive exactly where the library needs mixed partials.
Generic symbolic regression never sees `sigma^(n)` exactly, so recovered PDEs
are stencil artifacts.

## What only this tower unlocks

A library-free discoverer (`NeuralJetDiscoverer`) that recovers closed-form
differential identities from exact activation jets. Field-law recovery
(heat / wave / Burgers / Laplace) reads `extract_field_jet`. Piecewise /
hybrid automata harden gates under temperature collapse (`beta -> inf`) then
STLSQ-polish on the founding jet (`delta -> 0`). That split — exact jet plus
annealed partition — is the workload nested AD + FD-SINDy cannot run.

Dense cheat-sheet: `docs/handbook/ai-quickstart.md`.

## Use

| You want | Key entry points |
| --- | --- |
| Discover a 1-D ODE / activation identity | `discover_activation_identity`, `NeuralJetDiscoverer` |
| Fit a smooth field + its jet | `fit_neural_field_1d`, `fit_neural_field_nd`, `extract_field_jet` |
| Gradient / Hessian / Laplacian of a fitted field | `field_gradient`, `field_laplacian` |
| Discover a PDE (heat / wave / Burgers / Laplace) | `FieldLawDiscoverer`, `discover_field_pde_law`, `make_heat_field_split` |
| Piecewise / hybrid automaton | `fit_piecewise_law`, `fit_piecewise_ode_law`, `HybridAutomaton` |
| Learn gates from data, then harden + STLSQ | `fit_learned_piecewise_ode` |
| Curvature / metric on learned charts | `MetricField`, `laplace_beltrami`, `pullback_metric_field`, `scalar_curvature` |
| Lie point symmetries | `omnibias.symbolic.symmetry` |
| Rationalize-and-certify | `omnibias.symbolic.certify` | `rationalize_and_certify_discovery` |
| Piecewise SINDy on a partition | `omnibias.symbolic.piecewise` |
| Ingest pack tables | `omnibias.symbolic.ingest` |

```python
from omnibias.symbolic import discover_activation_identity

result = discover_activation_identity("exp", candidate_lhs_orders=(1,))
print(result.formula())            # -> "dy = 1*y"
```

A random-feature field is accurate inside its training box; request
`max_order` high enough for the PDE (Hessian needs 2). Pass
`random_state=<int>` for bit-reproducible fields on a given platform.

Cookbook: `docs/cookbook/piecewise-hybrid-automaton.md`.

## Extend

- Source: `packages/omnibias-symbolic`. Tests:
  `python -m pytest packages/omnibias-symbolic/tests -q`.
- Compose with `omnibias-holonomic`, `omnibias-difference`, `omnibias-fields`,
  `omnibias-partition`, `omnibias-discovery-engine`, `omnibias-geometry`.
- New discovery families register through `omnibias-discovery-engine`.

## Next invention

A planted Burgers field whose `FieldLawDiscoverer` recovers the exact
coefficient vector from the closed-form jet at a noise level that
finite-difference SINDy misses, then a Lie-symmetry nullspace that matches
the determining matrix over Q.

## Bakeoffs

`docs/benchmarks/public_csv_discovery_smoke.json`,
`docs/benchmarks/symmetry_discovery_smoke.json`.

## Further references

- API: `docs/api/symbolic.md`
- Handbook index: `docs/handbook/index.md`

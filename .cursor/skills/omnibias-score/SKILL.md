---
name: omnibias-score
description: Compose score / SDE operators — score, Ito generator, Fokker-Planck — from closed-form field gradients and Hessians, plus coupling-jet CNFs. Use when inventing a new SDE residual or an exact log-det flow.
---

# omnibias-score

Score-based / SDE operators for omnibias: the closed-form score (grad log p), the Ito
infinitesimal generator, and the Fokker-Planck adjoint, composed from the
omnibias-fields closed-form gradient / Hessian primitives.

## Why nested AD fails

Score matching via nested AD Hutchinson traces is noisy and order-capped. CNFs through
generic autodiff cannot use closed-form `sum log sigma'`. There is no Fokker-Planck
residual enclosure on a vanilla score net.

## What only this tower unlocks

Score-based / SDE operators for omnibias: the closed-form score (grad log p), the Ito
infinitesimal generator, and the Fokker-Planck adjoint, composed from the
omnibias-fields closed-form gradient / Hessian primitives.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Pure composition of `omnibias-fields` grad / Hessian ops.

| You want | Import |
| --- | --- |
| Score / Ito / Fokker-Planck | `omnibias.score` |
| Coupling-jet flow | `omnibias.score.flow.{torch,jax}.jet_flow` |
| Exact score matching | `omnibias.score.{torch,jax}.score_matching` |

## Extend

- Source: [`packages/omnibias-score`](../../../packages/omnibias-score).
- Namespace: `omnibias.score`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-score/tests -q`.
- Compose with `omnibias-fields`, `omnibias-verify`, `omnibias-measure` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A coupling-jet CNF whose exact `sum log sigma'` matches a Hutchinson baseline to 1e-10
and whose Fokker-Planck residual is a sealed Interval.


## Bakeoffs

Nested-AD cost and closed-form accuracy live in
[`docs/benchmarks/`](../../../docs/benchmarks/):
`laplacian_scaling.json`, `polylaplacian_order.json`,
`derivative_order.json`, `jet_vs_nested_ad_smoke.json`.
Heavy regeneration follows the workspace compute rule, not a skill taboo.

## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)

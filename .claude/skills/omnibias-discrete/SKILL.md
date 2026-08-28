---
name: omnibias-discrete
description: Build differentiable certified discrete optimization on the DiscreteProblem seam: anneal_descent, rounding / k-flip decode, brute-force oracle, and Lasserre/SOS gap certificates. Use when inventing a new binary energy or a MaxSAT / QUBO / submodular consumer.
---

# omnibias-discrete

Shared differentiable + certified discrete-optimization substrate for omnibias: the
DiscreteProblem seam, the annealed temperature-collapse (beta -> inf) sigmoid relaxation
solved by unrolled descent (torch + jax twins), a rounding + k-flip local-search decoder
with a brute-force oracle, and a rigorous optimality-gap certificate (Lasserre /
moment-SOS lower bound over the Boolean hypercube). Ships a MaxSAT front-end as its
first in-tree consumer.

## Why nested AD fails

Nested AD through `{0,1}^n` is undefined. Gumbel-softmax without a DiscreteProblem
polynomial cannot feed SOS. Generic annealing has no oracle sandwich and no
bit-identical torch/jax twins.

## What only this tower unlocks

Shared differentiable + certified discrete-optimization substrate for omnibias: the
DiscreteProblem seam, the annealed temperature-collapse (beta -> inf) sigmoid relaxation
solved by unrolled descent (torch + jax twins), a rounding + k-flip local-search decoder
with a brute-force oracle, and a rigorous optimality-gap certificate (Lasserre /
moment-SOS lower bound over the Boolean hypercube). Ships a MaxSAT front-end as its
first in-tree consumer.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Temperature collapse (`beta -> inf`) is the feasibility axis: the sigmoid
relaxation hardens to a vertex. The energy gradient is closed-form.

| You want | Import |
| --- | --- |
| Problem seam | `omnibias.discrete.DiscreteProblem` |
| Annealed descent | `omnibias.discrete.{torch,jax}.anneal_descent` |
| MaxSAT consumer | `omnibias.discrete.maxsat` |
| Soft evolution | `omnibias.discrete.evolution` |

## Extend

- Source: [`packages/omnibias-discrete`](../../../packages/omnibias-discrete).
- Namespace: `omnibias.discrete`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-discrete/tests -q`.
- Compose with `omnibias-discrete-consumer`, `omnibias-qubo`, `omnibias-logic`, `omnibias-sos` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A new DiscreteProblem whose closed-form flip_deltas, SOS bound, and brute-force oracle
agree on a named family at n that still fits CI.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)

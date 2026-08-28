---
name: omnibias-discrete-consumer
description: Add a differentiable certified discrete-optimization consumer on the omnibias-discrete substrate — DiscreteProblem seam, bit-identical torch/jax anneal_descent, certified lower bound, rounding / k-flip decode. Use when building a beta->inf discrete package or a new front-end on omnibias.discrete.
---

# Building a discrete-optimization consumer

`omnibias-discrete` is the shared `encode -> relax -> decode -> certify`
engine. A new consumer is thin: write the energy once; the substrate gives
the relaxation, decoder, oracle, and certificate. Worked references:
`omnibias.discrete.maxsat` and `omnibias.qubo`.

## Why nested AD fails

Nested AD through `{0,1}^n` is undefined. A one-off Gumbel softmax without a
polynomial cannot feed SOS. Forking anneal loops per package splits torch/jax
parity and the gap certificate.

## What only this tower unlocks

Temperature collapse (`beta -> inf`) sigmoid relaxation with a closed-form
energy gradient, bit-identical twins, rounding / k-flip decode, brute-force
oracle, Lasserre/SOS (or spectral / box-QP) sandwich. That recipe is the
feasibility-axis invention nested AD cannot run.

## Use

Scaffolding from `omnibias-new-package`; validation from
`omnibias-empirical-validation`. This skill is the domain contract.

**Seam — `omnibias.discrete.DiscreteProblem`:**

- `n: int`
- `energy(x)` — point `(n,) -> float` and batch `(m, n) -> (m,)`
- `to_polynomial() -> omnibias.sos.Polynomial` — agrees with `energy` on the cube
- optional `flip_deltas(x)` — closed-form bit-flip deltas for the decoder

**Relax.** Thin wrappers in `<pkg>/{torch,jax}/relaxation.py` calling
`omnibias.discrete.{torch,jax}.anneal_descent(grad_x_fn, scale, n, schedule)`.
Supply a closed-form `grad_x_fn` (chain the sigmoid yourself).

**Decode / certify.** Reuse rounding + k-flip + `brute_force` + `certify_gap`.
Pick the lower bound the math supports (SOS/Lasserre, spectral, LP dual).

Temperature collapse is the feasibility axis; founding `delta -> 0` is the
tower that differentiates the sigmoid. Keep `__lineage__` honest.

## Extend

Compose with `omnibias-discrete`, `omnibias-qubo`, `omnibias-logic`,
`omnibias-sos`, `omnibias-convex`. Tests must agree `energy` vs
`to_polynomial` on the cube and torch/jax parity of the relaxation.

## Next invention

A new DiscreteProblem (packing, transport, or a named logic fragment) whose
closed-form `flip_deltas`, SOS bound, and oracle sandwich a CI-size instance
with a `gates` JSON.

## Further references

- `docs/api/` pages for discrete / qubo / logic
- `packages/omnibias-discrete`

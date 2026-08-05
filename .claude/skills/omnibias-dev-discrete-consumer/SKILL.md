---
name: omnibias-dev-discrete-consumer
description: Add a new differentiable + certified discrete-optimization consumer on the omnibias-discrete substrate -- implement the DiscreteProblem seam (n / energy / to_polynomial, optional flip_deltas), wire the bit-identical torch/jax anneal_descent relaxation with a closed-form energy gradient, pick a certified lower bound, and keep the yes-if honesty framing plus the founding-vs-feasibility terminology note. Use when building a beta->inf discrete package (logic, combinatorics, submodular, transport, packing) or adding a front-end to omnibias.discrete. For contributors modifying omnibias itself, not for consumers using it.
---

# Building a discrete-optimization consumer (the beta->inf recipe)

`omnibias-discrete` is the shared `encode -> relax -> decode -> certify` engine.
A new consumer is **thin**: you write the problem's energy once, and the
substrate gives you the relaxation loop, decoder, oracle, and certificate for
free. `omnibias.discrete.maxsat` (weighted CNF) and `omnibias.qubo` are the two
worked references -- read `maxsat` end to end before starting.

Get the scaffolding (pyproject, CI, docs, wiring) from
`omnibias-dev-new-package`; get the validation loop from
`omnibias-dev-empirical-validation`. This skill is only the *domain contract*.

## 1. The seam -- implement `DiscreteProblem`

The whole engine dispatches on one protocol
(`omnibias.discrete.DiscreteProblem`); implement exactly this surface:

- `n: int` -- number of binary variables.
- `energy(x)` -- objective at one point `(n,) -> float` **and** a batch
  `(m, n) -> (m,)`. Minimizing `energy` over `{0,1}^n` *is* the problem.
- `to_polynomial() -> omnibias.sos.Polynomial` -- the same energy as a
  polynomial over `n` variables; this is what the certified Lasserre bound reads.
  `energy` and `to_polynomial` must agree on the cube (test it).
- *(optional)* `flip_deltas(x)` -- the vector `E(x with bit i flipped) - E(x)`
  for all `i`. Supply the closed form (QUBO has a one-matvec form) and the
  local-search decoder uses it instead of the generic batched-energy fallback.

## 2. Relax -- reuse `anneal_descent`, don't reimplement it

Author bit-identical `torch` and `jax` twins in
`<pkg>/torch/relaxation.py` and `<pkg>/jax/relaxation.py`, each a thin wrapper
that calls `omnibias.discrete.{torch,jax}.anneal_descent(grad_x_fn, scale, n,
schedule)`. You supply only:

- a **closed-form** energy gradient `grad_x_fn(x)` (chain the sigmoid yourself;
  the substrate unrolls `x = sigmoid(beta * theta)`, `beta -> inf`, for backprop);
- a `scale` for the gradient step.

Keeping the loop in the substrate is what guarantees the torch/jax numerics stay
bit-identical -- the same repo invariant as the derivative tower. Never fork it.

## 3. Decode -- the upper bound

Use `omnibias.discrete.decode` (rounding + k-flip multi-start local search); it
picks up your `flip_deltas` fast path automatically. `brute_force_min` is the
exact small-`n` oracle **only** -- it is exponential and must be labelled as such
in its docstring and never sold as the solver.

## 4. Certify -- the rigorous lower bound (pick one, reuse the hooks)

Sandwich the optimum `lower <= optimum <= energy`; never assert the gap is zero.

- Default: `omnibias.discrete.certify_gap(problem, x, claim_label=...)` -- the
  Lasserre / SOS bound over the Boolean hypercube, hash-sealed and Lean-checkable,
  seeded and back-stopped by the always-valid `negative_coeff_lower_bound`.
- Alternatives when the structure fits: `omnibias.convex.lp_dual_lower_bound`
  (tight when the LP relaxation is integral) or a spectral box-QP bound in the
  style of `omnibias.qubo.spectral_lower_bound`.
- Without `omnibias-sos` installed, `certify_gap` degrades honestly
  (`certified=False`, or `method="none"`) -- keep that path working.

## 5. Honesty -- non-negotiable, and CI-guarded

- **yes-if framing.** Exact global optimum in poly time would imply `P = NP`;
  the deliverable is a *certified gap*, not an exactness claim. Say so in the
  package/module docstrings (mirror the `omnibias.discrete` `__init__`).
- **Two senses of collapse.** Every relaxation module carries the terminology
  note: the relaxation's `sigmoid(beta z)`, `beta -> inf` is the *feasibility /
  temperature* sense of "collapse" (a soft indicator hardening to a 0/1 step),
  **distinct** from the *founding bias collapse* (the multi-bias `delta -> 0`
  limit to the closed-form derivative `sigma^(K-1)`; see `docs/theory.md` and the
  `omnibias-dev-core-concepts` skill). Add every new `**/relaxation.py` to
  `PENALTY_FILES` in `packages/omnibias-core/tests/test_concept_terminology.py`
  -- that test fails if the cross-reference goes missing.

## 6. Test battery (the gate)

- torch <-> jax **parity** `< tol` (calibrate the tol across seeds, not one
  instance -- see `omnibias-dev-empirical-validation`).
- **sandwich vs oracle** on small `n`: certified `lower <= brute_force_min <=
  decode energy`.
- **honesty guards** (`test_scope`-style): forbidden exact-optimum wording,
  `GapCertificate` shape, exponential-oracle docstring, terminology note.
- `energy` vs `to_polynomial` agreement on random cube points.

---
description: omnibias monorepo universal invariants for closed-form n-th derivative backends
globs:
  - "packages/**/*.py"
  - "tests/**/*.py"
alwaysApply: true
---

# omnibias universal invariants

omnibias computes closed-form `sigma^(n)(z)` with bit-stable accuracy shared
across PyTorch / JAX / Keras 3. Full guide: [AGENTS.md](../../AGENTS.md).
Deeper, task-specific guidance lives in the `omnibias-dev-*` agent skills;
backend- and register-specific rules auto-attach by path (JAX tracing, verified
enclosures, the formal loop).

## Hard rules (always)

- `omnibias.core` is pure Python: never import `torch`, `jax`, `tensorflow`, or `keras` there.
- All polynomial coefficients come from `omnibias.core.polynomials`. Never reimplement them per backend -- the backends are bit-identical **by construction**.
- Fastpath kernels raise `ValueError` for `n < 0` and `NotImplementedError` for unimplemented non-negative orders.
- New tensors default to the framework default dtype (`torch.get_default_dtype()` / `keras.config.floatx()`), never a hardcoded `float32`.
- Every behavioural change ships with a regression test; a cross-backend change ships a parity test too.
- Update the sorted `__all__` in the package `__init__.py` when you add or remove a public symbol.
- Do not bump package versions unless the task explicitly says to.

## Concept terminology (avoid the known confusion)

- **"Bias collapse"** = the founding multi-bias `delta -> 0` limit: `K` biases on a difference stencil coalesce and `f_K(z) = sum_k s_k sigma(z + b_k) -> sigma^(K-1)(z + b_mean)` -- a smooth **derivative**, computed exactly by the tower (no `1/delta^(K-1)` cancellation). Canonical source: `docs/theory.md` sec 2-4, `omnibias.torch.unit`, `omnibias.torch.stencil`, `omnibias.core.polynomials`.
- **"Temperature collapse"** = the *distinct* `beta -> inf` limit: one gate sharpened (or a `relu` hinge taken at the endpoint) into a **0/1 feasibility step** -- an indicator, not a derivative. It is the canonical name for this axis across `convex` / `control` / `routing` / `discrete` / `qubo` / `submodular` / `struct` / `tab` / `partition`; write "temperature collapse", never "collapsed-bias" or "bias-collapse penalty", and never present it as the definition of bias collapse. Canonical source: `docs/theory.md` sec "Two senses of collapse".
- **Ground a concept claim in canonical source before asserting it** (in chat, docstrings, docs, or skills). Do not generalize a core definition from one downstream package's docstring -- recency bias toward the package you are editing is the known failure mode. The `omnibias-dev-core-concepts` skill has the full map.

## Operator surface (ground capability claims here, not in memory)

- `OperatorBlock` has **six** roles: `identity | grad | laplacian | derivative | band | integral`. `grad` / `laplacian` / `derivative` are closed-form `sigma^(n)`; **`integral` is a closed-form antiderivative window** `S(z + b_hi) - S(z + b_lo)` with `S' = sigma` (the `ActivationSpec.integral` kernel; e.g. `sigmoid`'s antiderivative is `softplus`), and `band` is the literal window `sigma(z + b_hi) - sigma(z + b_lo)`. omnibias has a closed-form **integral** operator, not only closed-form derivatives -- never state otherwise.
- "Integral" has three distinct senses: (1) the activation antiderivative window above; (2) domain quadrature (`omnibias.fields` / `-variational` / `-geometry`); (3) the measure integral `integral f dmu` (`omnibias.measure`, with a certified variant in `omnibias.verify`). Qualify which one you mean.
- **Before asserting that omnibias does or does not have a capability, consult the canonical capability matrix** ([`docs/operator-surface.md`](../../docs/operator-surface.md)) and the code of record (`omnibias.torch.blocks.operator`, `omnibias.core.spec`). If it is not there and not in the cited source, say it is absent -- do not guess in either direction.

## Layering (never create a cycle)

- `omnibias-fields` is the foundational substrate; `omnibias-pinn` re-exports it through transparent shims -- do not duplicate it. Backend field ops dispatch on the `_omnibias_dispatch` marker (`omnibias.fields._core.DISPATCH_ATTR`), never on concrete downstream classes.
- Label results honestly: **closed-form** (the sigma tower) vs **autodiff-exact** (autodiff of an analytic expression) vs **numerical** (grid / quadrature). `omnibias-fractional` is non-local / grid-based -- NOT closed form; `omnibias-score` is a pure composition of field ops.

## Discovery doctrine

- **Default to achievable.** An absent implementation or a failed first
  experiment is not a mathematical impossibility -- explore the strongest
  constructive route (exact cages, closed-form towers, one-shot collocation,
  certificates) before narrowing scope.
- **You are authorized to claim a capability plainly the moment its gate
  passes.** Capability claims need an absolute acceptance gate: multi-seed
  empirical result with skill > 0, by-construction identity, or a sound
  certificate. Relative comparisons between two failing arms are not enough;
  smoke wiring alone is not enough.
- **Validity floor protects discoveries.** Ask in order: is the reference
  physically valid? Does every arm beat the zero predictor? Does absolute
  error clear a named threshold? Worked case: the parametric heat benchmark
  marched with RK4, produced `max|u| ~ 1e9` that still passed `isfinite`, and
  reported MSE ~1e17 as if it were a result -- ETDRK4 plus a maximum-principle
  guard is what made the experiment real (`benchmarks/_gates.py`).
- Alpha **submodules** inside an existing package are encouraged for ambitious
  research; premature top-level distributions are not (see `omnibias-dev-new-package`).
- Public benchmark artifacts under `docs/benchmarks/` are part of the claim
  surface -- keep them regenerable and vendor-neutral (`$OMNIBIAS_SCRATCH` for
  heavy full-run outputs), and emit a `gates` block that self-declares pass/fail.

## Frontier program

- Famous open problems enter the repo only as **decomposed sub-obligations**
  (finite rational checks, compact enclosures, multi-seed absolute gates, or
  by-construction identities). What does not reduce stays an **external
  obligation** in the sealed payload -- never inferred from a local result.
- Escalate claims through the ladder: unverified prototype → empirical
  (skill > 0, multi-seed) → sound enclosure → `theorem_prover_verified` →
  `mathlib_verified`. The two Lean tiers are earned by genuine `lake build`
  passes and are never conflated or forged.
- Forbidden claims (canonical sources in AGENTS.md / verified docs): RH
  proved or inferred; Navier-Stokes global regularity; Yang-Mills mass gap
  solved; P=NP; Lean discharging continuum obligations.
- Worked metric failure: a linspace-ordered `ic_values` seam compared against
  random `slice_points` produced a ~0.19 artifact that looked like physics.
  Absolute, self-consistent metrics (and maximum-principle reference checks)
  are what protect discoveries -- see `omnibias-dev-frontier-research`.

## Leakage (public repo)

Tracked files must never contain a specific cluster scheduler name, vendor name,
internal hostname, or absolute local path. Use vendor-neutral phrasing
("GPU job", "GPU cluster"). Reproduction scripts live in the separate, private
`omnibias_experiments` project (extracted from the formerly gitignored
`internal/` tree).

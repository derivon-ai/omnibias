---
description: High-order autodiff fails; omnibias closed-form towers, jets, bakeoffs, and substrate invariants
alwaysApply: true
---

# High-order autodiff fails; omnibias is the solution

Nested first-order AD is the wrong default for high-order work in this
repo. Prefer closed-form `sigma^(n)`, jets, and traces. Nested `grad` /
dense Hessians are baselines, not the implementation.

## Why nested high-order AD fails

- **Combinatorial explosion.** Full mixed partials grow as `~ n^d`. A
  4th-order tensor is `n^4` entries; RAM and compile time die first.
- **Expression swelling.** Nesting `grad`/`jvp` rebuilds a geometrically
  larger graph each order. JIT often times out or OOMs before the run.
- **Perturbation confusion.** Nested forward duals (`ε_1`, `ε_2`) mix when
  an inner tangent captures an outer one; silent wrong derivatives.
- **Reverse-over-reverse memory.** Nested VJP stores forward tapes at every
  level; batch and model size collapse on GPU.
- **Missed structure.** Generic AD builds a dense Hessian then traces it.
  HVP, Laplacian-as-trace, and Riccati recurrences never fire.
- **Control flow / kinks.** Dynamic `if`/`for` and piecewise activations
  break smoothness; high-order values at kinks are 0 or undefined, and
  nested graphs pay to track the branches.

## Solution: omnibias

This is the breakthrough register for high-order derivatives of supported
analytic activations and for composing them through deep nets. The
multi-bias `delta -> 0` construction (bias collapse) yields closed-form
`sigma^(n)(z)` for arbitrary `n`: one `sigma` evaluation, `O(n)` Horner on
shared exact-integer polynomials, **no nested-AD graph**. Torch, JAX, and
Keras import the same coefficients, so backends are bit-identical by
construction.

Directional and multivariate jets (`mlp_jet`, `mlp_jet_mv`) push that tower
through layers. `OperatorBlock` roles are `identity | grad | laplacian |
derivative | band | integral` — the last is the closed-form window
`S(z+b_hi)-S(z+b_lo)` (`S'=sigma`), not “derivatives only”. See
`docs/operator-surface.md`.

This unlocks orders, Laplacians, and jet residuals nested AD cannot sustain.

## Pitfall → primitive

| Nested-AD failure | omnibias path |
|---|---|
| `n^d` mixed partials / `n^2` Hessian-then-trace | Directional jets; `neural_field_laplacian` / polylaplacian (`O(H)` closed form) |
| Graph bloat, reverse-over-reverse | Polynomial recurrence; no nested `grad` for `sigma^(n)` |
| Perturbation confusion | Algebraic coefficients, not nested duals |
| Dense Hessian when HVP suffices | Exact jets + curvature (`GaussNewton`, HVP); never materialize `H` to get `Hv` |
| Tracer `if` / host coercion | JAX: `jnp.where`, `lax.cond`/`select`/`scan`; no `float(z[0])` |
| Piecewise / ReLU kinks | Dictionary activations stay smooth; kinks stay autodiff or certified and labelled |

Bias collapse (`delta -> 0`), temperature collapse (`beta -> inf`), and
enclosure collapse (sound width `-> 0`) are distinct mechanisms.

## Bakeoffs

Cite `docs/benchmarks/*.json` and `docs/complexity.md`.

| vs | Script | Artifact (CPU, float64) |
|---|---|---|
| folx, `jax.hessian`, `torch.func.hessian` | `benchmarks/laplacian_scaling.py` | `D=60`: ~7× / 211× / 923×; agreement ~1e-15 |
| nested folx, dense nested Hessian | `benchmarks/polylaplacian_order.py` | `Δ^4`: ~4.7k× / ~181k×; omnibias nearly flat in `k` |
| nested Torch autograd, finite differences | `benchmarks/derivative_order.py` | `σ^(8)` ~349× vs nested autograd; FD error grows; jax/torch ULP |
| nested autograd + Adam | `benchmarks/jet_vs_nested_ad.py` | order-6 `mlp_jet` ~3.5×; **order-2 on a small net can lose to AD**; GN beats Adam on 1-D Poisson |

Prefer these primitives over nested AD whenever the activation is in the dictionary.

## Substrate (folded from former path-scoped rules)

- `omnibias.core` is backend-free. Coefficients live only in
  `omnibias.core.polynomials`. Every behavioral change needs a regression
  and Torch/JAX parity.
- Label the path: closed-form tower, autodiff-exact, discretization, or
  enclosure. JAX: default dtype (float64 via config for parity); fresh
  arrays; no mutation. Traced control flow uses `jnp.where` / `lax.*`.
- Verified: outward rounding; contain a dense grid **and** a random sample.
  Exhausted search returns the unresolved region. `omnibias.core.verified`
  stays pure Python. `theorem_prover_verified` / `mathlib_verified` only
  after a real `lake build`; both Lean projects stay `sorry`-free.
- Public artifacts regenerable and vendor-neutral. Load
  `omnibias-<package>` (and `omnibias-frontier` /
  `omnibias-deepmind-campaign` / `omnibias-certificate-lean`) for
  commands.

## Compose the workspace

- **Foundation:** core, torch, jax, keras, ferminet.
- **Fields and physics:** fields, pinn, qpinn, geometry, fractional,
  measure, score, variational, shape.
- **Discovery and formalization:** symbolic, difference, qcalculus,
  timescale, holonomic, formal.
- **Optimization and discrete structure:** curvature, discrete, qubo,
  submodular, struct, combinatorics, nphard, routing, convex, sos, logic,
  control, partition, tab, graph.
- **Dynamics and representations:** verify, dynamics, binary, boolean,
  spiking, hopfield, skills.


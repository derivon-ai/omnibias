# omnibias-discrete

The shared **differentiable + certified discrete-optimization substrate**: the
`encode -> relax -> decode -> certify` machinery extracted from `omnibias-qubo` so a
second consumer (MaxSAT, here) can reuse it. Any object implementing the
`DiscreteProblem` seam (`n` + `energy` + `to_polynomial`) plugs into the whole pipeline.

Minimizing a pseudo-Boolean energy over `x ∈ {0,1}ⁿ` is NP-hard, so there is no poly-time
differentiable map to the *exact* global optimum (that would imply `P = NP`). The sound
object is a **yes-if** sandwich:

\[
\underbrace{\ell}_{\text{certified lower bound}} \;\le\; \text{optimum} \;\le\;
\underbrace{E(z)}_{\text{decoded binary point (upper bound)}} .
\]

Terminology: the relaxation's `sigmoid(beta·)`, `beta → ∞` is the **feasibility /
temperature** sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct
from the **founding bias collapse** -- the multi-bias `delta → 0` limit to the
closed-form derivative `sigma^(K-1)` (see [Theory](../theory.md)).

## The problem seam & Boolean constraints

::: omnibias.discrete._core.problem
    options:
      show_root_heading: false
      heading_level: 3

## Annealing schedule

::: omnibias.discrete._core.schedule
    options:
      show_root_heading: false
      heading_level: 3

## Result containers

::: omnibias.discrete._core.solution
    options:
      show_root_heading: false
      heading_level: 3

## Decoder & exact oracle (numpy)

::: omnibias.discrete._core.decode
    options:
      show_root_heading: false
      heading_level: 3

## Lower bounds (numpy)

::: omnibias.discrete._core.bound
    options:
      show_root_heading: false
      heading_level: 3

## Optimality-gap certificate

::: omnibias.discrete.certify
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable relaxation core (JAX)

::: omnibias.discrete.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Backend twin (torch)

Bit-identical PyTorch twin of the relaxation core.

::: omnibias.discrete.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## MaxSAT front-end (first consumer)

::: omnibias.discrete.maxsat.problem
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.discrete.maxsat.frontends
    options:
      show_root_heading: false
      heading_level: 3

### MaxSAT relaxation (JAX + torch)

::: omnibias.discrete.maxsat.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 4

::: omnibias.discrete.maxsat.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 4

## Sparse recovery (`omnibias.discrete.sparse`)

Certified sparse recovery driven by the `l_p → l_0` **penalty-exponent collapse** (a
concave `Σ_i x_i^p`, `p → 0`, whose reweighting drives small entries to `0`). This is the
**feasibility** sense of collapse, distinct from the founding `delta → 0` bias collapse.
Three layered forks, each explicit about which object its certificate seals:

- **Fork A** -- `SupportSelectionProblem`: the binary `z ∈ {0,1}ⁿ` *is* the support, so
  `E(z) = ½‖A z − b‖² + λ 1ᵀz` is a QUBO sealed directly by `certify_gap` (Lasserre/SOS).
- **Fork B** -- `BestSubsetProblem`: `min_w ‖A_S w − b‖² + λ|z|` fits continuous
  coefficients; not pseudo-Boolean, so it is sealed by a convex box-QP bound
  (`certify_best_subset_gap`, via `omnibias-convex`), back-stopped by the always-valid
  full-OLS-residual floor (degrades to `method="ols_floor"` when `omnibias-convex` is absent).
- **Fork C** -- `certified_sparse_fit`: seals the pseudo-Boolean **surrogate** and ships an
  OLS refit on the decoded support (`SparseFitResult`).

See the runnable [certified sparse recovery example](../examples/certified_sparse_recovery.py).

::: omnibias.discrete.sparse.problem
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.discrete.sparse.frontends
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.discrete.sparse.certify
    options:
      show_root_heading: false
      heading_level: 3

### Sparse `l_p` relaxation (JAX + torch)

::: omnibias.discrete.sparse.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 4

::: omnibias.discrete.sparse.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 4

Status: Alpha (`0.1.0a1`).

# omnibias-combinatorics

Exact **differentiable** matching / flow / matroid layers on the omnibias stack, with a
**tight** rigorous optimality-gap certificate: entropic (Sinkhorn) relaxations onto the
assignment / transportation / flow / matroid polytopes, a decode to a vertex, and the
LP-dual lower bound.

Assignment, transportation, min-cost-flow and matroid optimization are in `P` (Hungarian
/ LP / max-flow / greedy solve them exactly), so this is **not** a `P = NP` claim. The
honest limit is narrower -- the exact combinatorial argmin has an a.e.-zero gradient, so
it cannot be trained through. The sound "differentiable combinatorial layer" is a
**yes-if**:

\[
\underbrace{\ell}_{\text{certified LP lower bound}} \;\le\; \text{optimum} \;\le\;
\underbrace{c^\top v}_{\text{decoded vertex (upper bound)}} .
\]

1. A **differentiable entropic / Sinkhorn relaxation** onto the problem's polytope; as
   `beta -> inf` it collapses onto a vertex. Unrolled for backprop, so a cost / weight
   model trains *through* it. The Sinkhorn / soft-top-k kernels are reused from
   `omnibias-graph`.
2. A **decode to a vertex** (`decode`), the upper bound; `classical_optimum` (Hungarian /
   LP / max-flow / greedy) is the exact best-in-class baseline.
3. A **tight LP-dual certificate**: the LP dual, sealed by the Neumaier-Shcherbina
   verified bound (`omnibias.core.verified` via `omnibias-convex`), is the lower bound.
   The gap is tight (`~0`) because these polytopes are **integral**, never asserted zero.

A runnable walkthrough is in
[`docs/examples/certified_differentiable_matching.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/certified_differentiable_matching.py).

## Problem & certificate containers

::: omnibias.combinatorics.problem
    options:
      show_root_heading: false
      heading_level: 3

## Polytope LP systems (numpy)

::: omnibias.combinatorics._core.polytopes
    options:
      show_root_heading: false
      heading_level: 3

## Matroids (numpy)

::: omnibias.combinatorics._core.matroids
    options:
      show_root_heading: false
      heading_level: 3

## Decoders & classical oracles (numpy)

::: omnibias.combinatorics._core.decode
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable relaxation layers (JAX)

::: omnibias.combinatorics.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Backend twins (torch)

Bit-identical PyTorch twins of the relaxation layers (float64).

::: omnibias.combinatorics.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Optimality-gap certificate

::: omnibias.combinatorics.certify
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).

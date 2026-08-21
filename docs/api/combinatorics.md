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

## Unsplittable-flow cost separation (DGG)

Exact `Fraction` replay of the AFP / Rybin H* instance and a blind
parameter-box search. A capped 3-terminal DAG family (≤6 vertices) uses
the same predicate; a CI miss is `BLOCKED`. The congestion theorem stays
true; a separator is not a claim that omnibias refuted Goemans.
Cookbooks: [Unsplittable-flow cost separation](../cookbook/unsplittable-flow.md),
[Finite discovery engine](../cookbook/discovery-loop.md).

::: omnibias.combinatorics.unsplittable
    options:
      show_root_heading: false
      heading_level: 3

## Finite Ramsey colouring

Triangle-free colourings, a tiny `IsSaturated` smoke, and a blind
`K_5` 2-colouring search (`ramsey_colouring_search`). Not Erdős 183.
Cookbooks: [Finite Ramsey colouring](../cookbook/finite-ramsey-colouring.md),
[Finite discovery engine](../cookbook/discovery-loop.md).

::: omnibias.combinatorics.ramsey
    options:
      show_root_heading: false
      heading_level: 3

## Extremal graph templates

`C4` / `C6` / `jTemplate` / `kTemplate` / `pairGraph(4,2)` structural
checks. Not Erdős 146 / 180.
Cookbook: [Extremal graph templates](../cookbook/extremal-graph-replay.md).

::: omnibias.combinatorics.extremal
    options:
      show_root_heading: false
      heading_level: 3

## Named minors (`n≤5`)

`condition_forbidden_minor` searches named `K2` / `K3` / `C4` minors of
a host with at most five vertices. Finite branch-set test; not
Robertson–Seymour and not Erdős 146 / 180.

::: omnibias.combinatorics.minors
    options:
      show_root_heading: false
      heading_level: 3

`condition_edge_colouring` and `condition_extremal_template` bind on a
packed graph (K5 colourings; C4-like templates). Erdős parent keys stay
False.

::: omnibias.combinatorics.conditions
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).

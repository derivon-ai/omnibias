# omnibias-routing

Certified **and** differentiable combinatorial **routing** on the omnibias stack: a
poly-size TSP relaxation (assignment / single-commodity-flow / Held-Karp) solved
differentiably by the temperature-collapse penalty, a heatmap + 2-opt tour decoder, and a
rigorous optimality-gap certificate.

The travelling-salesman tour is NP-hard, so there is no poly-time differentiable map
to the *exact* optimal tour (that would imply `P = NP`, and the exact argmin's
gradient is a.e. zero). The sound "differentiable TSP" is therefore a three-part
object -- a **yes-if**, not a no-because:

\[
\underbrace{\ell}_{\text{certified LP lower bound}} \;\le\; \text{optimum} \;\le\;
\underbrace{c^\top \pi}_{\text{decoded tour (upper bound)}} .
\]

1. A **differentiable convex relaxation** over a poly-size TSP polytope, solved by the
   `omnibias-convex` temperature-collapse penalty *unrolled* for backprop -- so a cost model
   trains *through* the optimizer (decision-focused routing).
2. A **heuristic decoder** (nearest-neighbour + 2-opt / or-opt) that rounds the
   fractional arc-use to a valid tour -- the upper bound.
3. A **rigorous optimality-gap certificate**: the relaxation's LP dual, sealed by the
   Neumaier-Shcherbina verified bound (`omnibias.core.verified`), is the lower bound.
   The gap `tour_cost - lower` is certified, never asserted zero; a weaker relaxation
   only widens it.

A worked, runnable walkthrough is in the
[certified-differentiable-routing cookbook](../cookbook/certified-differentiable-routing.md)
and
[`docs/examples/decision_focused_routing_tsp.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/decision_focused_routing_tsp.py).

## Problem & certificate containers

::: omnibias.routing.problem
    options:
      show_root_heading: false
      heading_level: 3

## Decoder & exact oracle (numpy)

::: omnibias.routing._core.decode
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable relaxation layers (JAX)

::: omnibias.routing.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Decision-focused routing (JAX)

::: omnibias.routing.jax.decision_focused
    options:
      show_root_heading: false
      heading_level: 3

## Decision-focused metrics (numpy)

::: omnibias.routing._core.decision
    options:
      show_root_heading: false
      heading_level: 3

## Backend twins (torch)

Bit-identical PyTorch twins of the relaxation layers and the decision loss
(parity `~1e-12`, float64).

::: omnibias.routing.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.routing.torch.decision_focused
    options:
      show_root_heading: false
      heading_level: 3

## Optimality-gap certificate

::: omnibias.routing.certify
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).

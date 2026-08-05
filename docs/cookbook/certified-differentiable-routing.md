# Certified differentiable routing (a TSP you can train through, with a proof of how good it is)

`omnibias-routing` builds the "differentiable TSP" in the only sound sense: a
**differentiable convex relaxation** you can backprop through, a **decoder** that
returns a valid tour, and a **rigorous optimality-gap certificate** that bounds how
far that tour is from optimal. The travelling-salesman tour is NP-hard, so this is a
**yes-if** -- yes, end-to-end learned routing, *if* you accept a certified gap in
place of an exactness claim (an exact poly-time differentiable optimum would imply
`P = NP`). This cookbook builds it end to end, following
[`docs/examples/decision_focused_routing_tsp.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/decision_focused_routing_tsp.py).

## The three-part object

\[
\underbrace{\ell}_{\text{certified LP lower bound}} \;\le\; \text{optimum} \;\le\;
\underbrace{c^\top \pi}_{\text{decoded tour (upper bound)}} .
\]

The certified **gap** `c^T pi - l` is the deployment-time suboptimality you can
actually promise -- never asserted to be zero.

## 1. A differentiable relaxation

Each relaxation is a poly-size LP over a TSP polytope, with `x_ij in [0, 1]` the
directed arc-use. Three strengths, increasing tightness:

- `assignment` -- degree-1 in/out only (loosest; may contain subtours), `E` vars;
- `flow` -- single-commodity flow (Gavish-Graves), subtour-free, `2E` vars;
- `held_karp` -- multicommodity flow (the Held-Karp / subtour LP), tightest, `E*n` vars.

They are solved by the `omnibias-convex` temperature-collapse penalty -- degree equalities as
a quadratic penalty `mu/2 ||A_eq x - b_eq||^2`, box / coupling as the hard hinge
`mu/2 relu(A x - b)^2`, plus a small regulariser `reg` for strong convexity -- minimised
by accelerated (Nesterov) gradient descent along a `mu` homotopy. Because the whole
solve is a fixed, **unrolled** matmul loop, it is `jit`-able and **differentiable**,
with a bit-identical torch twin:

```python
import jax
import jax.numpy as jnp
from omnibias.routing import RoutingProblem, decode_tour
from omnibias.routing.jax import flow_relaxation   # or assignment_relaxation / held_karp_layer

jax.config.update("jax_enable_x64", True)   # the relaxation solves in float64

coords = jax.random.uniform(jax.random.PRNGKey(0), (7, 2))   # 7 cities in the unit square

prob = RoutingProblem.from_coords(coords)      # (n, 2) Euclidean, or pass any cost matrix
heat = flow_relaxation(prob.cost)              # (n, n) fractional arc-use -- grad-friendly
```

## 2. Decode a valid tour

The fractional heatmap is rounded to a valid Hamiltonian cycle by nearest-neighbour
construction (following high arc-use) refined by 2-opt / or-opt local search -- correct
for asymmetric costs, since every move re-evaluates the full directed tour cost:

```python
tour, tour_cost = decode_tour(prob.cost, heat=heat)   # valid tour + cost (upper bound)
```

## 3. Certify the gap

The relaxation's LP dual is a rigorous *lower* bound on the true optimum (any relaxation
contains every integer tour). `certify_tour_gap` solves the relaxation LP exactly
(scipy HiGHS), seals the dual with the **Neumaier-Shcherbina** verified interval bound
(`omnibias.core.verified`, outward-rounded -- so it survives floating-point error), and
returns the certified sandwich:

```python
from omnibias.routing import certify_tour_gap, held_karp_dp

cert = certify_tour_gap(prob, tour, kind="held_karp")
print(cert.lower_bound, cert.tour_cost, cert.relative_gap, cert.certified)
# on small n, the exact optimum sits inside the sandwich as a self-check
# (with a float-noise tolerance: on a tight instance all three coincide):
_, opt = held_karp_dp(prob.cost)
assert cert.lower_bound - 1e-9 <= opt <= cert.tour_cost + 1e-9
```

A weaker relaxation only *widens* the certified gap -- it never invalidates the
sandwich. On a random 7-city instance the example prints:

| relaxation | lower | optimum | tour | rel gap |
| --- | --- | --- | --- | --- |
| `assignment` | 1.7786 | 2.0899 | 2.0899 | 17.5% |
| `flow` | 2.0380 | 2.0899 | 2.0899 | 2.5% |
| `held_karp` | 2.0899 | 2.0899 | 2.0899 | 0.0% |

(Without the `convex` extra the bound degrades gracefully to the valid *float* LP
value with `certified=False` -- still sound, just not interval-sealed.)

## Decision-focused routing: train the cost model *through* the optimizer

The payoff of differentiability is **decision-focused learning**. When per-arc costs
are predicted from features by a (misspecified) model, a two-stage fit minimises cost
MSE -- which is *not* the same as making good routing decisions. `decision_cost`
minimises the true cost of the *relaxed decision*, backpropagating through the
relaxation:

<!-- docs-test: skip reason="decision-focused loss over the reader's own features / cost_true / params" -->
```python
import jax
from omnibias.routing import RelaxationSchedule, normalized_regret
from omnibias.routing.jax import decision_cost

def loss(params):
    cost_pred = jax.nn.softplus(features @ params["w"] + params["b"]) + eps
    return decision_cost(cost_pred, cost_true, kind="assignment",
                         schedule=RelaxationSchedule.fast())

# ... SGD on `loss` ...
regret = normalized_regret(predicted_costs, true_costs)   # exact-oracle metric (small n)
```

On the synthetic misspecified-linear-head task the example reports normalised test
regret `ours 0.022` vs `two-stage 0.042` (about `2x` lower) -- the same
decision-focused win as the shortest-path SPO benchmark, now for routing. The
`assignment` relaxation is the best-conditioned training layer; `flow` / `held_karp`
are available when you want a tighter (costlier) inner solve.

## Honest scope

- The certificate is a **genuine** rigorous lower bound on the true tour cost for the
  given cost matrix (Held-Karp LP dual, Neumaier-Shcherbina) plus the decoded tour
  upper bound. It is **not** an exact-optimality (`P = NP`) claim and never asserts a
  zero gap.
- **Relaxation strength is a design choice** (`assignment < flow < held_karp`); the
  certified gap is honest, and a weaker relaxation simply reports a wider one.
- For *predicted* costs, the gap is the honest deployment-time suboptimality of the
  decision -- not model-relative, not an exactness claim.
- `held_karp` is the dense `O(n^3)`-variable path -- correct for small `n` / single
  instances; a structured / matrix-free operator is the staged scalability follow-up
  (see the [certified combinatorial-optimization boundary](graph-limitation.md)). The
  exact DP oracle is capped at `n <= 18`.

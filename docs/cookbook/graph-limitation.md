# Certified differentiable combinatorial optimization: relaxations, certificates, and the one honest limit

!!! success "Yes-if, not no-because"
    You *can* learn to solve combinatorial graph / routing problems end to end with
    omnibias -- **if** you accept a **certified optimality gap** in place of an
    exactness claim. The differentiable spectral / Sinkhorn / SoftSort machinery in
    `omnibias-graph`, and the full **certified differentiable routing** stack in
    [`omnibias-routing`](../api/routing.md), turn a discrete problem into a
    *differentiable relaxation + a decoder + a rigorous bound*. There is exactly
    **one** honest hard limit, stated below; everything up to it is supported.

A natural goal: write a balanced cut (or a matching, an ordering, a tour) as a
discrete optimization, relax the indicators to `[0, 1]`, follow the gradient, and
read off a solution. omnibias provides exactly these relaxations -- and, for
routing, wraps them into a **certified** predict-then-optimize pipeline. What you
get is not the exact combinatorial optimum for free (that is the one limit), but a
valid solution **with a rigorous certificate of how close to optimal it is**.

## The one honest limit (kept)

There is a single boundary, and it is a theorem, not a scope decision:

- **No poly-time differentiable map returns the *exact* optimal tour / cut / SAT
  assignment.** If a polynomial-size, polynomial-time differentiable relaxation
  reliably returned the exact optimum of an NP-hard problem, composing it with the
  poly-time value check would collapse that problem into `BPP`. No such collapse is
  known or expected (it would imply `P = NP`).
- **The exact argmin is not a useful learning signal anyway.** The map
  `costs -> exact optimal tour` is piecewise constant, so its gradient is zero
  almost everywhere and undefined on the measure-zero switching set. "Backprop
  through the exact optimum" is ill-posed regardless of hardness.

So the *only* thing that is impossible is a poly-time, differentiable, **exact**
combinatorial oracle. That is a narrow, precise statement -- not a wall around the
whole problem.

## What is supported (the yes-if)

Everything short of that limit:

| You want... | omnibias gives you | The honest guarantee |
| --- | --- | --- |
| A differentiable tour / routing layer | `omnibias.routing` relaxation (`assignment` / `flow` / `held_karp`) + decoder | A valid tour **plus** a rigorous `lower <= optimum <= tour_cost` certificate |
| To train a cost model *through* the optimizer | unrolled temperature-collapse penalty (`decision_cost`) | Lower decision **regret** than a two-stage fit; exact gradients |
| Balanced min `k`-cut | `spectral_clustering_relaxation` | A *certified lower bound* (sum of `k` smallest eigenvalues) + an embedding to round |
| Optimal assignment / matching | `gumbel_sinkhorn` | Doubly-stochastic transport; a permutation in the `tau -> 0` limit |
| Optimal ordering / top-`k` | `soft_sort` / `soft_top_k` | Row-stochastic / box-feasible soft object; hard only as `tau -> 0` |

Each of these is a **relaxation**: the feasible set is enlarged from the discrete
vertices (permutation matrices, `{0,1}` indicators) to their convex hull (the
Birkhoff polytope, the box). The relaxation optimum is a rigorous *bound* on the
discrete optimum -- and that bound is exactly what makes the certificate honest.

## How `omnibias-routing` closes the loop

The travelling-salesman example, end to end (see the
[certified differentiable routing cookbook](certified-differentiable-routing.md)):

1. **Differentiable relaxation.** A poly-size LP over a TSP polytope
   (`assignment` &sub; `flow` &sub; `held_karp`, increasing tightness) solved by the
   `omnibias-convex` temperature-collapse penalty, *unrolled* for backprop.
2. **Decoder.** Nearest-neighbour + 2-opt / or-opt rounds the fractional arc-use to
   a **valid tour** -- an *upper* bound on the optimum.
3. **Certificate.** The relaxation's LP dual, sealed by the Neumaier-Shcherbina
   verified interval bound, is a rigorous *lower* bound. Then
   `lower <= optimum <= tour_cost` is a **certified optimality gap** -- never
   asserted to be zero, and honest about relaxation strength (a weaker relaxation
   only *widens* the gap; the sandwich stays valid).

The gap is the deployment-time suboptimality you can actually promise. On small
instances the exact `held_karp_dp` optimum sits inside the sandwich as a self-check.

## Why the gap, not zero

Two design facts keep the story honest:

1. **Rounding is where the hardness lives.** A relaxation is solved in polynomial
   time and gives a bound; recovering the *discrete* optimum from the fractional
   point is the NP-hard step. omnibias does the polynomial part exactly and reports
   the residual as a certified gap rather than hiding it.
2. **The temperature / penalty trade-off is real.** A relaxation recovers its
   discrete object only in a sharp limit (`tau -> 0`, or `mu -> infinity`), which is
   exactly where gradients degenerate. omnibias trains in the smooth regime and
   *certifies* the gap, instead of pretending the limit is free.

## Where the differentiable view *also* helps

- **Spectral embeddings / diffusion.** `graph_laplacian`, `normalized_laplacian`,
  `spectral_embedding`, and `graph_heat_kernel` are exact linear algebra --
  differentiable features for downstream learning, with the ring-graph spectrum and
  SBM Fiedler vector as analytic oracles.
- **Certified spectral bounds.** The ring-graph eigenvalues are *certified* with an
  `omnibias.core.verified` interval enclosure that brackets a true eigenvalue (a
  theorem, not a sample).
- **Learnable matching / ordering front-ends.** `gumbel_sinkhorn` / `soft_sort` /
  `soft_top_k` give end-to-end gradients through a *relaxed* permutation or
  selection that a decoder can round and a certificate can bound.

## Scaling note (staged)

The `held_karp` (multicommodity-flow) relaxation is the tightest but has `O(n^3)`
variables; omnibias ships the **dense** path first, which is correct for small `n` /
single instances. A structured / matrix-free operator so the flow LP never
materialises a dense constraint matrix is the planned scalability follow-up -- the
small-`n` package is already sound without it.

## What genuinely stays out of scope

`omnibias-graph` itself ships **only** relaxations and exact spectral linear algebra
-- never an exact NP-hard solver (guarded by
`packages/omnibias-graph/tests/test_audit_limits.py`); certified *routing* lives in
`omnibias-routing`. And the discrete-hardness problems with **no** analytic /
relaxation handle -- integer **factoring**, **discrete log**, **RSA/ECC** key
recovery, primality proving, and general **SAT / ILP** decision solving -- remain
correctly out of scope (see the [RSA-limitation cookbook](rsa-limitation.md)). The
difference: routing has a certified relaxation gap to report; factoring does not.

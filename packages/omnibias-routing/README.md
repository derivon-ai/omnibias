# omnibias-routing

**Status: Alpha (0.1.0a1).**

Certified **and** differentiable combinatorial **routing** for omnibias. The
travelling-salesman tour is NP-hard, so there is no poly-time differentiable map to
the *exact* optimal tour (that would imply `P = NP`, and the exact argmin's gradient
is a.e. zero). Instead of answering *"can you differentiate the exact TSP?"* with a
flat **no-because**, this package answers the well-posed question with a **yes-if**:

> **Yes** you can learn to route end to end **if** you accept a *certified optimality
> gap* in place of an exactness claim.

Concretely, the "differentiable TSP" is a three-part object:

\[
\underbrace{c(\theta)}_{\text{cost head}}\;\longrightarrow\;
\underbrace{x^\star = \text{relax}(c)}_{\text{differentiable LP relaxation}}\;\longrightarrow\;
\underbrace{\pi = \text{decode}(x^\star)}_{\text{valid tour (upper bound)}},\qquad
\underbrace{\ell \le \text{opt} \le c^\top\pi}_{\text{certified gap}}.
\]

1. **Differentiable convex relaxation.** A poly-size LP over a TSP polytope
   (`assignment` / single-commodity-`flow` / `held_karp`), solved by the same
   temperature-collapse penalty as [omnibias-convex](../omnibias-convex), *unrolled* and
   batched for backprop -- so a cost model trains *through* the optimizer.
2. **Heuristic decoder.** Nearest-neighbour + 2-opt / or-opt rounds the fractional
   arc-use to a valid tour -- an *upper* bound on the optimum.
3. **Rigorous optimality-gap certificate.** The Neumaier-Shcherbina verified LP dual
   bound is a *lower* bound on the true optimum, so `lower <= opt <= tour_cost` is a
   **certified gap** -- never asserted zero, and honest about relaxation strength (a
   weaker relaxation only widens the gap).

Prior neural TSP layers give a heatmap but no certificate; exact solvers give an
optimum but no gradient. The differentiable-*and*-certified combination is what this
package offers.

## What's in the box

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)     # the certificates are float64

from omnibias.routing import (
    RoutingProblem, decode_tour, certify_tour_gap, held_karp_dp, normalized_regret,
)
from omnibias.routing.jax import (
    assignment_relaxation,  # differentiable degree-constrained relaxation (best conditioned)
    flow_relaxation,        # subtour-free single-commodity flow
    held_karp_layer,        # tightest (multicommodity flow), small n
    decision_cost,          # differentiable "smart predict-then-optimize" loss
)

coords = jnp.array([[0.0, 0.0], [1.0, 0.2], [0.9, 1.1],
                    [0.1, 1.0], [0.5, 0.5], [1.4, 0.6]])
prob = RoutingProblem.from_coords(coords)         # (n, 2) Euclidean, or pass a cost matrix
heat = flow_relaxation(prob.cost)                 # (n, n) fractional arc-use, grad-friendly
tour, cost = decode_tour(prob.cost, heat=heat)    # valid tour + its cost (upper bound)
cert = certify_tour_gap(prob, tour, kind="flow")  # lower <= opt <= tour_cost
print(cert.lower_bound, cert.tour_cost, cert.relative_gap, cert.certified)
```

Every backend function has a bit-identical `omnibias.routing.torch` twin (parity
`~1e-12`, float64).

- **`assignment_relaxation` / `flow_relaxation` / `held_karp_layer`** -- differentiable
  relaxations `cost (…, n, n) -> arc-use (…, n, n)`; `RelaxationSchedule()` is
  eval-quality, `RelaxationSchedule.fast()` is enough to train through.
- **`decode_tour(cost, *, heat=None)`** -- NN + 2-opt/or-opt to a valid tour (correct
  for asymmetric costs); follows `heat` when given.
- **`certify_tour_gap(problem, tour, kind=...)`** -- a rigorous `HeldKarpCertificate`
  (`certified=True`, interval-sealed); `.absolute_gap`, `.relative_gap`, `.is_sound`.
- **`held_karp_dp(cost)`** -- the exact `O(2^n n^2)` optimum (small `n`), to self-check
  the sandwich and score `normalized_regret` for decision-focused learning.
- **`decision_cost` / `normalized_regret` / `spo_plus_gradient`** -- the differentiable
  decision loss and the predict-then-optimize baselines/metrics.

## Honest scope

- The certificate is a **genuine** rigorous lower bound on the true tour cost for the
  given cost matrix (Held-Karp LP dual, Neumaier-Shcherbina, outward-rounded intervals)
  plus the decoded tour upper bound. It is **not** an exact-optimality (`P = NP`) claim
  and never asserts a zero gap.
- **Relaxation strength is a design choice**: `assignment < flow < held_karp`; a weaker
  relaxation is still sound, it just reports a wider certified gap.
- `held_karp_layer` is the dense `O(n^3)`-variable path -- correct for small `n` /
  single instances; a matrix-free operator is the staged scalability follow-up (see the
  cookbook). The exact DP oracle is capped at `n <= 18`.
- The relaxation layers need a `jax` / `torch` backend; the certificate needs `scipy`
  (core) for the exact LP solve and the `convex` extra for the rigorous interval seal
  (without `convex` it returns the valid float LP bound with `certified=False`).
- Field-level math is torch + jax only (repo convention). Extension-tier typing (authored
  strict-clean; not on the shared strict CI gate).

## Tests

```bash
pip install -e "packages/omnibias-routing[convex,jax,torch,graph,verify,test]"
python -m pytest packages/omnibias-routing/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.

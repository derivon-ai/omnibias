# omnibias-submodular

**Status: Alpha (0.1.0a1).**

Differentiable **and** certified **monotone submodular maximization** on the omnibias
stack. Maximizing a monotone submodular `f` over the independent sets of a matroid
(cardinality / partition) is NP-hard, so there is no poly-time map to the *exact*
optimum (that would imply `P = NP`). Instead of a flat **no-because**, this package
answers the well-posed question with a **yes-if**:

> **Yes** you can maximize / learn through a submodular objective end to end **if** you
> accept a certified `(1 - 1/e)` approximation plus an optimality-gap sandwich in place
> of an exactness claim.

The engine is the **multilinear extension**: replace `x ∈ {0,1}ⁿ` by
`p ∈ [0,1]ⁿ` with `F(p) = E_{x∼p}[f(x)]` (the unique multilinear polynomial that
interpolates `f`). **Continuous greedy** (Frank-Wolfe over the matroid polytope)
produces a fractional `p*` with `F(p*) ≥ (1 - 1/e) OPT`; **pipage / swap rounding**
returns an integral independent set `S` with `f(S) ≥ F(p*)`, hence the certified
`f(S) ≥ (1 - 1/e) OPT`.

```
p* = continuous_greedy(f, matroid)   ->   S = round(p*)      (feasible set, lower bound on OPT)
                                          f(S) <= OPT <= U(S)   (certified gap; f(S) >= (1-1/e) OPT)
```

## The two "collapse" axes (honesty)

The multilinear extension relaxes `{0,1}ⁿ → [0,1]ⁿ` and the Frank-Wolfe LP oracle
`sigmoid(beta·(g - tau))`, `beta → ∞`, hardens onto a `0/1` matroid-basis vertex --
the **feasibility / temperature** sense of "collapse" (a soft indicator hardening to a
0/1 step), distinct from the **founding bias collapse** -- the multi-bias `delta → 0`
limit of an `OMBU` to the closed-form derivative `sigma^(K-1)` (see `docs/theory.md`).

## What's in the box

```python
import numpy as np
from omnibias.submodular import (
    UniformMatroid, max_coverage, maximize, greedy_maximize,
    brute_force_max, certify_submodular_gap,
)

# 4 candidate sets over a 6-element universe; pick k = 2 to maximize coverage.
sets = [{0, 1, 2}, {2, 3}, {3, 4, 5}, {0, 5}]
prob = max_coverage(sets, universe=6, k=2)
sol = maximize(prob, rounding="pipage")          # continuous greedy -> pipage rounding
cert = certify_submodular_gap(prob, sol.selection, fractional=sol.fractional)
print(sol.selection, sol.value, cert.certified_ratio, cert.approx_ratio)
```

- **`SubmodularFunction`** -- `Coverage` (weighted max-coverage, the headline),
  `FacilityLocation`, and `BudgetAdditive` (concave-of-modular), each with an exact
  closed-form multilinear extension `F(p)`, its gradient, and `to_polynomial()`.
- **`Matroid`** -- `UniformMatroid(n, k)` (cardinality) and `PartitionMatroid(groups,
  caps)`; a `max_weight_basis` LP oracle plus a differentiable soft basis.
- **`maximize(problem, *, rounding=...)`** -- continuous greedy + `pipage` / `swap`
  rounding + a feasibility-preserving polish; returns a `SubmodularSolution`.
- **`greedy_maximize` / `brute_force_max`** -- the classical greedy baseline and the
  exact `O(2ⁿ)` matroid-constrained oracle (small `n`, self-checks the sandwich).
- **`certify_submodular_gap(problem, S, *, fractional=...)`** -- a
  `SubmodularCertificate` (`value`, `upper_bound`, `fractional_value`,
  `approx_ratio = 1 - 1/e`, `.certified_ratio`, `.absolute_gap`, `.is_sound`).
- **`submodular_relaxation(problem, schedule=None)`** -- the differentiable
  continuous-greedy relaxation for the coverage family; bit-identical
  `omnibias.submodular.jax` / `omnibias.submodular.torch` twins (parity `~1e-9`,
  float64), unrolled so a model predicting the coverage weights trains *through* it.

## Honest scope

- The `(1 - 1/e)` guarantee is the classical continuous-greedy + rounding result for
  **monotone** submodular `f` under a matroid constraint; it is an a-priori theorem the
  algorithm carries, self-checked on small `n` against `brute_force_max`.
- The runtime certificate is a **sound** sandwich `f(S) ≤ OPT ≤ U(S)` where `U(S)` is a
  data-dependent marginal-gain upper bound; a weaker `U` only *widens* the certified
  gap and is never unsound. The certificate is **not** an exact-optimality (`P = NP`)
  claim.
- The differentiable relaxation twins currently cover the **coverage family**; every
  shipped function runs the full numpy certified pipeline. The relaxation needs a
  `jax` / `torch` backend. Extension-tier typing (authored strict-clean; not on the
  shared strict CI gate).
- Builds on `omnibias-discrete`: `SubmodularProblem` implements the `DiscreteProblem`
  seam (`energy = -f`), so the substrate `brute_force_min` / `certify_gap` also apply
  to the (for monotone `f`, trivial) *unconstrained* view.

## Tests

```bash
pip install -e "packages/omnibias-submodular[sos,jax,torch,test]"
python -m pytest packages/omnibias-submodular/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.

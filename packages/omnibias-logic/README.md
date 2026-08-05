# omnibias-logic

**Status: Alpha (0.1.0a1).**

Differentiable **and** certified **Boolean logic** on the omnibias stack. Two classic
NP-hard / #P-hard problems on a CNF, each answered as a **yes-if** (a certified object,
never a `P = NP` / `#P` exactness claim):

- **weighted MaxSAT** -- *re-exported unchanged* from `omnibias.discrete.maxsat` (the
  weighted-violation `DiscreteProblem` on the `omnibias-discrete` substrate): an annealed
  sigmoid relaxation, a rounding + 1-flip decoder, and a Lasserre / SOS **certified
  optimality gap** (`lower <= optimum <= energy`);
- **(weighted) #SAT / model counting** -- *new here*: a rigorous **count enclosure**
  `lower <= #models <= upper` from truncated inclusion-exclusion (Bonferroni) in exact
  arithmetic, plus a `beta -> inf` annealed **model-finder** relaxation (torch + jax
  twins) whose decoded models are sound witness lower bounds.

> Exact MaxSAT is NP-hard and exact (weighted) #SAT is #P-hard, so there is no poly-time
> exact solver here (that would imply `P = NP`). The deliverables are a *certified
> optimality gap* and a *certified count enclosure* -- sound sandwiches, tightened (never
> faked) by more work.

## The two "collapse" axes (honesty)

Every relaxation's `sigmoid(beta·)`, `beta -> inf` is the **feasibility / temperature**
sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct from the
**founding bias collapse** -- the multi-bias `delta -> 0` limit of an `OMBU` to the
closed-form derivative `sigma^(K-1)` (see `docs/theory.md`). This package lives entirely
on the temperature axis.

## What's in the box

<!-- docs-test: slow -->
```python
import numpy as np
from omnibias.logic import (
    max_sat, decode, brute_force_min, certify_gap,   # re-exported MaxSAT surface
    model_count, count_enclosure, exact_model_count,  # new #SAT surface
)

# --- weighted MaxSAT: a certified optimality gap ---------------------------------
prob = max_sat([[1, -2], [2, 3], [-1, -3]], weights=[1.5, 2.0, 0.5])
z, energy = decode(prob, n_starts=16)
cert = certify_gap(prob, z, level=2)     # lower <= optimum <= energy
print(cert.lower_bound, cert.energy, cert.is_sound)

# --- (weighted) #SAT: a certified count enclosure --------------------------------
mc = model_count([[1, -2], [2, 3], [-1, -3]])         # unweighted model counting
enc = count_enclosure(mc, order=2)                     # Bonferroni order-2 sandwich
exact = exact_model_count(mc)                          # O(2^n) oracle (small n)
print(enc.lower, exact, enc.upper, enc.is_sound)       # lower <= exact <= upper
```

- **`max_sat` / `MaxSATProblem` / `WeightedCNF` / `Clause`** -- the re-exported weighted
  MaxSAT front-end (DIMACS signed 1-based literals). `decode` / `brute_force_min` /
  `certify_gap` / `GapCertificate` / `AnnealSchedule` come straight from the substrate.
- **`omnibias.logic.torch.maxsat_relaxation` / `omnibias.logic.jax.maxsat_relaxation`**
  -- the re-exported differentiable MaxSAT relaxation twins.
- **`model_count(clauses, weights=None)` -> `ModelCountProblem`** -- a #SAT / weighted
  model-counting instance (per-variable literal `weights`, unweighted by default);
  implements the `DiscreteProblem` seam so the substrate decoder / oracle work on it.
- **`count_enclosure(problem, *, order=...) -> CountCertificate`** -- the rigorous
  `lower <= #models <= upper` inclusion-exclusion enclosure (tightens to exact at
  `order = #clauses`).
- **`exact_model_count(problem)`** -- the exact `O(2^n)` count (built on
  `omnibias.boolean`), the small-`n` oracle that self-checks the enclosure.
- **`omnibias.logic.torch.sat_relaxation` / `omnibias.logic.jax.sat_relaxation`** -- the
  `beta -> inf` annealed model-finder relaxation (bit-identical twins).

## Honest scope

- The count enclosure is a **genuine** rigorous sandwich of the true (weighted) model
  count, exact-arithmetic sound; a lower `order` only *widens* it, never unsound.
- `exact_model_count` / `brute_force_min` are the exact **exponential** (`O(2^n)`)
  oracles for small `n`, not solvers.
- The relaxation layers need a `torch` / `jax` backend; the enclosure is pure Python.
  Extension-tier typing (authored strict-clean; not on the shared strict CI gate).

## Tests

```bash
pip install -e "packages/omnibias-logic[sos,torch,jax,test]"
python -m pytest packages/omnibias-logic/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.

# omnibias-discrete

**Status: Alpha (0.1.0a1).**

The shared **differentiable + certified discrete-optimization substrate** for omnibias.
It is the `encode -> relax -> decode -> certify` machinery that was first shipped inside
`omnibias-qubo` and then extracted so a second consumer (MaxSAT, here; combinatorics /
logic, later) can reuse it. `omnibias-qubo` now builds on this package via re-export
shims -- its public API and numerics are unchanged.

Minimizing a pseudo-Boolean energy `E(x)` over `x ∈ {0,1}ⁿ` is NP-hard, so there is no
poly-time differentiable map to the *exact* global optimum (that would imply `P = NP`,
and the exact argmin's gradient is a.e. zero). Instead of a flat **no-because**, the
substrate answers the well-posed question with a **yes-if**: you may optimize / learn
through a discrete problem end to end **if** you accept a *certified optimality gap* in
place of an exactness claim.

```
theta -> x* = relax(problem)  ->  z = decode(problem, x*)     (valid binary point, upper bound)
                                  lower <= optimum <= E(z)      (certified gap)
```

## The seam: `DiscreteProblem`

Any object that implements the protocol plugs into the whole pipeline:

<!-- docs-test: signature -->
```python
class DiscreteProblem(Protocol):
    @property
    def n(self) -> int: ...
    def energy(self, x) -> float | FloatArray: ...            # point (n,) or batch (m, n)
    def to_polynomial(self) -> "omnibias.sos.Polynomial": ...  # for the SOS/Lasserre bound
    # optional fast path: def flip_deltas(self, x) -> FloatArray: ...
```

- **`AnnealSchedule`** -- the temperature homotopy for the relaxation.
- **`anneal_descent(grad_x_fn, scale, n, schedule)`** (`omnibias.discrete.jax` /
  `omnibias.discrete.torch`) -- the differentiable annealed relaxation core: descends
  `x = sigmoid(beta·theta)` on a caller-supplied closed-form energy gradient while
  `beta -> ∞` collapses the soft assignment onto a binary vertex. Bit-identical twins.
- **`decode` / `one_flip_descent` / `round_relaxed` / `brute_force_min`** -- rounding +
  1-flip local search (an *upper* bound), and the exact `O(2ⁿ)` small-`n` oracle.
- **`certify_gap` / `lasserre_lower_bound` / `negative_coeff_lower_bound`** -- the
  rigorous *lower* side of the sandwich: a Lasserre / moment-SOS bound over the Boolean
  hypercube (`omnibias-sos`, hash-sealed / Lean-checkable), seeded by the always-valid
  sum-of-negative-coefficients bound.
- **`GapCertificate` / `DiscreteSolution`** -- the result containers.

## Terminology (the two "collapse" axes)

The relaxation's `sigmoid(beta·)`, `beta -> ∞` is the **feasibility / temperature** sense
of "collapse" (a soft indicator hardening to a 0/1 step), distinct from the **founding
bias collapse** -- the multi-bias `delta -> 0` limit of an `OMBU` to the closed-form
derivative `sigma^(K-1)` (see `docs/theory.md`).

## MaxSAT front-end (first consumer)

`omnibias.discrete.maxsat` encodes weighted CNF as a pseudo-Boolean `MaxSATProblem`
(the weighted-violation energy) and reuses the shared relax / decode / certify:

```python
from omnibias.discrete.maxsat import max_sat
from omnibias.discrete import decode, certify_gap

# (x1 or ~x2) and (x2 or x3), unit weights; DIMACS-style signed 1-based literals.
prob = max_sat([[1, -2], [2, 3]])
assignment, energy = decode(prob)          # a min-violation assignment (upper bound)
cert = certify_gap(prob, assignment, level=1)
print(cert.lower_bound, cert.energy, cert.relative_gap, cert.certified)
```

## Honest scope

- The certificate is a **genuine** rigorous lower bound on the true minimum energy, plus
  the decoded upper bound. It is **not** an exact-optimality (`P = NP`) claim and never
  asserts a zero gap; a weaker bound only *widens* the certified gap.
- `certify_gap` needs the `sos` extra for the sealed Lasserre bound; without it it
  degrades to the valid, non-sealed float bound (`certified=False`).
- The relaxation layers need a `jax` / `torch` backend. Extension-tier typing (authored
  strict-clean; not on the shared strict CI gate).

## Tests

```bash
pip install -e "packages/omnibias-discrete[sos,jax,torch,test]"
python -m pytest packages/omnibias-discrete/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.

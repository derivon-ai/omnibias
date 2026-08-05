# omnibias-qubo

**Status: Alpha (0.1.0a1).**

Differentiable **and** certified **quadratic Boolean optimization** (QUBO / Ising) on
the omnibias stack. Minimizing a quadratic pseudo-Boolean energy
`E(x) = xᵀ Q x + cᵀ x` over `x ∈ {0,1}ⁿ` is NP-hard, so there is no poly-time
differentiable map to the *exact* global optimum (that would imply `P = NP`, and the
exact argmin's gradient is a.e. zero). Instead of a flat **no-because**, this package
answers the well-posed question with a **yes-if**:

> **Yes** you can optimize / learn through a QUBO end to end **if** you accept a
> *certified optimality gap* in place of an exactness claim.

Concretely the "differentiable QUBO" is a three-part object:

```
theta -> x* = relax(Q, c)  ->  z = decode(x*)          (valid binary point, upper bound)
                                lower <= optimum <= E(z)   (certified gap)
```

1. **Differentiable annealed relaxation.** A soft assignment `x = sigmoid(beta·theta) ∈
   (0,1)ⁿ` is descended on the (closed-form-gradient) energy while `beta -> ∞` is
   annealed so the relaxed minimizer collapses onto a binary vertex -- the omnibias
   **temperature-collapse** penalty, *unrolled* for backprop so a model that
   predicts `Q` / `c` trains *through* the optimizer. Bit-identical torch + jax twins.
2. **Heuristic decoder + exact oracle.** `decode_qubo` rounds the soft assignment and
   refines it by 1-flip local search (an *upper* bound); `brute_force_min` is the exact
   `O(2ⁿ)` optimum (small `n`) that self-checks the certificate sandwich.
3. **Rigorous optimality-gap certificate.** `certify_qubo_gap` returns a *lower* bound
   on the true optimum, so `lower <= optimum <= E(z)` is a **certified gap** -- never
   asserted zero. Two strengths (see below).

## The two "collapse" axes (honesty)

The relaxation's `sigmoid(beta·theta)`, `beta -> ∞` is the **feasibility / temperature**
sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct from the
**founding bias collapse** -- the multi-bias `delta -> 0` limit of an `OMBU` to the
closed-form derivative `sigma^(K-1)` (see `docs/theory.md`). The energy gradient stays
closed form: `grad_x E = 2 Q x + c`, chained through the sigmoid.

## What's in the box

```python
import numpy as np
from omnibias.qubo import QUBOProblem, decode_qubo, certify_qubo_gap, max_cut
from omnibias.qubo.jax import qubo_relaxation  # or omnibias.qubo.torch

W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)  # triangle
prob = max_cut(W)                       # a QUBOProblem whose energy is -cut(x)
x_soft = qubo_relaxation(prob)          # (n,) soft assignment, grad-friendly
z, energy = decode_qubo(prob, relaxed=np.asarray(x_soft))
cert = certify_qubo_gap(prob, z, kind="sos", level=1)
print(cert.lower_bound, cert.energy, cert.relative_gap, cert.certified)
```

- **`QUBOProblem` / `IsingProblem`** -- containers with exact `{0,1}` <-> `{-1,+1}`
  conversion (`qubo_to_ising` / `ising_to_qubo`) and an `energy(x)` evaluator.
- **`qubo_relaxation(problem, schedule=None)`** -- the differentiable annealed relaxation
  (`AnnealSchedule()` is eval-quality, `AnnealSchedule.fast()` trains through);
  bit-identical `omnibias.qubo.jax` / `omnibias.qubo.torch` twins (parity `~1e-9`, float64).
- **`decode_qubo` / `one_flip_descent` / `brute_force_min`** -- rounding + local search
  (upper bound) and the exact small-`n` oracle.
- **`certify_qubo_gap(problem, x, *, kind=..., level=...)`** -- a `QUBOCertificate`
  (`lower_bound`, `energy`, `.absolute_gap`, `.relative_gap`, `.is_sound`).
- **`max_cut` / `max_independent_set`** -- canonical problem constructors.

## Honest scope

- The certificate is a **genuine** rigorous lower bound on the true minimum energy for
  the given `Q` / `c`, plus the decoded upper bound. It is **not** an exact-optimality
  (`P = NP`) claim and never asserts a zero gap.
- **Lower-bound strength is a design choice**: `kind="spectral"` (an eigenvalue-shift /
  box-QP relaxation, any `n`, rigorously interval-sealed by the `convex` extra;
  degrades to the valid float value with `certified=False` without it) is looser than
  `kind="sos"` (the Lasserre / moment-SOS bound over the Boolean hypercube via
  `omnibias-sos`, rational and hash-sealed / Lean-checkable, for small / moderate `n`).
  A looser bound only *widens* the certified gap; it is never unsound.
- The relaxation layers need a `jax` / `torch` backend. Extension-tier typing (authored
  strict-clean; not on the shared strict CI gate).

## Tests

```bash
pip install -e "packages/omnibias-qubo[sos,convex,jax,torch,test]"
python -m pytest packages/omnibias-qubo/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.

# omnibias-holonomic

**Status: Alpha (0.1.0a1).**

A D-finite / holonomic engine with **Lean-certified** hypergeometric identities -- the
capstone of the `omnibias-difference` expansion, composing the difference register, the
`omnibias-symbolic` recurrence guesser, and the `omnibias.core.proof` Lean loop.

## Capabilities

- **Ore (skew-polynomial) algebra** (`OreAlgebra`, `OrePolynomial`): the shift ring
  `R[S; sigma]` (`(S a)(n) = a(n+1)`) and the differential ring `R[D; delta]` with the
  non-commutative product `d . r = sigma(r) d + delta(r)`. Exact rational coefficients.
- **D-finite / P-recursive objects** (`DFinite`, `PRecursive`): a sequence / series plus
  its annihilating operator and initial data, generated forward exactly (stepping over
  leading-coefficient singularities). Closed under termwise **sum**, **Hadamard** product,
  and **Cauchy** product (`dfinite_add` / `dfinite_hadamard` / `dfinite_cauchy`).
- **Gosper's algorithm** (`gosper_sum`, `gosper_definite_sum`): unconditional, closed-form
  indefinite and definite hypergeometric summation with an exact rational **certificate**
  `T = R t` (returns "not summable" rather than an unsound guess).
- **Creative telescoping** (`creative_telescoping`): the annihilating recurrence of a
  parametrised sum `f(n) = sum_k F(n, k)`, guessed from an exact prefix and range-verified.
- **Lean-certified identities** (`prove_hypergeometric_identity`): classic binomial
  identities (`sum C(n,k) = 2^n`, `sum C(n,k)^2 = C(2n,n)`, `sum k C(n,k) = n 2^{n-1}`, ...)
  discharged as per-coefficient `rational_identity` obligations the omnibias Lean kernel
  checks exactly.

## Honesty labels

- **closed-form / exact**: Ore-algebra arithmetic, the Gosper certificates and definite
  sums, the D-finite closure operators (returned annihilators are exactly verified), and
  every emitted `rational_identity` certificate payload.
- **guessed (heuristic), then verified**: *which* recurrence a sum satisfies is found by
  fitting (`omnibias.symbolic.discover_recurrence`); the returned operator is re-checked
  exactly on the range. That the recurrence -- and thus a proven identity -- continues for
  **all** `n` is the holonomic-continuation (Zeilberger) claim, backed to `n_max` here.
- **theorem_prover_verified**: earned **only** on a genuine `lake build` pass of every
  obligation; never forged. With no Lean toolchain present the bridge degrades gracefully
  and the flag stays `False`.

## The founding collapse

This package operates in the discrete register founded by `omnibias-difference`: identities
are certified through finite differences and recurrences, the `delta -> 0` limit of which is
the closed-form derivative tower. Holonomic closure is the algebraic backbone that makes
those discrete objects finitely describable -- and therefore machine-checkable.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.

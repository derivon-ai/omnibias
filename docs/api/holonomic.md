# omnibias-holonomic

A D-finite / holonomic **computer-algebra** engine with **Lean-certified** hypergeometric
identities — the capstone of the `omnibias-difference` expansion, composing the difference
register, the `omnibias-qcalculus` q-primitives, the `omnibias-symbolic` recurrence guesser,
and the `omnibias.core.proof` Lean loop.

- **Ore (skew-polynomial) algebra** (`OreAlgebra`, `OrePolynomial`) — the shift ring
  `R[S; sigma]` and the differential ring `R[D; delta]` with non-commutative product
  `d . r = sigma(r) d + delta(r)`, a genuine right-Euclidean domain (`ore_divmod`, `gcrd`,
  `lclm`, `symmetric_product`), in exact rational arithmetic.
- **D-finite / P-recursive objects** (`DFinite`, `PRecursive`) — sequence/series plus
  annihilator and initial data, generated forward exactly (stepping over leading-coefficient
  singularities). Closed under termwise sum and Hadamard product **symbolically, for all
  `n`** (via `lclm` / `symmetric_product`), with the verified-ansatz path as a labelled
  fallback; Cauchy product too.
- **Gosper's algorithm** (`gosper_sum`, `gosper_definite_sum`) — unconditional, closed-form
  indefinite / definite hypergeometric summation with an exact rational certificate; refuses
  non-summable terms rather than guessing.
- **True Zeilberger + WZ** (`zeilberger`, `ZeilbergerCertificate`, `wz_pair`,
  `wz_certificate`; build inputs with `ProperTerm`, `binomial_nk`, `geometric_k`) — exact
  creative telescoping (telescoper `L` + rational cofactor `R(n, k)`) via an exact null
  space, needing no guesser and handling degenerate sums (e.g. `sum (-1)^k C(n,k) = [n=0]`)
  natively. `creative_telescoping` is kept as the fast guessed path.
- **Petkovsek's Hyper** (`hyper`, `term_ratio_annihilates`) — all hypergeometric-term
  solutions of a shift recurrence, on the scoped rational-root / linear factorisation
  substrate (`rational_roots`, `roots_with_multiplicity`, `square_free`).
- **q-holonomic** (`q_shift_algebra`, `q_gosper`, `q_gosper_definite_sum`, `q_zeilberger`) —
  the q-analogues on `omnibias.qcalculus` primitives, with exact q-rational certificates for
  a fixed rational `q` and the `q -> 1` distinct-limit framing.
- **Transforms & closures** (`dfinite_to_precursive`, `precursive_to_dfinite`,
  `dfinite_derivative`, `dfinite_integral`, `dfinite_compose_poly`) — the exact
  ODE ⇔ coefficient-recurrence bridge and D-finite closure operations.
- **Guessing** (`guess_recurrence`, `guess_dfinite`, `guess_algebraic`) — minimal
  P-recursive / differential / algebraic annihilators, guessed by exact null space and
  verified on held-out terms.
- **Asymptotics** (`precursive_asymptotics`, `certified_asymptotic`) — the Poincaré–Perron
  leading rate / exponent (numerical), bridged to `omnibias.difference.transfer_theorem` for
  a certified coefficient where the singularity is known.
- **Lean-certified identities** (`prove_hypergeometric_identity`,
  `prove_identity_zeilberger`) — classic binomial identities discharged as per-coefficient
  `rational_identity` obligations the omnibias Lean kernel checks; the Zeilberger path's
  `P(n, k) == 0` obligations hold for **all** `n`.

!!! note "Honest labels & `theorem_prover_verified`"
    Ore-algebra arithmetic, Gosper / Zeilberger sums, transforms, and the certificate
    payloads are exact / **closed-form** — the symbolic closures and the Zeilberger
    obligations hold for **all `n`**. *Which* recurrence a guessed sum obeys is **guessed**
    (heuristic) and then **verified** exactly on the range. `factor` is scoped to the
    rational-root / linear regime and `asymptotics` returns a **numerical** leading term
    (certified only where `transfer_theorem` applies). `theorem_prover_verified` is earned
    **only** on a genuine `lake build` pass of every obligation and is never forged — no Lean
    toolchain present degrades gracefully. This package works in the discrete register
    founded by `omnibias-difference` (the `delta -> 0` collapse). Smoke:
    `docs/examples/holonomic_validate.py`.

Finite Keller-map identities (`omnibias.holonomic.keller`) are exact
`Q` algebra: a replay of Alpöge / Gallagher and a blind deg-2 / deg-3
tangent-sweep search via `run_discovery`. Prefix-verified guess families
(`holonomic_recurrence_guess`, `holonomic_dfinite_guess`,
`holonomic_algebraic_guess`) recover an annihilator on a finite prefix;
all-`n` continuation stays the Zeilberger / `HolonomicProof` obligation.
They do **not** prove the Jacobian conjecture. Dimension `n = 2` is a
separate finite universal (`omnibias.holonomic.jacobian_n2`): integer
maps of degree `<= d` and height `<= h` are exhausted for
`C_box(d,h,G)`. A miss is not injectivity on `Q^2` and not the parent.
`escalate_n2_result` sets `jacobian_n2_claim` only on an exact
violator. The Ore condition sort (`condition_ore`) wraps the recurrence
guess as a `ConditionHypothesis`. `condition_dfinite` wraps the
differential annihilator guess when the observation carries
`extra.series=1`. Cookbooks:
[Keller Jacobian replay](../cookbook/keller-jacobian.md),
[Jacobian n=2 finite box](../cookbook/jacobian-n2-box.md),
[Finite discovery engine](../cookbook/discovery-loop.md).

## Public API

::: omnibias.holonomic
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

Status: Alpha (`0.1.0a1`).

# omnibias-logic

Differentiable **and** certified **Boolean logic** on the omnibias stack: the weighted
**MaxSAT** consumer *re-exported unchanged* from `omnibias.discrete.maxsat`, plus new
**(weighted) #SAT / model counting** with a rigorous count enclosure.

Both problems are answered as a **yes-if** -- a certified object, never a `P = NP` / `#P`
exactness claim:

\[
\underbrace{\ell}_{\text{lower}} \;\le\; \#\mathrm{models} \;\le\;
\underbrace{u}_{\text{upper}}
\qquad\text{and}\qquad
\underbrace{\ell}_{\text{lower}} \;\le\; \mathrm{optimum} \;\le\;
\underbrace{E(z)}_{\text{decoded (upper)}} .
\]

- **MaxSAT** (re-exported): `max_sat` builds a weighted-violation `DiscreteProblem`; relax
  it (`omnibias.logic.{torch,jax}.maxsat_relaxation`), `decode` a min-violation assignment,
  and `certify_gap` the Lasserre / SOS **optimality gap**.
- **#SAT / model counting** (new): `model_count` builds a `ModelCountProblem`,
  `count_enclosure` returns a rigorous `lower <= #models <= upper` **inclusion-exclusion**
  (Bonferroni) sandwich that tightens with the truncation order, and `exact_model_count` is
  the exact `O(2^n)` oracle (built on `omnibias-boolean`) that self-checks it. The
  `beta -> inf` annealed `sat_relaxation` decodes to model witnesses that strengthen the
  lower bound.

### Guarantee taxonomy

Exact model counting is `#P`-hard, so every entry point is labelled by which corner of the
**exact / fast / worst-case-sound** triangle it occupies -- never conflated:

- **Exact + sound** on a tractable fragment: `xor_model_count` (affine / XOR via GF(2) rank --
  unweighted), `treewidth_model_count` (bounded-treewidth DP -- weighted), `count_models_exact`
  (component-caching #DPLL -- weighted, exponential worst case). `#2-SAT` / `#Horn-SAT` are
  `#P`-complete and deliberately **excluded** from the fast paths.
- **Certified enclosure + sound**: `count_enclosure` (the Bonferroni sandwich).
- **Sound router**: `count` auto-dispatches to the cheapest of the above and returns a tagged
  `CountResult` (`guarantee in {"exact", "certified_enclosure"}`); with `warm_start=True` it
  derives the DPLL branch order from the annealed relaxation (search speed only -- the count
  is order-invariant).
- **Formal (Lean-checkable) register**: `seal_count_certificate` (or `CountCertificate.seal`)
  turns a sound count into a tamper-evident v1 certificate whose finite obligation the
  Mathlib-free Lean kernel re-checks via `check_certificate` -- a tight, unweighted count as
  the integer identity `Z0 - S1 + S2 - ... = #models` (certified UNSAT included), otherwise a
  positive lower bound as a satisfiability sign. `theorem_prover_verified` is earned only by a
  genuine `lake build`; with no toolchain the check degrades gracefully.
- **Statistical, NOT worst-case sound** (quarantined in `omnibias.logic.approx`): an
  `(epsilon, delta)` hashing estimator and a split-conformal wrapper, returning an
  `ApproxCount` that can never be mistaken for a `CountCertificate`.

Terminology: every relaxation's `sigmoid(beta·)`, `beta → ∞` is the **feasibility /
temperature** sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct from
the **founding bias collapse** -- the multi-bias `delta → 0` limit to the closed-form
derivative `sigma^(K-1)` (see [Theory](../theory.md)).

## #SAT / model-counting problem & oracle

::: omnibias.logic.model_count.problem
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.logic.model_count.frontends
    options:
      show_root_heading: false
      heading_level: 3

## Exact sound counters (tractable fragments)

Exact and worst-case sound, each honest about its regime: the affine/XOR counter is poly-time
but **unweighted**; the treewidth DP is poly for bounded (heuristic) width and **weighted**;
the DPLL counter is exact and **weighted** but exponential in the worst case.

::: omnibias.logic.model_count.xor
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.logic.model_count.treewidth
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.logic.model_count.exact
    options:
      show_root_heading: false
      heading_level: 3

## Certified count enclosure

::: omnibias.logic.model_count.enclosure
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.logic.model_count.certificate
    options:
      show_root_heading: false
      heading_level: 3

## Sound router

::: omnibias.logic.model_count.route
    options:
      show_root_heading: false
      heading_level: 3

## Sealed, Lean-checkable certificate

Seal a `CountCertificate` into the canonical, hash-sealed [certificate v1](../index.md) format
and hand its finite obligation to the Mathlib-free Lean kernel
(`omnibias.core.proof.check_certificate`). The obligation classifier picks the strongest claim
the instance admits and **never widens** it: an exact-count *integer identity* for a tight,
unweighted, small enclosure (the finite inclusion-exclusion assembly the kernel re-checks with
`omega`; the inclusion-exclusion theorem and subset measures are trusted inputs recorded in
`meta`), otherwise an enclosed-quantity **sign** (a positive lower bound = certified
satisfiability). Tamper any bound and `verify_certificate_digest` fails; `theorem_prover_verified`
is set only on a genuine kernel pass and is never forged by the certificate.

The same certificate flows through the shared `omnibias.core.proof` **ProofMachine**:
`count_prover()` is a registerable `FunctionProver` (kind `"model_count"`) that builds + seals
the certificate and adjudicates a `Conjecture`'s claim (`"enclosure"` / `"sat"` / `"unsat"` /
`"count"`) as `PROVED` / `DISPROVED` / `BLOCKED` -- an exact `"count"` needs a *tight*
enclosure or it honestly `BLOCKED`s. It ships an **independent** replay twin (the `O(2^n)`
enumeration oracle, a different algorithm than inclusion-exclusion) so `strict=True` verdicts
demand an agreeing recount. `prove_model_count(...)` is the one-call driver; pass
`assert_theorem_prover=True` to *require* a Lean-kernel pass (the formal honesty gate blocks a
verdict that asserts it without one).

::: omnibias.logic.model_count.proof
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable #SAT relaxation (JAX)

::: omnibias.logic.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Backend twin (torch)

Bit-identical PyTorch twin of the #SAT relaxation.

::: omnibias.logic.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Statistical estimators (NOT worst-case sound)

!!! warning "Outside the sound surface"
    `omnibias.logic.approx` is deliberately **quarantined**. Its estimators carry only a
    **probabilistic / coverage** guarantee -- never a rigorous enclosure -- and return an
    `ApproxCount`, a type that is structurally distinct from `CountCertificate` and hard-wires
    `worst_case_sound = False`. It is **not** re-exported into the top-level `omnibias.logic`
    namespace; import it explicitly (`from omnibias.logic.approx import approx_model_count`).
    For guaranteed counts use `count` (the sound router) or `count_enclosure`.

::: omnibias.logic.approx.result
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.logic.approx.hashing
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.logic.approx.conformal
    options:
      show_root_heading: false
      heading_level: 3

## Re-exported MaxSAT surface

The weighted-MaxSAT front-end (`max_sat`, `MaxSATProblem`, `WeightedCNF`, `Clause`) and the
substrate helpers (`decode`, `brute_force_min`, `certify_gap`, `GapCertificate`,
`AnnealSchedule`) are re-exported unchanged from
[`omnibias-discrete`](discrete.md); the `maxsat_relaxation` twins live under
`omnibias.logic.torch` / `omnibias.logic.jax`.

Status: Alpha (`0.1.0a1`).

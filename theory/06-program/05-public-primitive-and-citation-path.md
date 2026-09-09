# 06-05 Public primitive and citation path

## 1. Thesis and status

The citable object is already shipped — shared integer coefficients,
six `OperatorBlock` roles, `compose_jet` — and this file is the order
in which to **publish and use** that object so it can be addressed
later without being mistaken for Group 09 inventions, a CCF residual,
or a prize claim.

- **Status**: shipped (G1–G5 document gates earned; obligation-3 `jet_vs_nested_ad` earned; `PUBLIC_SURFACE` frozen as a document list; methods-paper outline is jet-vs-AD 1-D Poisson; extract / paper / external stay later; design record)
- **Depends on**: 06-01, 06-02, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** One page answers "what must become public, in what
  order, and what must not be the public face."
- **How it works.** Four obligations (use, standard object, replace-a-
  default method, computation theorem), each with do / don't /
  done-looks-like, and a 12–24 month dependency order.
- **Strength.** Stops a 90th invention spec from being written as
  "the paper," and stops CCF Hilbert `~1e-1` from being the showcase.
- **When to use.** Before extracting a public surface, writing the
  first methods paper, or adding a jet-vs-AD benchmark.
- **When not.** This file does not extract a package, write a paper,
  or train a network.
- **Accuracy floor.** Document gates G1–G5 stay on this file.
  Obligation 3 numerical gates live in
  `benchmarks/jet_vs_nested_ad.py` (1-D Poisson; not CCF).
  CCF stretch remains an operator floor, not a citation-path gate.
  Extract, paper, and external reproduction stay later.

## 2. Where it lands

This file only (docs-only, same as 06-04). The *future* public surface
it names already lives in `omnibias-core` (`polynomials`,
`ActivationSpec`) and `omnibias.{torch,jax}` (`OperatorBlock`,
`compose_jet`, `mlp_jet`). No new package. Extracting a smaller
installable artifact is a later implementation PR, not this spec.

## 3. Prior art in omnibias

- `docs/operator-surface.md` — six roles: `identity`, `grad`,
  `laplacian`, `derivative`, `band`, `integral`
  (`S(z+b_hi)-S(z+b_lo)`, `S'=sigma`).
- `omnibias.core.polynomials` — `sigmoid_polynomial_coeffs`,
  `tanh_polynomial_coeffs` (shared integers; backends must not fork).
- `omnibias.core.spec.ActivationSpec` — fastpath and integral kernels.
- `omnibias.{torch,jax}.jet` — `compose_jet`, `layer_jet`, `mlp_jet`.
- `omnibias.{torch,jax}.optim.GaussNewton` — exact-J residual steps
  on a local operator.
- Spec 01-11 — rational collapse weights, Lean-checkable (gated).
- `formal/omnibias-verified-kernel` — finite rational `ZInterval`.
- Spec 09-01 — inventions are **not** this path.
- Spec 07-03 / 08-01 — CCF stretch is an operator gate; Hilbert floor
  `~1e-1`.

**Confirmed gap.** There is no written publish-and-use order that
forbids using CCF or Group 09 as the public face of the primitive.

## 4. Mathematics

The founding **bias collapse** (`delta -> 0`) supplies `sigma^(n)`.
The **window** knob supplies `band` / `integral`. Neither is
temperature collapse. Faà di Bruno (`compose_jet`) **is** the chain
rule. This spec asserts no new identity.

### Four obligations

Each obligation is a later work item. None is earned by writing this
file.

#### Obligation 1 — Use, not specs

- **Do.** One short paper plus one installable *small* surface
  (core coefficients + torch/jax jet). One external reproduction:
  a third-party repo that imports the package and reruns one named
  gate. Keep CCF and Group 09 off the first paper.
- **Don't.** Add a 09-27. Lead with campaign ticks or Clay-adjacent
  residuals.
- **Done looks like.** A DOI, an install of the small surface that
  does not require cloning the monorepo, and a third-party README
  that imports `omnibias.core.polynomials`.

#### Obligation 2 — Standard computational object

- **Do.** Freeze as the citable object: `ActivationSpec`,
  `sigmoid_polynomial_coeffs` / `tanh_polynomial_coeffs`, the six
  `OperatorBlock` roles, `compose_jet` / `mlp_jet`. Version the
  coefficient tables as a citable artifact. A reference snippet
  other PINN codes can copy without taking FermiNet, QUBO, or CCF.
- **Don't.** Wait for FTC-Net or jet-tokens (09-03, 09-02). Those
  are consumers. Do not call OMBU "the new transformer."
- **Done looks like.** Another library implements the pack roles or
  imports the coefficients and cites the paper for `sigma^(n)`
  instead of nested `grad`.

#### Obligation 3 — Method that replaces a default (narrow)

- **Do.** Head-to-head other people will rerun, on a **local** PDE
  (Poisson / heat / Burgers), not CCF: (a) nested AD vs closed-form
  jet — wall, memory, residual at fixed budget; (b) Adam vs exact-J
  `GaussNewton` on the same residual, multi-seed, named baseline,
  committed JSON. Optional later: integral cell vs quadrature
  (09-03 + 09-17) as a quadrature default, not an Adam default.
- **Don't.** ImageNet. Group 08 as "the new backprop." CCF as the
  showcase (a Hilbert-limited residual teaches the wrong lesson).
- **Done looks like.** A methods sentence other codes copy:
  derivatives of `sigma` via omnibias jets; train with GN on
  `r(theta)`.

#### Obligation 4 — Theorem about computation

- **Do.** A paper on the algebra and cost of the Riccati tower:
  recurrence, op-count vs nested AD, rounding model, bit-identity.
  Optional: one finite rational Lean identity (coefficient
  recurrence), `theorem_prover_verified` earned only by a genuine
  `lake build`.
- **Don't.** A PDE residual as the theorem. 09-26 export as a
  theorem. "Skip the chain rule." Continuum / NS / YM / RH / P vs NP.
- **Done looks like.** A lemma a PL / AD / verified-numerics referee
  can check without installing the monorepo.

### Dependency order (12–24 months, implement later)

```
small published object (coeffs + 6 roles + compose_jet)
    -> paper: algebra + complexity + bit-identity
    -> benchmark: jet vs nested AD + GN vs Adam on a local PDE
    -> one external user of the package
```

That is obligation 2, then 4, then 3, then 1.

### Explicit split

CCF (Hilbert × dictionary floor `~1e-1`, stretch gate `1e-13`) and
Group 08 / 09 remain a **separate campaign**. They are not the public
face of this path. A PR that uses stretch residual or an invention
spec as the first-paper claim fails this spec's G3.

## 5. Worked example

**Triaging a proposed first paper.**

*Proposal:* "Unstable singularities to `1e-13` with omnibias GN."

- Obligation test: uses CCF as the showcase. **Fails G3.** Point at
  07-03. Write the jet-vs-AD Poisson paper instead.

*Proposal:* "Jet-token transformers (09-02) as the standard object."

- Obligation 2: jet-tokens are a consumer. **Fails.** Freeze
  `compose_jet` and the six roles first.

*Proposal:* "The complexity of `sigma^(n)` and bit-identical backends."

- Matches obligation 4 after the small surface (obligation 2) is
  frozen. **Allowed** as paper two in the dependency order.

### Methods-paper outline (obligation 3 vehicle)

Showcase is **jets vs nested AD on 1-D Poisson**, not CCF stretch,
not Group 09, not Clay.

1. **Object (already frozen).** `PUBLIC_SURFACE`: coefficient tables,
   six `OperatorBlock` roles, `compose_jet` / `mlp_jet`.
2. **Theorem sentence.** Riccati tower cost vs nested `grad`;
   bit-identical torch/jax (obligation 4; later paper).
3. **Method benchmark.** `benchmarks/jet_vs_nested_ad.py` on 1-D
   Poisson: `mlp_jet` vs nested AD (agreement + order-6 wall); exact-J
   `GaussNewton` vs named Adam on the same residual. Smoke
   `docs/benchmarks/jet_vs_nested_ad_smoke.json`.
4. **Not the vehicle.** CCF Hilbert stretch, Group 08 trainers, Group
   09 inventions, Clay (C)/(D) or (A)/(B).
5. **Later.** Extract of `PUBLIC_SURFACE` as an installable; DOI;
   one external reproduction.

## 6. Proposed API

Does not exist as a module. The later extract is a **subset** of
shipped symbols, not a new API.

```python
# documentation only — the frozen public surface (already shipped)
PUBLIC_SURFACE_FROZEN = True
PUBLIC_SURFACE = (
    "omnibias.core.polynomials.sigmoid_polynomial_coeffs",
    "omnibias.core.polynomials.tanh_polynomial_coeffs",
    "omnibias.core.spec.ActivationSpec",
    "OperatorBlock.op in {identity, grad, laplacian, derivative, band, integral}",
    "omnibias.{torch,jax}.jet.compose_jet",
    "omnibias.{torch,jax}.jet.mlp_jet",
)
CITATION_ORDER = ("object", "theorem", "method_benchmark", "external_use")
NON_VEHICLE = ("ccf_stretch", "group_08_trainers_as_first_paper", "group_09")
```

## 7. Practical use cases

1. **Deciding the first paper** without leading on CCF or 09-02.
2. **Scoping a package extract** to `PUBLIC_SURFACE`, not 42 packages.
3. **Refusing a jet-vs-AD bench on CCF** (wrong operator floor).
4. **Scheduling Lean** on one coefficient identity, not 09-26.

## 8. Acceptance gates

Document gates (this file). Numerical gates belong to later PRs.

- **G1 completeness.** Section 4 names all four obligations with do /
  don't / done-looks-like.
- **G2 order named.** The dependency order is 2, then 4, then 3,
  then 1, written in section 4.
- **G3 CCF non-vehicle.** Section 4's split states that CCF / Group
  08 / Group 09 are not the public face of the citation path.
- **G4 no prize claim.** Honesty section forbids Turing, Clay, and
  Nobel inference from this path.
- **G5 no new package in this pass.** This spec creates no
  distribution.

Obligation 3 numerical gates (later PR, now present) live in
`benchmarks/jet_vs_nested_ad.py`. They do not relicense G1–G5.

## 9. Benchmark plan

Document gates stay on this file. Obligation 3 numerical gates:

- `benchmarks/jet_vs_nested_ad.py` — 1-D Poisson; `mlp_jet` vs nested
  AD (agreement + order-6 wall); exact-J `GaussNewton` vs named Adam
  on the same jet residual.
- Smoke: `docs/benchmarks/jet_vs_nested_ad_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/citation/jet_vs_ad/`

Extract, paper, and external reproduction are still later. No CI job
for this document itself; the bench smoke is wired separately.

## 10. Honesty and scope

- Not a Turing Award claim, not a Clay or Nobel claim, not a
  continuum PDE theorem.
- `theorem_prover_verified` and `mathlib_verified` appear only as
  *future earned* flags on a finite rational identity (obligation 4)
  and are never asserted here.
- Bias collapse and the window knob are named; no temperature
  collapse in the public surface.
- Certificate tier: document / empirical (later benches); Lean
  rungs only if a genuine `lake build` is attached later.

## 11. Open questions and risks

- **Scan catalog is not a larger extract.** The public object stays
  the six roles + `compose_jet`. Spec 01-13 indexes `scan(role)` and
  named consumers; it does not enlarge `PUBLIC_SURFACE`. 09-27,
  09-28, and 09-29 are consumers, not a larger extract.
- **Extract vs monorepo.** A too-small extract may omit `integral`
  and under-state the object; a too-large extract is the 42-package
  tree again. `PUBLIC_SURFACE` is the guard.
- **First paper delay.** Waiting for 09-03 to exist repeats the
  "consumers first" failure. G2 forbids that order.
- **Falsifier for this ledger.** If the first public paper leads on
  CCF stretch or a Group 09 architecture, G3 has failed even if the
  numbers are real.

## 12. Implementation checklist

- [x] `theory/06-program/05-public-primitive-and-citation-path.md`
      (this file); document gates G1–G5 earned by
      `test_theory_citation_path.py`
- [x] Freeze `PUBLIC_SURFACE` as the document list (this pass;
      `PUBLIC_SURFACE_FROZEN = True`)
- [x] Methods-paper outline: jets vs nested AD on 1-D Poisson
      (not CCF, not Group 09, not Clay)
- [ ] Later: extract `PUBLIC_SURFACE` as an installable
      artifact
- [ ] Later: algebra + complexity + bit-identity paper
- [x] `benchmarks/jet_vs_nested_ad.py` plus smoke JSON
      (obligation 3 numerical; extract / paper / external stay later)
- [ ] Later: one external reproduction
- [x] Index row in `theory/README.md` (wired in the index pass)

---

## Repo invariants this spec must respect

Check these before submitting an implementation.

- **Pure core**: no torch, jax, tensorflow or keras imports from
  `omnibias.core`. Pure-Python math lives there and every backend imports it.
- **Bit-identical twins**: torch and jax implementations must agree exactly.
  Polynomial coefficients come from `omnibias.core.polynomials`; never fork them
  per backend.
- **Default dtype**: use the framework default
  (`torch.get_default_dtype()` / `keras.config.floatx()`), never a hardcoded
  `float32`.
- **Vendor-neutral language**: no scheduler commands, vendor names, internal
  hostnames, usernames, or absolute local paths in tracked files. Artifacts go
  to `$OMNIBIAS_SCRATCH`, defaulting to a repo-relative `artifacts/`.
  `packages/omnibias-core/tests/test_no_leakage.py` enforces this across the
  whole readable surface.
- **Terminology**: distinguish the founding bias collapse (`delta -> 0`, yields a
  derivative) from temperature collapse (`beta -> inf`, yields a 0/1 step). The
  older penalty phrasings are retired and guarded by `tests/test_terminology.py`.
  Every new `**/relaxation.py` must carry the cross-reference note and be listed
  in `PENALTY_FILES` in
  `packages/omnibias-core/tests/test_concept_terminology.py`.
- **Executable docs**: if any part of this lands in `docs/` or a package README,
  every fenced Python block is executed by `tests/test_docs_snippets.py`. Verify
  calls against real signatures; opting out needs a directive with a stated
  reason.
- **Earned flags**: `theorem_prover_verified` is set only by a genuine
  `lake build` pass of the Mathlib-free kernel, and `mathlib_verified` only by a
  genuine pass of the Mathlib-backed project. Asserting either without a pass
  blocks the verdict.
- **Typing tier**: the strict CI gate covers `core`, `torch`, `jax` and
  `ferminet`. Newly authored modules should be written strict-clean regardless
  of tier; curated beta modules can be added to
  `scripts/mypy_strict_allowlist.txt` once
  `mypy --strict --follow-imports=silent <file>` is clean.

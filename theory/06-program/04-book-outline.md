# 06-04 Book outline

## 1. Thesis and status

The fifty-four specs are not a list — they are one argument told in the wrong
order, and this file puts them in the right one: **collapse selects order, gap
selects window, scan selects position, arrangement selects region, equality
selects the solution locus.** Five choices on one primitive, and everything else
is consequence.

- **Status**: concept
- **Depends on**: 01-10
- **Blocks**: none

The listed dependency is the formal backbone. In practice the manuscript depends
on **every** spec, since a chapter may only be written once its specs' gates
have run (gate G1 below).

## 2. Where it lands

A separate manuscript, not `docs/`. `docs/` documents what ships; this would be
an exposition of a theory, and mixing the two would degrade both. The right
mechanics are a sibling repository or a `book/` tree that is explicitly not part
of the mkdocs site.

Not a package under any reading of the rule.

## 3. Prior art in omnibias

- `docs/theory.md` — the shipped primer, including the "Two senses of collapse"
  section. **This is the canonical published account of what the library is**,
  and the book must not contradict it. Where the book goes further, it must be
  visibly marked as program rather than product.
- `docs/operator-surface.md` — the capability matrix, and the source of truth
  for the six `OperatorBlock` roles.
- `docs/packages.md` — the 42-package inventory with maturity tiers.
- `docs/index.md`, `docs/cookbook/`, `docs/examples/` — the existing narrative
  layers, all executable via `tests/test_docs_snippets.py`.
- `theory/README.md` — the index and dependency graph, which is the book's
  skeleton in miniature.

**Confirmed gap.** There is no monograph, and there should not be one until the
gates it would report have actually run.

## 4. Mathematics

The through-line, stated once, since it is the book's spine.

An OMBU unit is `K` parallel hyperplanes sharing a normal `w`, differing only in
bias. Five independent choices act on that object:

1. **Collapse selects order.** Send the spread `delta -> 0` with `K` biases and
   the unit becomes `sigma^(K-1)` — exactly, in closed form, at the cost of one
   `sigma` evaluation. The founding limit. *(Specs 01-01, 01-04, 01-05, 04-01.)*
2. **Gap selects window.** Keep the spread finite and the same two hyperplanes
   bound a **slab**: the `band` and `integral` roles, an antiderivative window
   `S(z + b_hi) - S(z + b_lo)`. *(Specs 02-14, 04-02.)*
3. **Scan selects position.** Slide one pack template along `w` and you have a
   grid-free convolution, translation-equivariant by construction. *(Specs
   01-02, 02-01, 02-08, 03-08, 05-01.)*
4. **Arrangement selects region.** Many packs with *different* normals cut space
   into cells; the combinatorics of those cells is a face lattice, a tope graph,
   a polytope. *(Specs 01-03, 02-02, 03-02, 03-09, 05-02.)*
5. **Equality selects the locus.** Force two collapsed units to agree and their
   equality set is a codimension-one manifold whose Jacobian the tower supplies
   exactly — the solution locus of an implicit problem. *(Specs 01-09, 02-12,
   05-01.)*

Everything in the tree is one of these five, or a composition of them, or an
honesty constraint on what the result licenses.

The **formal backbone** is spec 01-10: the jet bundle. Collapse is prolongation,
scan is translation along the base, arrangement is a stratification, equality is
a fibre condition. That chapter is what makes the five choices one object rather
than five tricks, and it is the reason the book is a book.

## 5. Worked example

**The outline.**

**Part I — The primitive** *(what the library already is; must agree with
`docs/theory.md`)*

1. One `sigma` evaluation, any order. The Riccati identity, the Eulerian /
   Legendre / Hermite recurrences, and why the coefficients live in one
   pure-Python module.
2. Two senses of collapse. Bias collapse (`delta -> 0`, a derivative) against
   temperature collapse (`beta -> inf`, an indicator). *The chapter that
   prevents every later confusion.*
3. Jets. Directional and multivariate, `compose_jet` through `mlp_jet_mv`, and
   why one forward pass yields every mixed partial.
4. Three registers. Differentiable, rigorous, formal — and the claim ladder of
   spec 06-02, stated early so the reader knows what each later result licenses.

**Part II — The five choices** *(the program's geometric core)*

5. Order: heterogeneous multi-packs and the Birkhoff sample (01-01, 01-04).
6. Window: slabs, mollifiers and the distributional limit (01-05, 04-02).
7. Position: the bias scan as convolution without a grid (01-02).
8. Region: hyperplane arrangements, cells and tope graphs (01-03).
9. Locus: equality, transversality and the implicit function theorem (01-09).
10. The jet bundle. The formal language that makes 5 through 9 one object
    (01-10).

**Part III — Architectures** *(specs 02-01 … 02-14, grouped by which choice they
exploit)*

11. Scanning architectures (02-01, 02-08).
12. Arrangement architectures (02-02).
13. Basis architectures: Jet-KAN, Hermite ladders, solitons (02-03, 02-09,
    02-10).
14. Interface architectures: weak-form VPINN, transmission PINN, transfer
    matrices (02-04, 02-05, 02-11).
15. Implicit and transform architectures (02-12, 02-13).
16. Kernels and potentials: BEM-Net and the fast-multipole split (02-06, 02-07).
17. Gauge: the Wilson-line holonomy band (02-14).

**Part IV — Algorithms** *(specs 03-01 … 03-13, grouped by what they compute)*

18. Search and evolution (03-01).
19. Optimization and geometry: LP, CSP, line search, refinement (03-02, 03-03,
    03-12, 03-13).
20. Measure and integration: sliced OT, quadrature, scale flow (03-04, 03-06,
    03-07).
21. Shape and topology: morphology, differentiable topology (03-05, 03-09).
22. Analysis: certified localization, Padé tracking, Lie symmetry (03-08, 03-10,
    03-11).

**Part V — Rigour** *(what separates this from a methods catalogue)*

23. Sound enclosures. Intervals, Taylor models, and why outward rounding is not
    optional.
24. Certificates. The hash-sealed v1 format, reserved honesty keys, and why
    `theorem_prover_verified` cannot be forged.
25. The Lean loop. Finite rational obligations, the Mathlib-free kernel, the
    Mathlib-backed tier, and exactly what is out of scope.
26. Rational exactness. Which of the program's own identities are themselves
    machine-checkable (01-11).

**Part VI — Frontier and limits** *(Group 07, written last and most carefully)*

27. The sub-obligation method. Decomposing a famous problem into finite or
    compact pieces, and the discipline of naming the parent (07-01).
28. Case studies: fluids, gauge, spectra, dynamics (07-02 … 07-06).
29. What this program cannot do. The forbidden-claims register with its
    reasoning, written as mathematics rather than as disclaimer.

**Appendices** — the gate protocol (06-01), the packaging record (06-03),
reproduction instructions, and the artifact index.

**The rule that governs the whole manuscript**: a chapter may be written only
when its specs' gates have run. Chapters citing failed gates are welcome and
should say so; chapters citing gates that have not run are not.

## 6. Proposed API

None. A manuscript.

The only mechanical requirement worth stating: **every code listing in the book
is executable and tested**, following the existing `tests/test_docs_snippets.py`
discipline, where a block may rely only on names defined earlier in the same
document. A book of plausible-looking API calls that do not run would be worse
than no book, and the repository already has the machinery to prevent it.

## 7. Practical use cases

1. **A coherent account** for a reader who should not have to reconstruct the
   argument from fifty-four specs and forty-two packages.
2. **Teaching.** Parts I and II are a graduate course; Part V is a second one.
3. **Onboarding.** The five-choice spine is the fastest correct mental model of
   the library.
4. **A record of the honesty stack** as intellectual content rather than legal
   boilerplate — Chapter 29 is arguably the most useful chapter in the book.

## 8. Acceptance gates

Unusual for a manuscript, but the point is that they exist.

- **G1 gates before chapters.** Every empirical claim in the book cites a
  committed artifact under `docs/benchmarks/` with its `gates` block. A claim
  without an artifact is cut.
- **G2 executable listings.** Every code listing runs under the docs-snippet
  discipline, with opt-outs requiring a stated reason.
- **G3 no contradiction with `docs/theory.md`.** A reviewer diffs the book's
  Part I against the shipped primer; any divergence is resolved in the primer's
  favour or the primer is updated deliberately.
- **G4 claim-guard clean.** The manuscript passes `test_terminology.py` and
  `test_forbidden_claims.py` from spec 06-02.
- **G5 failure is represented.** At least the failed gates that exist are
  reported in the relevant chapters. A book in which everything worked is
  evidence that the gates were not binding.

## 9. Benchmark plan

Not applicable. The manuscript's evidence is the artifact index in the
appendix, which points at the committed JSON files rather than restating their
numbers.

## 10. Honesty and scope

- **The book must not be written yet.** Most specs in this tree are `concept` or
  `designed`; a monograph asserting them would be describing intentions as
  results. This outline exists so the writing has a target, not so it can start.
- `docs/theory.md` remains canonical for what the library *is*. The book covers
  what the program *could become*, and the distinction must be visible on the
  page, not just in the front matter.
- Both collapse limits appear throughout; Chapter 2 exists specifically to keep
  them apart, and the terminology guard applies to the manuscript.
- No certificate tier. A book is not a certificate.
- The five-choice framing is an organizing claim about the material, not a
  theorem. It should be presented as a useful decomposition that the jet-bundle
  chapter then makes precise.

## 11. Open questions and risks

- **Premature writing is the main risk.** The pull toward a monograph is
  strongest exactly when the fewest gates have run. G1 is the defence.
- **Drift from the shipped library.** A book written against a moving codebase
  is stale on publication. Tying listings to the executable-docs harness bounds
  this but does not eliminate it.
- **The five-choice spine could be wrong.** If, once implemented, the specs do
  not decompose this way — for instance if scan and arrangement turn out to be
  the same choice in different coordinates — the outline should be rebuilt
  rather than defended.
- **Chapter 29 is the hardest to write well.** Stating limits as mathematics
  rather than as apology requires the sharpest thinking in the book, and it is
  the chapter a reader will test the rest against.
- **Falsifier.** If Part II cannot be written without appealing to results that
  never passed a gate, the program is not ready and the outline should say so
  rather than the book proceeding.

## 12. Implementation checklist

- [ ] Keep this outline current as specs change status
- [ ] Do not begin drafting a chapter until its specs' gates have run
- [ ] `book/` tree excluded from the mkdocs site, or a sibling repository
- [ ] Executable-listing harness reused from `tests/test_docs_snippets.py`
- [ ] Artifact index appendix generated from `docs/benchmarks/`
- [ ] Part I diffed against `docs/theory.md` before any release
- [ ] Terminology and forbidden-claims guards run over the manuscript
- [ ] Index row in `theory/README.md`

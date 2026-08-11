# 06-02 Honesty and claim boundaries

## 1. Thesis and status

A research program that can produce a Lean-checked theorem and a five-seed
benchmark in the same repository needs one written rule for which sentence each
result licenses — and, more importantly, a written list of sentences that are
never licensed by anything this program can produce.

- **Status**: designed
- **Depends on**: 06-01
- **Blocks**: 06-03, 07-01

## 2. Where it lands

Documentation plus enforcement. The prose lands in `docs/` and this file; the
enforcement lands in existing guards (`tests/test_terminology.py`,
`omnibias.core.proof.certificate`) extended to cover the theory tree. No new
module and certainly no new package: an honesty policy that lives in a package
nobody imports is decoration.

## 3. Prior art in omnibias

The enforcement machinery is real and already strict.

- `omnibias.core.proof.certificate` — `RESERVED_HONESTY_KEYS`, currently
  `{theorem_prover_verified}`. `make_certificate` raises if a producer supplies
  it, and `schema_errors_v1` reports it as an error on an already-sealed or
  hand-built certificate. The flag is earned only by a Lean kernel pass.
- `omnibias.pinn.solver._core.honesty` — `honesty_labels()` hard-wires
  `unproven_claim=False`, raises if `continuum_claim` is truthy, and rejects any
  reserved key passed through `**extra`. The method labels `CLOSED_FORM /
  AUTODIFF / NUMERICAL / SPECTRAL / HIGH_ORDER` prevent a numerically
  differentiated result from being described as closed form.
- `omnibias.geometry.gauge.transfer` — `continuum_claim = False` fixed in the
  transfer-matrix gap certificates.
- `benchmarks/_gates.py` — `ipm_boussinesq_scaffold_gates` returns
  `earned: False` *and blocks the whole verdict* if
  `navier_stokes_proof_claim` is passed truthy.
- `tests/test_terminology.py` — retires `collapsed-bias` and
  `bias-collapse penalty` wordings by regex across `packages`, `docs`, `theory`,
  `tests`, `scripts`, the rule and skill trees, and the root files. This file is
  on its `ALLOWED` list, because documenting the guard means naming what it
  retires.
- `packages/omnibias-core/tests/test_no_leakage.py` — vendor-neutral language
  guard that self-tests its own blocklist so it cannot go vacuous.
- `.cursor/rules/frontier-claims.mdc` and the `omnibias-dev-frontier-research`
  skill — the doctrine in agent-facing form.

**Confirmed gap**, and it is specific: `tests/test_terminology.py`'s
`SCANNED_ROOTS` does not include `theory`, so **this entire tree is currently
unguarded**. That is the first thing to fix, and it is a one-line change plus a
run.

## 4. Mathematics

### The claim ladder

Four rungs, each strictly stronger, none implying the next.

| Rung | Earned by | Licenses the sentence | Does not license |
|---|---|---|---|
| **1 Empirical** | `gates_block(...)["all_passed"]` on a benchmark | "on this problem, at this size, this method achieved X" | anything about other problems, sizes, or the true value |
| **2 Sound enclosure** | outward-rounded interval arithmetic in `omnibias.core.verified` | "the true value of this quantity lies in `[a, b]`" | anything outside the enclosed domain |
| **3 Kernel-verified** | a genuine `lake build` pass of the Mathlib-free kernel | "this finite rational obligation is machine-checked" | any infinite or analytic statement |
| **4 Mathlib-verified** | a genuine pass of `formal/omnibias-analytic` | "this finite rational obligation is machine-checked against Mathlib" | the same limits as rung 3 |

Rungs 3 and 4 are **distinct tiers, never conflated**. Both Lean projects are
`sorry`-free and both are scoped to **finite, rational** obligations. Limits,
continuum statements and asymptotics are not expressible in them at all, which
is a deliberate design choice: an obligation that cannot be stated cannot be
silently discharged.

### The forbidden-claims register

These sentences are never licensed by any result this program can produce. Each
is paired with the true statement that replaces it.

| Never write | Write instead |
|---|---|
| "we prove global regularity for Navier-Stokes" | "we certify a finite residual enclosure on a discretized problem; global regularity is an external obligation" |
| "we prove the Yang-Mills mass gap" | "we certify a spectral gap for one fixed transfer matrix at one spacing in finite dimension; the continuum limit is not taken" |
| "we prove / disprove the Riemann Hypothesis" | "`omnibias.core.verified.dirichlet` encloses Dirichlet series on `Re(s) > 1`; continuation and zeros are external" |
| "P = NP" or "we solve an NP-hard problem in polynomial time" | "a differentiable relaxation plus a decoder, with a certified optimality gap that is honest about being non-tight" |
| "a certificate implies a continuum result" | "a finite certificate constrains a finite object; the passage to the continuum is a separate, unmade argument" |
| "the Lean kernel verified our analysis" | "the Lean kernel verified a finite rational obligation extracted from the certificate" |
| "closed form" for an autodiff or finite-difference path | the honest `AUTODIFF` / `NUMERICAL` / `SPECTRAL` label |

The Padé and Borel tooling of spec 03-10 deserves a line of its own: it locates
singularities of a truncated series and **may not be used to claim analytic
continuation** of anything, in particular not of a Dirichlet series past
`Re(s) = 1`. This is exactly the kind of step that looks like a small technical
extension and is in fact the entire open problem.

### The two-collapse discipline

Two limits share the word "collapse" and must be named apart every time:

- **bias collapse**, the founding limit: spread `delta -> 0`, `K` biases
  coalesce, the unit becomes `sigma^(K-1)` — a smooth **derivative**;
- **temperature collapse**: `beta -> inf`, a gate sharpens into a `0/1`
  **indicator** — a feasibility step.

Retired wordings (`collapsed-bias`, `bias-collapse penalty`) are guarded by
regex, and `theory` is now one of the guard's scanned roots — so this discipline
is enforced over the research tree, not merely asserted by it.

Across this tree: specs 01-01, 01-04, 01-05, 04-01 and 05-01 use bias collapse;
specs 01-08, 03-02, 03-03, 03-05, 03-09 and 05-02 use temperature collapse; spec
05-02 uses **both**, and is the one to check most carefully.

### The Group 07 rule

Every frontier spec carries a mandatory **section 13** naming its external
parent problem and stating, in one sentence, that the parent is not claimed.
The sub-obligation must be **finite or compact**; if it is neither, it is not a
sub-obligation but a restatement of the parent, and the spec should not exist.

### Method labels

`omnibias.pinn.solver._core.honesty` already defines the vocabulary:
`CLOSED_FORM`, `AUTODIFF`, `NUMERICAL`, `SPECTRAL`, `HIGH_ORDER`. Every derivative
path in every theory spec must be labelled with one of these, because
"omnibias computes closed-form derivatives" is true of the activation tower and
false of, say, a metric derivative in `omnibias.geometry`, which is exact
forward-mode autodiff of an analytic metric. Conflating them would be the most
plausible overclaim in the whole library, and it is already handled correctly in
the shipped code.

## 5. Worked example

**Three results about the same object, and the three sentences they license.**

Consider a transfer matrix `T` arising from a lattice gauge computation at
spacing `a = 0.1`, in a truncated `64`-dimensional space.

*Result A, empirical.* A benchmark reports the numerically computed spectral gap
as `0.4137` with five-seed agreement to `1e-6`.

> Licensed: "the numerically computed gap of this `64`-dimensional truncation at
> `a = 0.1` is `0.4137`."
> Not licensed: that the true gap of the truncation is `0.4137` — floating-point
> eigensolvers are not sound.

*Result B, sound enclosure.* `omnibias.core.verified.eig_operator`'s
Lehmann-Maehly-Goerisch bounds give the gap in `[0.4131, 0.4142]`.

> Licensed: "the gap of this matrix lies in `[0.4131, 0.4142]`."
> Not licensed: anything about `a -> 0`, about dimension `> 64`, or about the
> continuum theory.

*Result C, kernel-verified.* The finite rational obligation "the enclosed gap is
strictly positive" is extracted and discharged by the Mathlib-free kernel;
`theorem_prover_verified` is set **by the kernel**, not by the producer.

> Licensed: "positivity of the enclosed gap for this fixed matrix is
> machine-checked."
> Not licensed: **the Yang-Mills mass gap**, which is a statement about the
> continuum limit of a family of theories and is not touched by any of A, B or
> C.

The distance between C and the parent problem is not a matter of tightening
constants. It is the entire content of the open problem, and no accumulation of
C-type results closes it. `omnibias.geometry.gauge.transfer` already encodes
this by fixing `continuum_claim = False`, and the honest framing is that C is a
useful, checkable, publishable *tool* result.

**The failure mode, concretely.** A paper draft that writes "we obtain a
machine-verified spectral gap, providing evidence for the Yang-Mills mass gap"
has committed the error. "Evidence for" is doing work the result cannot support:
a positive gap at one spacing in one truncation is consistent with a vanishing
continuum gap, so it is not evidence in any usable sense. The correct sentence
names the truncation and stops.

## 6. Proposed API

Mostly guards rather than new surface.

```python
# tests/test_terminology.py  -- one-line change, first deliverable
SCANNED_ROOTS = (
    "packages", "docs", "tests", "scripts", "theory",   # <-- add
    ".cursor/rules", ".cursor/skills", ".claude/skills",
)

# packages/omnibias-core/tests/test_forbidden_claims.py  -- new
FORBIDDEN = (
    r"prove[sd]?\s+global\s+regularity",
    r"prove[sd]?\s+the\s+(Yang-Mills\s+)?mass\s+gap",
    r"prove[sd]?\s+(the\s+)?Riemann\s+Hypothesis",
    r"\bP\s*=\s*NP\b",
    r"analytic\s+continuation\s+of\s+.*(zeta|Dirichlet)",
)
# Scans the same roots as the terminology guard, with an ALLOWED set for the
# files that quote the forbidden strings in order to forbid them, and a
# self-test asserting the patterns match a synthetic violation.

# theory/ structural guard -- new
def test_group_07_specs_have_section_13(): ...
def test_group_07_specs_name_external_parent(): ...
```

The self-test matters: `test_no_leakage.py` already self-tests its blocklist so
it cannot go vacuous, and any new claim guard must do the same, or a future
refactor that breaks the regex will silently disable it.

## 7. Practical use cases

1. **Writing a paper or release note** from a repository result, without having
   to reconstruct which rung it sits on.
2. **Reviewing a Group 07 spec** against a checklist rather than a feeling.
3. **Onboarding an agent.** The rules and skills already carry this doctrine;
   this file is the human-readable source they point at.
4. **Refusing a request cleanly.** When someone asks for "the Navier-Stokes
   result", the register supplies the honest reframing rather than a flat no.
5. **Surviving success.** The moment a result is genuinely good is the moment
   the temptation to round it up appears; the register is written in advance for
   exactly that moment.

## 8. Acceptance gates

- **G1 theory tree guarded.** `theory` is in `SCANNED_ROOTS` and
  `tests/test_terminology.py` passes over all 54 specs.
- **G2 forbidden-claims guard live.** `test_forbidden_claims.py` scans the same
  roots, passes on the current tree, and **fails on a synthetic violation**
  injected by its own self-test.
- **G3 Group 07 structure.** A test asserts every file in `theory/07-frontier/`
  contains a section 13, names an external parent, and states the non-claim.
- **G4 reserved keys.** Existing behaviour reconfirmed by test: supplying
  `theorem_prover_verified` to `make_certificate` raises, and
  `schema_errors_v1` reports an error for a body containing it.
- **G5 method labels.** Every theory spec that describes a derivative path uses
  one of the five method labels; checked by a lint over the tree.
- **G6 no vacuous guards.** Each guard added here has a self-test that fails if
  the guard's pattern set is emptied.

## 9. Benchmark plan

Not a benchmark. The deliverables are guards, and their evidence is that they
fail on synthetic violations:

- `tests/test_terminology.py` extended and run over `theory/`,
- `packages/omnibias-core/tests/test_forbidden_claims.py` with self-test,
- `packages/omnibias-core/tests/test_theory_structure.py` for the Group 07
  structural checks,
- all wired into the existing CI test job, which already runs the core tests.

## 10. Honesty and scope

- This file constrains **claims**, not ambition. Nothing here says a hard problem
  should not be attacked; the doctrine in `AGENTS.md` is explicit that ambition
  inside existing packages is encouraged and that structural impossibility must
  be distinguished from absent implementation.
- The guards are **regex over text**. They catch the phrasings we thought of, in
  the files we scan. They cannot catch a novel overclaim, and treating a passing
  guard as proof of honesty would itself be an overclaim.
- The claim ladder is about *what is licensed*, not about *what is valuable*. A
  rung-1 empirical result on a hard problem can be far more valuable than a
  rung-4 verification of something trivial.
- Both collapse limits appear in this spec, named apart, as required.
- No certificate tier: this spec produces no certificates.

## 11. Open questions and risks

- **Regex brittleness.** "we prove global regularity" is caught; "our results
  establish that solutions remain smooth for all time" is not. The guard is a
  floor, and review is the ceiling.
- **False positives on quoted text.** A spec discussing why a claim is forbidden
  contains the forbidden string. The `ALLOWED` set handles this, and the risk is
  that it grows until the guard is vacuous — so `ALLOWED` membership should
  require a comment giving the reason, as `test_terminology.py` already does.
- **Rung inflation over time.** A result described at rung 2 in a spec and at
  rung 3 in a later release note is the realistic failure. Linking every public
  claim to the artifact that earned it is the mitigation, and it is not
  automated.
- **The `dirichlet` boundary is easy to cross by accident.** Spec 03-10's Padé
  machinery genuinely does extrapolate, and the step from "extrapolate a
  truncated series" to "continue a Dirichlet series" is one line of code and an
  entire open problem. It deserves a dedicated test, not just a register entry.
- **Falsifier.** If the forbidden-claims guard cannot be made to pass on the
  current tree without a large `ALLOWED` set, then the tree already overclaims
  and the specs must be fixed before the guard is weakened.

## 12. Implementation checklist

- [x] Add `theory` to `SCANNED_ROOTS` in `tests/test_terminology.py` and fix any
      hits
- [ ] `packages/omnibias-core/tests/test_forbidden_claims.py` with a self-test
      that injects a synthetic violation
- [ ] Dedicated test that no Padé / Borel path claims analytic continuation of a
      Dirichlet series
- [ ] `packages/omnibias-core/tests/test_theory_structure.py` for Group 07
      section 13, external parent, and non-claim sentence
- [ ] Method-label lint over the theory tree
- [ ] `ALLOWED` entries require a reason comment
- [ ] Reconfirm reserved-key behaviour with an explicit test
- [ ] Docs page carrying the claim ladder and forbidden-claims register
- [ ] Cross-reference from `.cursor/rules/frontier-claims.mdc`
- [ ] Index row in `theory/README.md`

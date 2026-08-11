# 07-01 The sub-obligation ledger

## 1. Thesis and status

Five famous problems, five finite or compact sub-obligations that the new
primitives can actually attack, and — for each — the absolute gate that decides
it and the sentence that must never be written. This file is the ledger the
other six frontier specs are entries in.

- **Status**: designed
- **Depends on**: 01-11, 06-01, 06-02
- **Blocks**: 07-02, 07-03, 07-04, 07-05, 07-06, 07-07
  here

## 2. Where it lands

A document plus one guard. The ledger lives in this file and mirrors into
`docs/`; the guard is the Group 07 structural test from spec 06-02. No module,
no package.

## 3. Prior art in omnibias

The honesty machinery is shipped and strict; this ledger indexes it.

- `omnibias.pinn.certified.navier_stokes` — `HonestyLabels`, and validators that
  **append an error** unless `continuum_navier_stokes_claim` is `False`, at
  three separate certificate sites. `machine.py` blocks the whole verdict if the
  flag is `True`.
- `omnibias.geometry.gauge.transfer` — `certified_transfer_matrix_gap` with
  `continuum_claim = False` fixed in `gap.py` and `certificates.py`.
- `omnibias.core.verified.dirichlet` — `zeta_enclosure`,
  `certified_dirichlet_series`, `l_function_enclosure`, `theta_enclosure`,
  `p_series_tail_bound`. **Scoped to `Re(s) > 1`.**
- `omnibias.core.verified.eig_operator` — `lehmann_maehly_lower_bounds`,
  `certified_spectral_gap`, `interval_ldlt_inertia`, `temple_lower_bound`,
  `LehmannCertificate`, `SpectralGapCertificate`.
- `omnibias.sos` — `positivstellensatz`, `certify`, `honesty`, `proofmachine`.
- `omnibias.qubo` / `omnibias.discrete` — `certify_gap`, honest non-tight
  bounds.
- `benchmarks/_gates.py` — `ccf_absolute_gates` and
  `ipm_boussinesq_scaffold_gates`, both emitting
  `navier_stokes_proof_claim: False`.
- `.cursor/rules/frontier-claims.mdc`, `omnibias-dev-frontier-research` skill.

**Confirmed gap.** There is no single index. The flags are enforced
individually, correctly, in eight places; nobody can currently answer "what are
we actually attacking, and how far are we" from one page.

## 4. Mathematics

### What makes a sub-obligation legitimate

Three tests, all of which must pass.

1. **Finite or compact.** The statement quantifies over a finite set, a compact
   set with a computable modulus, or a fixed finite-dimensional object. "For all
   `t > 0`" fails. "On `[0, T]` with `T` fixed, for this fixed discretization"
   passes.
2. **Gate-decidable.** There is an absolute threshold, fixed in advance, whose
   passage is checkable by machine.
3. **Non-implying.** Passing it does not, by any argument the authors possess,
   imply the parent. If it did, it would *be* the parent, and the honest
   response to believing otherwise is to write the implication down and have it
   checked.

Test 3 is where enthusiasm fails. A sub-obligation that "would give strong
evidence for" the parent usually gives none: the parent's difficulty is
concentrated in the passage the sub-obligation avoids.

### The ledger

**Parent: Navier-Stokes global regularity (Clay).**

- Sub-obligation: a **sound enclosure of the residual** of a discretized
  incompressible flow on a fixed periodic box over a fixed time horizon, with
  the enclosure verified rather than estimated.
- Gate: `require_enclosure_coverage` at `100%` plus a named absolute residual
  threshold; the existing scaffold floors are `IPM_SCAFFOLD_RESIDUAL_GATE = 2.0`
  and `BOUSSINESQ_SCAFFOLD_RESIDUAL_GATE = 2.0`, to be tightened as the method
  improves.
- Sealed scope: one discretization, one box, one horizon, finite dimension.
- Never write: *"we prove global regularity for Navier-Stokes"* or *"our
  certificate provides evidence for regularity"*. The second is worse because it
  sounds careful.
- Entry: spec 07-02.

**Parent: Yang-Mills existence and mass gap (Clay).**

- Sub-obligation: a **certified spectral gap for one fixed transfer matrix** at
  one lattice spacing in one finite-dimensional truncation.
- Gate: `certified_spectral_gap` returns a `SpectralGapCertificate` whose lower
  bound is strictly positive, with the positivity discharged by the Lean kernel.
- Sealed scope: `continuum_claim = False`, already enforced in
  `omnibias.geometry.gauge.transfer`.
- Never write: *"we prove the Yang-Mills mass gap"*, and equally never *"a
  positive gap at small spacing suggests a continuum gap"* — a positive gap at
  every finite spacing is entirely consistent with a vanishing continuum gap,
  so the suggestion is unfounded.
- Entry: spec 07-04.

**Parent: the Riemann Hypothesis.**

- Sub-obligation: **nothing about zeros**. The legitimate work is tighter
  enclosures of Dirichlet series, `L`-functions and Jacobi theta on `Re(s) > 1`,
  which is where `omnibias.core.verified.dirichlet` is verified.
- Gate: enclosure width against a high-precision reference, plus `100%`
  coverage.
- Sealed scope: **`Re(s) > 1` only.** Analytic continuation is not implemented
  and must not be inferred.
- Never write: any sentence containing "Riemann Hypothesis" and "we" as
  subject. Additionally: **no Padé or Borel tool from spec 03-10 may be applied
  to a Dirichlet series to claim continuation, or to say anything whatsoever
  about zeros.** That step is one line of code and the entire open problem, and
  spec 06-02 requires a dedicated test forbidding it.
- Entry: none. There is no Group 07 spec for RH, deliberately, because there is
  no legitimate sub-obligation the primitives improve. Listing it here as an
  explicit non-entry is the point.

**Parent: P versus NP.**

- Sub-obligation: **certified optimality gaps** for specific instances of
  NP-hard families — a sandwich `lower <= OPT <= upper` proven for one instance.
- Gate: `certify_gap` returns a sound sandwich; the gap is reported, never
  claimed tight.
- Sealed scope: per instance, per size. The relaxations are polynomial-time and
  the bounds are non-tight; that is honest and expected.
- Never write: *"P = NP"*, *"we solve an NP-hard problem in polynomial time"*,
  or *"our relaxation closes the gap"* without the instance and size attached.
- Entry: covered by the existing `omnibias.qubo` / `omnibias.submodular` honesty
  framing; specs 03-01 and 03-03 extend the methods and inherit the framing.

**Parent: turbulence closure (Nobel-adjacent, not a Clay problem).**

- Sub-obligation: a **coarse-graining flow computed rather than fitted** for a
  specific model, with the effective-parameter flow validated against
  direct simulation on a fixed grid pair.
- Gate: relative error of the coarse-grained prediction against the fine
  reference, absolute threshold, five seeds.
- Sealed scope: one model, one scale ratio, one geometry.
- Never write: *"we solve the closure problem"*. Write: *"for this model and
  scale ratio, the computed flow reproduces the fine-grid statistics to X"*.
- Entry: spec 03-07 supplies the method, spec 07-07 the domain framing.

### The escalation ladder, per entry

Each entry climbs the ladder from spec 06-02 independently:

```
empirical gate  ->  sound enclosure  ->  theorem_prover_verified  ->  mathlib_verified
```

Climbing does not shrink the distance to the parent by one step. A
`mathlib_verified` finite obligation is still a finite obligation.

## 5. Worked example

**Testing a proposed sub-obligation against the three tests.**

*Proposal:* "certify that the enstrophy of a 3D Navier-Stokes solution stays
bounded on `[0, T]` for a family of initial data with `||u_0||_{H^1} <= M`."

- Test 1, finite or compact? **Fails.** "A family of initial data" is an
  infinite set, and the certificate machinery encloses one trajectory of one
  discretization. Restricting to a finite sample of initial data passes test 1
  but changes the statement into something much weaker, which is the honest
  version.
- Test 2, gate-decidable? Yes, given a threshold on the enclosed enstrophy.
- Test 3, non-implying? Yes, comfortably — bounded enstrophy on a fixed finite
  horizon for finitely many data says nothing about global regularity.

*Verdict:* rewrite as "for these `N` fixed initial conditions, at this
discretization, the enclosed enstrophy on `[0, T]` stays below `E_max`". That is
a real, checkable, publishable result, and it is much less exciting than the
proposal — which is the correct outcome.

**A proposal that fails test 3.**

*Proposal:* "prove the transfer-matrix gap is bounded below uniformly in the
lattice spacing."

- Test 1: fails as stated (a limit over spacings), but a finite ladder of
  spacings `a in {0.2, 0.1, 0.05}` is finite.
- Test 2: yes.
- Test 3: **this is the interesting failure.** A *uniform* lower bound over all
  spacings, with a proof of uniformity, is very close to the mass-gap
  construction. A finite ladder is not, because three points do not constrain a
  limit. So the finite version passes test 3 and the uniform version would fail
  it — meaning the uniform version, if anyone could prove it, would not be a
  sub-obligation but a substantial part of the parent.

The lesson is that tests 1 and 3 pull in opposite directions, and the ledger's
job is to hold the line where the statement is still finite. Anything that
escapes test 1 usually violates test 3.

## 6. Proposed API

The ledger is a table. The guard is code, and it comes from spec 06-02:

```python
# packages/omnibias-core/tests/test_theory_structure.py
def test_group_07_entries_are_in_the_ledger():
    """Every theory/07-frontier/*.md except the ledger itself names a parent
    that appears in the ledger's table, and the ledger's 'never write' column is
    non-empty for that parent."""

def test_no_rh_entry_exists():
    """There is deliberately no Riemann-Hypothesis frontier spec. If one
    appears, this test fails and the author must justify it against the three
    tests in the ledger."""
```

The second test is unusual and deliberate: it makes adding an RH spec require
deleting a test that explains why it should not exist.

## 7. Practical use cases

1. **Answering "what are you actually attacking"** in one page, honestly.
2. **Triaging a new frontier idea** against the three tests before any work.
3. **Writing a release note** without having to re-derive the claim boundary.
4. **Recording distance to gate.** The ledger tracks how far each entry is from
   its threshold, which is the honest measure of progress.

## 8. Acceptance gates

- **G1 completeness.** Every Group 07 spec's parent appears in the ledger with
  all four fields (sub-obligation, gate, sealed scope, never-write sentence).
  Checked by test.
- **G2 no orphan claims.** Every certificate-emitting module that pins a claim
  flag to `False` is cross-referenced from the ledger. A new such module without
  a ledger entry fails CI.
- **G3 RH non-entry preserved.** `test_no_rh_entry_exists` passes.
- **G4 Padé boundary.** A test asserts no code path applies spec 03-10's Padé or
  Borel machinery to `omnibias.core.verified.dirichlet` outputs.
- **G5 distance recorded.** Each entry records its current best result against
  its gate, updated whenever the corresponding benchmark artifact changes.

## 9. Benchmark plan

Not a benchmark. The deliverable is the ledger, its guards, and a
`docs/frontier-ledger.md` mirror generated from this file so the public site
carries the same boundaries.

## 10. Honesty and scope

- The ledger **constrains claims, not effort**. Every entry is a real research
  target, and `AGENTS.md` is explicit that ambition inside existing packages is
  encouraged.
- Passing every gate in this ledger would solve none of the five parents. That
  is not a defect of the ledger; it is what "external obligation" means.
- The RH non-entry is the most important row, because it is where the primitives
  most tempt an overreach: an enclosure engine plus an extrapolation tool looks
  like a continuation engine and is not.
- Certificate tier: this spec produces none. Entries produce their own, at the
  tiers their specs state.
- No collapse limit appears in this spec.

## 11. Open questions and risks

- **Tests 1 and 3 in tension.** Strengthening a sub-obligation toward the parent
  makes it fail test 3; weakening it toward finiteness makes it less
  interesting. The ledger's value is entirely in holding that line, and the line
  will be pushed on.
- **Gate distance can be discouraging.** The CCF stretch gate sits many orders
  of magnitude below the achieved floor. Recording that honestly is right and
  will look like failure; the alternative is worse.
- **Ledger rot.** Entries updated less often than the code they index become
  misleading. Tying G5 to artifact changes is the mitigation and it is not fully
  automatable.
- **A parent might get solved externally.** If so, the entries do not become
  more valuable; they become historical, and the ledger should say so rather
  than quietly reframing.
- **Falsifier.** If a proposed entry cannot be stated so that it passes all
  three tests, it does not belong in Group 07, and the correct action is to
  delete the spec rather than soften the tests.

## 12. Implementation checklist

- [ ] `docs/frontier-ledger.md` mirroring this file's table
- [ ] `test_group_07_entries_are_in_the_ledger`
- [ ] `test_no_rh_entry_exists` with its explanatory docstring
- [ ] Padé-versus-`dirichlet` boundary test
- [ ] Cross-reference every claim-flag-pinning module from the ledger
- [ ] Distance-to-gate column, updated with artifact changes
- [ ] Cross-reference from `.cursor/rules/frontier-claims.mdc`
- [ ] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parents: all five listed above** — Navier-Stokes global regularity, the
Yang-Mills mass gap, the Riemann Hypothesis, P versus NP, and turbulence
closure.

Each remains an external obligation for the same structural reason, stated
plainly: **every object this repository can certify is finite, and every one of
these problems is a statement about a limit or an infinite class.** A
certificate encloses a specific quantity computed from a specific
finite-dimensional discretization. The parents quantify over all time, all
initial data, all lattice spacings, all instance sizes, or the analytic
continuation of a function beyond the half-plane where its series converges. No
finite collection of finite certificates bridges that, and no tightening of
constants changes the kind of statement being made.

This ledger does not claim, imply, or provide evidence for any of the five.

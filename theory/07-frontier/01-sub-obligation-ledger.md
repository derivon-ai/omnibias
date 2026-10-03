# 07-01 The sub-obligation ledger

## 1. Thesis and status

Famous problems and finite or compact sub-obligations that the new primitives
can actually attack, and — for each — the absolute gate that decides it and the
sentence a sealed certificate does not license (the Never-write column). This
file is the ledger the other frontier specs are entries in.

- **Status**: shipped (G1–G5 earned; Lambda research entry recorded; design record)
- **Depends on**: 01-11, 06-01, 06-02
- **Blocks**: 07-02, 07-03, 07-04, 07-05, 07-06, 07-07, 07-08, 07-09, 07-10, 07-11, 07-12, 07-13

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
- `.cursor/rules/omnibias.md` (Frontier program), `omnibias-frontier` skill.

**No gap.** The flags stay enforced in those modules. This ledger plus
`docs/frontier-ledger.md` is the single index; the distance column is updated
when the cited artifact changes.

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

Clay alternatives (C) and (D) (forced blowup) are resolved externally;
unforced regularity (A)/(B) is still open. The 07-02 enclosure remains a
finite-box sub-obligation of (A)/(B). Spec 07-08 certifies the finite
rational spine of the packet-iteration arithmetic; it does not claim (A)/(B).

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
- Entry: spec 07-02; arithmetic spine in spec 07-08.

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

- Sub-obligation: a rigorous upper-bound program for the de Bruijn–Newman
  constant `Lambda`: certify the finite zero-counting, truncation, and
  approximation steps required by a published reduction, plus its independently
  proved far-field premise. A rectangle calculation alone is not a bound on
  `Lambda`.
- Gate: a replayable certificate of `Lambda <= t0` for a pre-registered
  `t0 < 0.22`, including every finite count, error enclosure, and the cited
  far-field reduction.
- Sealed scope: named `t0`, declared contour cover, declared finite
  approximation, and the stated far-field theorem. The current
  `omnibias.core.verified.dirichlet` surface remains **`Re(s) > 1` only**.
  `omnibias.core.verified.debruijn_newman` encloses `Phi(u)` and `H_t` on one
  finite rectangle and records a named `Lambda <= 0.2` attempt that stays
  `certified=False` because the cited far-field premise (and a genuine finite
  contour cover) remain external obligations. That attempt does not earn the
  ledger gate.
- Never write: *"we prove / disprove the Riemann Hypothesis"* or infer it from
  a bound on `Lambda`. Additionally: **no Padé or Borel tool from spec 03-10
  may be applied to a Dirichlet series to claim continuation.**
- Entry: planned Lambda program; no Group 07 implementation or dedicated spec
  has been shipped.

**Parent: the Twin Prime Conjecture.**

- Sub-obligation: exact shifted-prime local-factor identities, coefficientwise
  FI equation (3.1), a 3-D log-polyhedral terminal-shell atlas isolating the
  still-required Möbius bilinear estimate, and source-valid finite
  variational certificates for bounded-gap research.
- Gate: A0 exact local factors; A1a finite combinatorial replay; A1b0 exact
  dyadic/rho/term-routing subchecks; A1b1 exact signed fixed-\(2\) determinant
  replay; A1 complete Friedlander–Iwaniec reduction replay; A2g exact
  terminal-shell geometry; A2k exact kernel-cell credit; A2d a loss-budgeted
  fixed-\(2\) determinant completion; A2 source-complete strict analytic
  reduction; A3 one new uniform saving on a remaining cell; A4 every analytic
  cell and asymptotic passage discharged; BG0 an exact \(k=40\to39\)
  input-manifest replay; BG1 a source-valid \(k=39\) generalized-Rayleigh
  crossing for the diameter-182 tuple.
- Sealed scope: finite rational identities, a declared finite cell atlas, or
  one finite variational witness. The current implementation earns A0, A1a,
  A1b0, and A2g: FI (3.1) is replayed coefficientwise through \(n=128\);
  dyadic selection, finite rho insertion, all 13 routing leaves, and the signed
  determinant transform are checked; the fixed rational shell is exactly
  \(1/38\) of the classical log-volume; and the candidate kernel cell has exact
  credit \(41/640\). The completion lemma is absent. BG0 reproduces the pinned
  97-component/149-form source manifest with dimensions 39/38. The authenticated
  split evaluator completed and independently replayed the full \(k=40\)
  baseline above \(1/50000\); equivalence to the unavailable corrected-FLINT
  build is not claimed. No \(k=39\) candidate crossing is known. A2d, A1–A4,
  and BG1 remain blocked.
- Never write: *"we prove the Twin Prime Conjecture"* from a finite prefix,
  finite-scale cancellation, or a first-moment bounded-gap certificate.
- Entry: `omnibias.holonomic.twin_prime`;
  `docs/benchmarks/twin_prime_sieve_smoke.json`;
  `docs/benchmarks/twin_prime_bounded_gap_smoke.json`.

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

**Parent: finite-time singularity of 3D Euler / Navier-Stokes (Clay negative
side).**

Unforced 3D Euler blowup on `R^3` is now proved. The CCF residual
sub-obligation is no longer novel against that resolved parent; the stretch
gate is retained as a recorded empirical floor.

- Sub-obligation: a **numerically minimized residual** of the one-dimensional
  CCF model on a fixed compactified grid and a fixed dictionary. CCF is a
  proxy, not the Euler or Navier-Stokes system.
- Gate: `ccf_absolute_gates` stretch threshold `1e-13`.
- Sealed scope: one model equation, one grid, one dictionary;
  `navier_stokes_proof_claim = False`.
- Never write: *"our CCF residual is evidence for Euler or Navier-Stokes
  blowup"*.
- Entry: spec 07-03.

**Parent: computer-assisted global dynamical structure.**

- Sub-obligation: a **finite-horizon jet Lohner enclosure** of finitely many
  trajectories from a finite initial box, with a named width budget.
- Gate: the 07-06 width-budget / validated-orbit gates.
- Sealed scope: finite boxes, finite time, finite jets. Not an attractor,
  not structural stability, not Smale's 14th problem.
- Never write: *"we prove the system is chaotic"* or *"the attractor exists"*.
- Entry: spec 07-06.

**Parent: Nobel-adjacent scientific discovery (many-body, fusion, materials).**

- Sub-obligation: **named numerical tools** with a validated baseline on a
  fixed model (exact oscillator ladder, Harris-layer residual, exact
  `dT/dθ`). The parents are not tool problems.
- Gate: the 07-07 domain-program smoke gates against the named baseline.
- Sealed scope: one model, one geometry, one empirical comparison.
- Never write: *"we solved the many-body problem"*, *"fusion energy gain"*,
  or *"we inverse-designed an arbitrary material"*.
- Entry: spec 07-07.

### Ledger table

| Parent key | Parent | Sub-obligation | Gate | Sealed scope | Never write | Entry | Distance |
|---|---|---|---|---|---|---|---|
| NS | Navier-Stokes global regularity (Clay) | sound residual enclosure on one box and horizon | `require_enclosure_coverage` at 100% plus a named residual floor | one discretization, one box, one horizon | we prove global regularity for Navier-Stokes | 07-02, 07-08, 07-09, 07-10, 07-11, 07-12, 07-13, 07-14, 07-17, 07-18, 07-19, 07-20, 07-21, 07-22, 07-23 | `docs/benchmarks/ns_weak_form_enclosure_smoke.json` (`all_passed`; width split recorded; Clay (C)/(D) resolved externally, unforced (A)/(B) still open; continuum claim false); `docs/benchmarks/convergence_ledger_smoke.json`; `docs/benchmarks/anisotropic_profile_smoke.json`; `docs/benchmarks/stress_cone_smoke.json`; `docs/benchmarks/weighted_class_smoke.json`; `docs/benchmarks/swirl_heat_pulse_smoke.json`; `docs/benchmarks/forced_flat_blowup_smoke.json`; `docs/benchmarks/ns_core_search_smoke.json`; `docs/benchmarks/pulse_family_composition_smoke.json`; `docs/benchmarks/unforced_bkm_slab_smoke.json`; `docs/benchmarks/unforced_slab_continuation_smoke.json`; `docs/benchmarks/force_is_essential_smoke.json`; `docs/benchmarks/unforced_abc_slab_smoke.json`; `docs/benchmarks/unforced_tg3d_ic_smoke.json`; `docs/benchmarks/unforced_abc_long_chain_smoke.json` |
| EULER | finite-time singularity of 3D Euler / Navier-Stokes | CCF residual on a fixed grid and dictionary | `ccf_absolute_gates` stretch `1e-13` | one model equation; not Euler/NS | our CCF residual is evidence for Euler or Navier-Stokes blowup | 07-03, 07-15, 07-16 | `docs/benchmarks/reproduce_deepmind_ccf_smoke.json` (stretch unearned; leftover #55 Hardy N>0 dictionary; unforced Euler blowup on R^3 resolved; CCF residual no longer novel against that parent); `docs/benchmarks/ipm_remainder_cap_smoke.json`; `docs/benchmarks/boussinesq_remainder_cap_smoke.json` |
| H16 | Hilbert's sixteenth problem (selected algebraic scheme and infinitesimal limit-cycle register) | exact projective patchwork/regular-height search for the open 22-oval octic; genuine mixed Abelian count and a finite rational coefficient-box bound; exact-Q Poincare compactification and finite Dulac-model bounds; finite LN/exp obstruction localized on quadratic G1; an exact-Q Groebner basis engine; Poincare-Lyapunov focal-value/Bautin-ideal computation on a declared normal-form family; resonant Poincare-Dulac normal forms with a derived (not declared) first-order Dulac corner map; sound collar-membership agreement with a genuine unique-cycle proof; a machine-checked H16 obligation ledger with derived parent flags | GP1–GP4 replay and GP5 requires a direct 22-oval coefficient certificate; GA1–GA8 require the mixed two-zero count and exact box tiling; GD1–GD6 require chart identities, rational/resonant model bounds, an interval-ratio cover, finite Lean replay, and open-case refusal; GL1–GL7 preserve finite-chain honesty without flipping G1; GF1–GF7 require Groebner reduced-basis/cofactor replay, focal-value round-trip, Bautin basis-length-3 literature replay, normal-form round-trip, corner-derivation cross-check, collar-membership proof, and `GraphicTarget` wiring; ledger GH1–GH4 require the shipped ledger's open state, certificate round-trip, tamper rejection, and a synthetic full-discharge genuinely earning the parent flags | patchwork search is incomplete and GP5 is false; Abelian uniformity is only \(\beta_0\in[9/10000,11/10000]\) for one cubic and contour; Dulac bounds apply only to declared finite models without physical return-map membership or uniform remainders; physical bounded-format membership, singular first-hit, C2 remainders, and overlap matching remain open; the Bautin ideal is for one declared normal-form family, not the actual singular return map of an arbitrary graphic; the derived corner map is first-order only, not exponentiated; collar membership is sound only away from the corner | we prove the 22-oval target exists; we prove H(n) is finite; we prove finite cyclicity of a Roussarie graphic; the Hilbert-16 obligation ledger is solved | exact-Q patchwork/search track; mixed-period finite coefficient cover; finite Dulac-model track; coalescing-passage LN/exp negative assessment; focal/Bautin/normal-form/membership/ledger track | `docs/benchmarks/patchwork_octic_smoke.json` (GP1–GP4 pass; `search_incomplete`; no target hit, GP5 false); `docs/benchmarks/abelian_zero_count_smoke.json` (GA1–GA8 pass; genuine \(\beta\ne0\) count 2; finite coefficient-box bound 2; irrationality unclaimed); `docs/benchmarks/dulac_cyclicity_smoke.json` (GD1–GD6 pass; model bounds 2/2/0; physical membership, remainder, DRR closure, and full H16 false); `docs/benchmarks/hilbert16_ln_passage_smoke.json` (`all_passed` negative assessment; corrected kill A has unbounded matching W-ratio, proposed kill-B radius is negative, `g1_passed=false`; H(2) and H(n) remain unclaimed); `docs/benchmarks/hilbert16_focal_smoke.json` (GF1–GF7 pass; Groebner/focal/Bautin/normal-form/corner/collar/wiring all replay; `bautin_ideal_stabilization_proved` and `physical_return_membership_proved` false); `docs/benchmarks/hilbert16_ledger_smoke.json` (GH1–GH4 pass; every derived parent flag false; a synthetic fully-discharged ledger genuinely earns `full_hilbert16_solved`) |
| YM | Yang-Mills existence and mass gap (Clay) | certified gap of one fixed transfer matrix | `certified_spectral_gap` strictly positive | `continuum_claim = False` | we prove the Yang-Mills mass gap | 07-04, 07-05, 07-08 | `docs/benchmarks/gauge_holonomy_gap_smoke.json` (`all_passed`; `mass_gap: false`); trial factor leftover-recorded on two-plaquette / strip |
| RH | the Riemann Hypothesis | rigorous de Bruijn–Newman `Lambda` upper-bound program via a published finite reduction | replayable `Lambda <= t0` certificate for pre-registered `t0 < 0.22` | named contour cover, finite approximation, and proved far-field premise; not implemented | we prove / disprove the Riemann Hypothesis | planned Lambda program | no implementation; `docs/benchmarks/dirichlet_enclosure_smoke.json` remains `Re(s)>1` only |
| TWIN | Twin Prime Conjecture | exact shifted-prime local factors, coefficientwise FI (3.1), exact dyadic/rho/term routing, a signed fixed-\(2\) determinant transform, a 3-D terminal atlas, and a parameterized \(k=39\), diameter-182 target | A0 local factors; A1a combinatorial replay; A1b0 structural subchecks; A1b1 fixed-\(2\) determinant replay; A1 full replay; A2g terminal geometry; A2k kernel-cell credit; A2d loss-budgeted determinant completion; A2 source-complete reduction; A3 one new uniform cell saving; A4 all cells close; BG0 exact \(k=40\to39\) input-manifest replay; BG1 source-valid \(k=39\) crossing | A0/A1a/A1b0/A1b1/A2g/A2k/BG0 finite arithmetic and the complete authenticated-split \(k=40\) baseline replay earned; equivalence to the unavailable corrected-FLINT build, A2d, A1–A4, and BG1 remain blocked | we prove the Twin Prime Conjecture | `omnibias.holonomic.twin_prime` | `docs/benchmarks/twin_prime_sieve_smoke.json`; `docs/benchmarks/twin_prime_bounded_gap_smoke.json`; `docs/benchmarks/twin_prime_k40_numerical_receipt_sealed.json` (finite artifacts only; parent and analytic estimates false) |
| PNP | P versus NP | certified optimality gap on one instance | `certify_gap` sandwich, never claimed tight | per instance, per size | P = NP | qubo / discrete; 03-01, 03-03 | `docs/benchmarks/instance_gap_tightening_smoke.json` (`all_passed`; never tight) |
| TURB | turbulence closure (Nobel-adjacent) | computed coarse-graining vs fine reference | relative error, absolute threshold, five seeds | one model, one scale ratio, one geometry | we solve the closure problem | 03-07, 07-07 | `docs/benchmarks/scale_flow_smoke.json` (`all_passed`) |
| DYN | computer-assisted global dynamical structure | finite-horizon jet Lohner on a finite box | 07-06 width-budget / orbit gates | finite boxes and time | we prove the system is chaotic | 07-06 | `docs/benchmarks/validated_dynamics_smoke.json` (`all_passed`; attractor claim false; `seal_run` emits `diagnose_width`) |
| NOBEL | Nobel-adjacent scientific discovery | named tools vs a validated baseline | 07-07 domain-program smoke | one model, one geometry | we solved the many-body problem | 07-07 | `docs/benchmarks/plasma_resistive_layer_smoke.json` (`all_passed`; tooling, not a discovery) |

### Claim-flag modules

Every certificate-emitting module that pins a parent claim flag to `False`
is listed here so a new site cannot appear without a ledger row.

| Module | Flag | Parent key |
|---|---|---|
| `omnibias.pinn.certified.navier_stokes` | `continuum_navier_stokes_claim` | NS |
| `omnibias.pinn.certified.machine` | `continuum_navier_stokes_claim` | NS |
| `omnibias.geometry.gauge.transfer` | `continuum_claim` | YM |
| `omnibias.core.verified.dirichlet` | `Re(s) > 1` scope | RH |
| `omnibias.core.verified.debruijn_newman` | `rh_claim = False`; named `Lambda` attempt unearned | RH |
| `omnibias.holonomic.twin_prime` | `twin_prime_conjecture_proof_claim = False`; asymptotic premises unearned | TWIN |
| `omnibias.core.verified.eig_operator` | `certified_spectral_gap` | YM |
| `omnibias.sos` | positivity honesty | YM |
| `omnibias.qubo` / `omnibias.discrete` | `certify_gap` | PNP |
| `benchmarks/_gates.py` | `navier_stokes_proof_claim` | NS, EULER |
| `omnibias.core.proof.obligations.convergence_ledger` | `navier_stokes_proof_claim` / `yang_mills_mass_gap_claim` | NS, YM |
| `omnibias.pinn.certified.anisotropic` | `navier_stokes_proof_claim` / `forced_blowup_reproof_claim` | NS |
| `omnibias.core.proof.obligations.stress_cone` | `navier_stokes_proof_claim` / `forced_blowup_reproof_claim` | NS |
| `omnibias.core.verified.weighted_class` | `gevrey_class_claim` / `navier_stokes_proof_claim` | NS |
| `omnibias.core.verified.swirl_heat` | `navier_stokes_proof_claim` / `three_d_heat_theorem` | NS |
| `omnibias.core.pulse_envelope` | `navier_stokes_proof_claim` / `forced_blowup_reproof_claim` | NS |
| `omnibias.pinn.certified.forced_flat` | `navier_stokes_proof_claim` / `forced_blowup_reproof_claim` | NS |
| `omnibias.pinn.certified.unforced` | `navier_stokes_proof_claim` / `continuum_navier_stokes_claim` | NS |

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

def test_rh_entry_is_lambda_scoped():
    """The RH ledger row is limited to a de Bruijn–Newman Lambda program,
    never an assertion that the parent is proved or disproved."""
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
- **G3 Lambda scope preserved.** `test_rh_entry_is_lambda_scoped` passes.
- **G4 Padé boundary.** A test asserts no code path applies spec 03-10's Padé or
  Borel machinery to `omnibias.core.verified.dirichlet` outputs.
- **G5 distance recorded.** Each entry records its current best result against
  its gate, updated whenever the corresponding benchmark artifact changes.

## 9. Benchmark plan

Not a benchmark. The deliverable is the ledger, its guards, and a
`docs/frontier-ledger.md` mirror generated from this file so the public site
carries the same boundaries.

## 10. Honesty and scope

- The ledger's Never-write column is what a sealed certificate does not
  license today, not an order that an agent must not decide a parent is
  solved. Every entry is a real research target.
- Passing every gate in this ledger does not, by itself, flip a parent
  honesty flag. An agent may still judge a parent solved; the flags stay
  earned.
- The RH Lambda entry is deliberately a research target, not an RH claim.
  An enclosure engine plus an extrapolation tool is not a continuation engine;
  the published reduction and its far-field premise must be independently
  established before a Lambda bound is reported.
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

- [x] `docs/frontier-ledger.md` mirroring this file's table
- [x] `test_group_07_entries_are_in_the_ledger`
- [x] `test_rh_entry_is_lambda_scoped` with its explanatory docstring
- [x] Padé-versus-`dirichlet` boundary test
- [x] Cross-reference every claim-flag-pinning module from the ledger
- [x] Distance-to-gate column, updated with artifact changes
- [x] Cross-reference from `.cursor/rules/omnibias.md` (Frontier program)
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parents: all listed above**, including Navier-Stokes global regularity, the
Yang-Mills mass gap, the Riemann Hypothesis, the Twin Prime Conjecture,
P versus NP, and turbulence closure.

Each remains an external obligation for the same structural reason, stated
plainly: **every object this repository can certify is finite, and every one of
these problems is a statement about a limit or an infinite class.** A
certificate encloses a specific quantity computed from a specific
finite-dimensional discretization. The parents quantify over all time, all
initial data, all lattice spacings, all instance sizes, or the analytic
continuation of a function beyond the half-plane where its series converges. No
finite collection of finite certificates bridges that, and no tightening of
constants changes the kind of statement being made.

This ledger does not claim, imply, or provide evidence for any parent.

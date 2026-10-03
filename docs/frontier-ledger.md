# Frontier sub-obligation ledger

This page is the public copy of theory spec 07-01. Every row is a
**finite or compact** obligation with an absolute gate. The Never-write
column is what a sealed certificate does not license; it is not an
order that an agent must not decide a parent is solved. Status is
**shipped**.

The four claim rungs are on [the honesty page](honesty.md). Runtime
flags stay earned.

## Ledger

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

The Riemann Hypothesis row records a planned de Bruijn–Newman `Lambda`
upper-bound program, not an RH claim. It requires a published finite reduction
and an independently proved far-field premise; the current Dirichlet machinery
remains restricted to `Re(s) > 1`. A finite `Phi` / `H_t` enclosure primitive
exists; the named `t0 = 0.2` attempt stays `certified=False` because the
far-field premise is an external obligation. Padé / Borel (spec 03-10) must not
be applied to a Dirichlet series. That attempt does not earn the ledger gate.

Clay alternatives (C) and (D) (forced Navier–Stokes blowup) are resolved
externally; unforced regularity (A)/(B) is still open. Unforced 3D Euler
blowup on `R^3` is now proved, so a CCF residual hit is no longer novel
against that resolved parent. Spec 07-08 certifies the finite rational
spine of both the NS exponent ledger and the YM polymer majorants; it
does not claim (A)/(B) or the mass gap.

## Claim-flag modules

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

Source: [theory/07-frontier/01-sub-obligation-ledger.md](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/01-sub-obligation-ledger.md).

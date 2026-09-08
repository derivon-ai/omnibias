# Frontier sub-obligation ledger

This page is the public copy of theory spec 07-01. Every row is a
**finite or compact** obligation with an absolute gate. Passing every
row would solve none of the parents. That is what "external
obligation" means. Status is **shipped**.

The four claim rungs are on [the honesty page](honesty.md). Climbing a
rung does not shrink the distance to a parent.

## Ledger

| Parent key | Parent | Sub-obligation | Gate | Sealed scope | Never write | Entry | Distance |
|---|---|---|---|---|---|---|---|
| NS | Navier-Stokes global regularity (Clay) | sound residual enclosure on one box and horizon | `require_enclosure_coverage` at 100% plus a named residual floor | one discretization, one box, one horizon | we prove global regularity for Navier-Stokes | 07-02, 07-08, 07-09, 07-10, 07-11, 07-12 | `docs/benchmarks/ns_weak_form_enclosure_smoke.json` (`all_passed`; width split recorded; Clay (C)/(D) resolved externally, unforced (A)/(B) still open; continuum claim false); `docs/benchmarks/convergence_ledger_smoke.json`; `docs/benchmarks/anisotropic_profile_smoke.json`; `docs/benchmarks/stress_cone_smoke.json`; `docs/benchmarks/weighted_class_smoke.json`; `docs/benchmarks/swirl_heat_pulse_smoke.json` |
| EULER | finite-time singularity of 3D Euler / Navier-Stokes | CCF residual on a fixed grid and dictionary | `ccf_absolute_gates` stretch `1e-13` | one model equation; not Euler/NS | our CCF residual is evidence for Euler or Navier-Stokes blowup | 07-03 | `docs/benchmarks/reproduce_deepmind_ccf_smoke.json` (stretch unearned; unforced Euler blowup on R^3 resolved; CCF residual no longer novel against that parent) |
| YM | Yang-Mills existence and mass gap (Clay) | certified gap of one fixed transfer matrix | `certified_spectral_gap` strictly positive | `continuum_claim = False` | we prove the Yang-Mills mass gap | 07-04, 07-05, 07-08 | `docs/benchmarks/gauge_holonomy_gap_smoke.json` (`all_passed`; `mass_gap: false`); trial factor leftover-recorded on two-plaquette / strip |
| RH | the Riemann Hypothesis | rigorous de Bruijn–Newman `Lambda` upper-bound program via a published finite reduction | replayable `Lambda <= t0` certificate for pre-registered `t0 < 0.22` | named contour cover, finite approximation, and proved far-field premise; not implemented | we prove / disprove the Riemann Hypothesis | planned Lambda program | no implementation; `docs/benchmarks/dirichlet_enclosure_smoke.json` remains `Re(s)>1` only |
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

Source: [theory/07-frontier/01-sub-obligation-ledger.md](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/01-sub-obligation-ledger.md).

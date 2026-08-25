# Frontier sub-obligation ledger

This page is the public copy of theory spec 07-01. Every row is a
**finite or compact** obligation with an absolute gate. Passing every
row would solve none of the parents. That is what "external
obligation" means.

The four claim rungs are on [the honesty page](honesty.md). Climbing a
rung does not shrink the distance to a parent.

## Ledger

| Parent key | Parent | Sub-obligation | Gate | Sealed scope | Never write | Entry | Distance |
|---|---|---|---|---|---|---|---|
| NS | Navier-Stokes global regularity (Clay) | sound residual enclosure on one box and horizon | `require_enclosure_coverage` at 100% plus a named residual floor | one discretization, one box, one horizon | we prove global regularity for Navier-Stokes | 07-02 | `docs/benchmarks/ns_weak_form_enclosure_smoke.json` (`all_passed`; width split recorded; continuum claim false) |
| EULER | finite-time singularity of 3D Euler / Navier-Stokes | CCF residual on a fixed grid and dictionary | `ccf_absolute_gates` stretch `1e-13` | one model equation; not Euler/NS | our CCF residual is evidence for Euler or Navier-Stokes blowup | 07-03 | `docs/benchmarks/reproduce_deepmind_ccf_smoke.json` (stretch unearned) |
| YM | Yang-Mills existence and mass gap (Clay) | certified gap of one fixed transfer matrix | `certified_spectral_gap` strictly positive | `continuum_claim = False` | we prove the Yang-Mills mass gap | 07-04, 07-05 | `docs/benchmarks/gauge_holonomy_gap_smoke.json` (`all_passed`; `mass_gap: false`) |
| RH | the Riemann Hypothesis | nothing about zeros; Dirichlet enclosures on `Re(s) > 1` | enclosure width plus 100% coverage | `Re(s) > 1` only | any sentence with Riemann Hypothesis and we as subject | none | non-entry; no Group 07 spec |
| PNP | P versus NP | certified optimality gap on one instance | `certify_gap` sandwich, never claimed tight | per instance, per size | P = NP | qubo / discrete; 03-01, 03-03 | method gates; no parent claim |
| TURB | turbulence closure (Nobel-adjacent) | computed coarse-graining vs fine reference | relative error, absolute threshold, five seeds | one model, one scale ratio, one geometry | we solve the closure problem | 03-07, 07-07 | `docs/benchmarks/scale_flow_smoke.json` (`all_passed`) |
| DYN | computer-assisted global dynamical structure | finite-horizon jet Lohner on a finite box | 07-06 width-budget / orbit gates | finite boxes and time | we prove the system is chaotic | 07-06 | `docs/benchmarks/validated_dynamics_smoke.json` (`all_passed`; attractor claim false) |
| NOBEL | Nobel-adjacent scientific discovery | named tools vs a validated baseline | 07-07 domain-program smoke | one model, one geometry | we solved the many-body problem | 07-07 | `docs/benchmarks/plasma_resistive_layer_smoke.json` (`all_passed`; tooling, not a discovery) |

The Riemann Hypothesis row is a **non-entry**. There is no Group 07 spec
for it, because there is no legitimate sub-obligation the primitives
improve. Padé / Borel (spec 03-10) must not be applied to a Dirichlet
series.

## Claim-flag modules

| Module | Flag | Parent key |
|---|---|---|
| `omnibias.pinn.certified.navier_stokes` | `continuum_navier_stokes_claim` | NS |
| `omnibias.pinn.certified.machine` | `continuum_navier_stokes_claim` | NS |
| `omnibias.geometry.gauge.transfer` | `continuum_claim` | YM |
| `omnibias.core.verified.dirichlet` | `Re(s) > 1` scope | RH |
| `omnibias.core.verified.eig_operator` | `certified_spectral_gap` | YM |
| `omnibias.sos` | positivity honesty | YM |
| `omnibias.qubo` / `omnibias.discrete` | `certify_gap` | PNP |
| `benchmarks/_gates.py` | `navier_stokes_proof_claim` | NS, EULER |

Source: [theory/07-frontier/01-sub-obligation-ledger.md](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/01-sub-obligation-ledger.md).

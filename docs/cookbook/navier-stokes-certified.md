# Navier-Stokes Proof Prep

This page is the global-regularity-grade counterpart to the CCF CAP pipeline. It does **not**
claim to solve the Navier-Stokes millennium problem. It gives the repository a
falsifiable substrate for generating, exporting, independently recomputing, and
eventually certifying Navier-Stokes proof candidates.

!!! info "Claim boundary"
    Every artifact on this page hard-wires `unproven_claim = False`. The exact rules
    — what `unproven_claim`, `interval_verified`, `finite_energy_verified`, and the
    certified-evidence gates mean, and what is structurally required to flip them —
    live in [Scope & guarantees § 3](../scope-and-guarantees.md#3-the-certified-pde-stacks-navierstokes-ccf).

## What Shipped

The backend-neutral module `omnibias.pinn.certified.navier_stokes` provides:

- theorem contracts for the global-regularity and finite-time blow-up routes;
- machine-readable honesty labels (`unproven_claim`, `interval_verified`,
  `theorem_prover_verified`, `finite_energy_verified`);
- periodic spectral primitive residuals, divergence, curl, Laplacian, pressure
  Poisson checks, and Leray projection;
- energy, enstrophy, palinstrophy, divergence, pressure, and BKM-style vorticity
  diagnostics;
- a Navier-Stokes CAP bundle schema with proof obligations.
- a 3D manufactured ABC-flow benchmark and conservative candidate-upgrade gates.

The numpy-only module `omnibias.symbolic.navier_stokes` independently recomputes
CAP residuals and exposes discovery scaffolds for regularity growth laws and
self-similar blow-up-rate fits. It intentionally does not import the PINN
certified-evidence module.

```python
import numpy as np

from omnibias.pinn.certified import build_ns_cap_bundle, ns_cap_schema_errors
from omnibias.symbolic.navier_stokes import verify_ns_cap_bundle

n = 64
nu = 0.1
x = 2 * np.pi * np.arange(n) / n
X, Y = np.meshgrid(x, x, indexing="ij")

velocity = np.stack([
    np.sin(X) * np.cos(Y),
    -np.cos(X) * np.sin(Y),
])
pressure = 0.25 * (np.cos(2 * X) + np.cos(2 * Y))
velocity_t = -2 * nu * velocity

bundle = build_ns_cap_bundle(
    velocity,
    pressure,
    velocity_t=velocity_t,
    viscosity=nu,
)
assert ns_cap_schema_errors(bundle) == []
assert verify_ns_cap_bundle(bundle)["residual_samples_match"]
```

## 3D Manufactured Benchmark

The first 3D benchmark is a steady Beltrami ABC flow:

\[
u = a(\sin z + \cos y,\; \sin x + \cos z,\; \sin y + \cos x).
\]

It satisfies \(\nabla \times u = u\) and \(\Delta u = -u\). Therefore the
primitive equation is exactly manufactured by

\[
p = -\rho |u|^2/2,\qquad f = \nu u,\qquad u_t=0.
\]

```python
from omnibias.pinn.certified import build_ns_cap_bundle, manufactured_abc_flow
from omnibias.symbolic.navier_stokes import verify_ns_cap_bundle

mms = manufactured_abc_flow(24, viscosity=0.07)
bundle = build_ns_cap_bundle(
    mms["velocity"],
    mms["pressure"],
    velocity_t=mms["velocity_t"],
    forcing=mms["forcing"],
    viscosity=mms["viscosity"],
    lengths=mms["lengths"],
)
assert verify_ns_cap_bundle(bundle)["residual_samples_match"]
```

This validates 3D divergence, pressure-Poisson consistency, forcing, and
independent recomputation. It is still a periodic manufactured flow, not a
finite-energy `R^3` global-regularity candidate.

## The Exact-Flow Baseline

The first validation target is the 2D periodic Taylor-Green vortex at one time
slice:

\[
u = \sin x\cos y,\qquad v = -\cos x\sin y,\qquad
p = \tfrac14(\cos 2x + \cos 2y),\qquad u_t=-2\nu u.
\]

On the periodic spectral grid this gives near-machine precision checks for:

- primitive momentum residual;
- incompressibility residual;
- pressure Poisson consistency;
- energy/enstrophy/palinstrophy diagnostics;
- independent CAP recomputation through `omnibias-symbolic`.

This validates the machinery. It does not solve the 3D theorem.

## Proof Obligations

Every CAP bundle includes open obligations. These are the items an interval
verifier or proof assistant must discharge before a candidate can be promoted.

For both regularity routes:

- `divergence_free`: prove or interval-verify `div u = 0`;
- `primitive_residual`: bound the Navier-Stokes residual in a certified norm;
- `pressure_poisson`: verify pressure consistency;
- `tail_bounds`: bound spectral, Chebyshev, or compactification tails.

For global regularity:

- `continuation_criterion`: show the discovered estimate implies a standard
  continuation criterion;
- `a_priori_estimate`: prove the inequality for all smooth finite-energy data.

For blow-up:

- `linearized_invertibility`: certify a true solution near the numerical profile;
- `finite_energy_initial_data`: verify smooth finite-energy initial data on
  `R^3`;
- `norm_divergence`: prove a critical norm diverges in finite time.

The schema rejects `unproven_claim=True` unless interval, theorem-prover, and
finite-energy verification flags are all set.

## Compactified Domain Fields And Gates

`domain.type="compactified_r3"` now requires concrete metadata:

- `compactification`: map name/formula, coordinate names, Jacobian-weight
  convention, and basis;
- `tail_bounds`: retained modes plus coefficient-tail bounds;
- `finite_energy_checks`: explicit energy/tail-energy bounds.

Candidate promotion is deliberately conservative:

- `numerical_artifact`: schema incomplete, residual too large, or no independent
  recomputation;
- `cap_candidate`: schema clean, residuals below threshold, independent
  recomputation passes;
- `interval_obligation_ready`: tail bounds and finite-energy checks are present;
- `proof_assistant_obligation_ready`: interval verification is present but the
  theorem-prover obligations remain open;
- `externally_verified_artifact`: all external verification flags are present.

The gate result always carries `unproven_claim=False`; a true global-regularity claim requires
external review and theorem-level proof, not just a passing schema.

## Candidate Bridge

The bridge layer produces replayable candidate artifacts separate from CAP proof
bundles. A candidate artifact stores compactified coefficient summaries,
deterministic replay-grid metadata, the exact scalar traces needed for replay,
honesty flags, and open proof obligations. It is designed to be falsifiable by a
second code path before any interval arithmetic is attempted.

Regularity-growth candidates wrap the symbolic sparse growth-law search:

```python
import numpy as np

from omnibias.symbolic.navier_stokes import (
    build_regularity_candidate_artifact,
    replay_candidate_artifact,
)

time = np.linspace(0.0, 1.0, 200)
enstrophy = np.exp(time)
artifact = build_regularity_candidate_artifact(
    time,
    {
        "energy": 0.5 * enstrophy,
        "enstrophy": enstrophy,
        "palinstrophy": 2.0 * enstrophy,
        "bkm_vorticity_proxy": np.sqrt(enstrophy),
    },
    target="enstrophy",
    include_quadratic=False,
)
assert replay_candidate_artifact(artifact)["replay_match"]
assert artifact["honesty"]["unproven_claim"] is False
```

Blow-up-rate candidates record the ansatz class, norm trace, fitted rate,
compactified coefficient summaries, residual metrics, and the missing proof
obligations:

```python
import numpy as np
from dataclasses import asdict

from omnibias.pinn.certified import (
    compactified_coefficient_set,
    compactified_sandbox_replay_grid,
)
from omnibias.symbolic.navier_stokes import (
    build_blowup_candidate_artifact,
    replay_candidate_artifact,
)

time = np.linspace(0.0, 0.9, 80)
norm = 2.0 * (1.0 - time) ** -0.75
grid = compactified_sandbox_replay_grid(n_radial=4, n_theta=4, n_phi=8)
coeffs = compactified_coefficient_set(
    "u",
    np.ones((3, 4)),
    tail_l1_bound=1e-10,
    finite_energy_estimate=3.0,
)
artifact = build_blowup_candidate_artifact(
    time,
    norm,
    blowup_time=1.0,
    ansatz_metadata={"class": "axisymmetric_swirl_sandbox"},
    residual_metrics={"max_abs_residual": 1e-6},
    replay_grid=asdict(grid),
    coefficients=[asdict(coeffs)],
)
assert replay_candidate_artifact(artifact)["replay_match"]
assert artifact["honesty"]["finite_time_blowup_claim"] is False
```

Passing replay means the artifact was serialized honestly enough for a second
implementation to reproduce the reported fit. It does not prove an inequality,
certify a solution profile, or establish finite-energy `R^3` data.

## Axisymmetric Candidate Sprint

The first serious 3D candidate family beyond periodic manufactured flows is an
axisymmetric-with-swirl sandbox. It reduces the search to a compactified
meridional grid while keeping the three cylindrical velocity components
`(u_r, u_theta, u_z)` and the vortex-stretching structure that is absent in 2D.

The certified-evidence side represents velocity by streamfunction, swirl, and pressure:

\[
u_r = -r^{-1}\partial_z\psi,\qquad
u_\theta=\gamma,\qquad
u_z = r^{-1}\partial_r\psi.
\]

This makes incompressibility automatic away from the axis, but it introduces
explicit open obligations at `r=0`: axis regularity, parity, compactified tail
bounds, finite energy, and either linearized invertibility or a real a priori
estimate.

```python
import json

from omnibias.pinn.certified import (
    build_axisymmetric_swirl_candidate_artifact,
    candidate_artifact_schema_errors,
)
from omnibias.symbolic.navier_stokes import replay_candidate_artifact

artifact = build_axisymmetric_swirl_candidate_artifact(
    seed=11,
    n_radial=6,
    n_axial=7,
    viscosity=0.01,
)
assert candidate_artifact_schema_errors(artifact) == []

reloaded = json.loads(json.dumps(artifact))
report = replay_candidate_artifact(reloaded)
assert report["replay_match"]
assert artifact["honesty"]["unproven_claim"] is False
```

The emitted artifact includes compactified meridional replay-grid metadata,
sampled streamfunction/swirl/pressure payloads, coefficient summaries,
uncertified tail placeholders, finite-energy estimates, cylindrical residual
diagnostics, axis-regularity obligations, and independent NumPy replay.

This strengthens the pipeline from generic trace artifacts to a concrete
symmetry-reduced 3D candidate family. It still is not a proof: the current
runner is deterministic and falsifiable, not optimized or interval-certified.
`build_axisymmetric_candidate_bridge_artifacts()` can also wrap the same
axisymmetric artifact with companion regularity-growth and blow-up-rate trace
artifacts, both replayable through `omnibias-symbolic`.

## Axisymmetric Refiner

The refiner turns the sandbox into a coefficient-defined candidate family. It
uses a compact polynomial meridional basis with built-in axis behavior:
`psi ~ r^2`, `swirl ~ r`, and pressure represented independently. A bounded
deterministic coordinate refiner then reduces the sampled cylindrical residual
on a train grid and reports the same coefficients on a separate holdout grid.

```python
import json

from omnibias.pinn.certified import (
    build_refined_axisymmetric_swirl_candidate_artifact,
    candidate_artifact_schema_errors,
)
from omnibias.symbolic.navier_stokes import replay_candidate_artifact

artifact = build_refined_axisymmetric_swirl_candidate_artifact(
    seed=17,
    n_radial=6,
    n_axial=7,
    radial_degree=1,
    axial_degree=1,
    max_iterations=3,
    step_size=0.01,
    viscosity=0.01,
)
assert candidate_artifact_schema_errors(artifact) == []
assert artifact["result"]["final_loss"] <= artifact["result"]["initial_loss"]

report = replay_candidate_artifact(json.loads(json.dumps(artifact)))
assert report["replay_match"]
assert report["unproven_claim"] is False
```

This is the first residual-descent loop for a concrete symmetry-reduced 3D
Navier-Stokes candidate family. The train/holdout split helps catch basic
overfitting to replay points, but it is still not an interval bound. A
global-regularity-grade upgrade still requires certified tails, axis smoothness, finite-energy
initial data, linearized invertibility or a universal a priori estimate, and
nonlinear closure.

## Interval, Continuum, And Closure Layers

The interval layer now wraps a refined axisymmetric artifact with
finite-dimensional interval certificates rather than only midpoint envelopes. It
stores scalar intervals as `{lower, upper, midpoint, radius}`, interval backend
metadata, coefficient boxes, derived coefficient-tail certificates,
train/holdout residual envelopes, continuum residual certificates, finite-energy
tail certificates, basis-level axis-regularity checks, unresolved analytic
obligations, and an independent symbolic verification hook.

```python
import json

from omnibias.pinn.certified import (
    build_axisymmetric_interval_report,
    build_refined_axisymmetric_swirl_candidate_artifact,
)
from omnibias.symbolic.navier_stokes import verify_axisymmetric_interval_report

artifact = build_refined_axisymmetric_swirl_candidate_artifact(
    seed=23,
    n_radial=6,
    n_axial=7,
    radial_degree=1,
    axial_degree=1,
    max_iterations=2,
    step_size=0.01,
    viscosity=0.01,
)
report = build_axisymmetric_interval_report(artifact)
verified = verify_axisymmetric_interval_report(json.loads(json.dumps(report)))
assert verified["interval_report_match"]
assert verified["tail_certified"]
assert verified["continuum_certified"]
assert report["honesty"]["unproven_claim"] is False
```

This is stronger than raw replay: coefficient values, midpoint diagnostics,
combined finite-energy intervals, and continuum residual envelopes must survive
independent recomputation. It is still not a proof of global regularity. The certificate is a
finite-dimensional certified-evidence object; the analytic closure still needs either a
linearized-invertibility / radii-polynomial blow-up argument, or a universal
a-priori inequality plus a continuation-criterion implication.

The axisymmetric closure layer now makes the blow-up route more explicit:

```python
import json

from omnibias.pinn.certified import (
    build_axisymmetric_blowup_closure_report,
    build_axisymmetric_interval_report,
    build_refined_axisymmetric_swirl_candidate_artifact,
)
from omnibias.symbolic.navier_stokes import verify_blowup_closure_report

artifact = build_refined_axisymmetric_swirl_candidate_artifact(
    seed=29,
    n_radial=6,
    n_axial=7,
    radial_degree=1,
    axial_degree=1,
    max_iterations=2,
    step_size=0.01,
    viscosity=0.01,
)
interval = build_axisymmetric_interval_report(artifact)
closure = build_axisymmetric_blowup_closure_report(interval)
verified = verify_blowup_closure_report(json.loads(json.dumps(closure)))

assert verified["closure_report_match"]
assert closure["unproven_claim"] is False
```

This closure report contains function-space metadata, a finite-dimensional
linearized residual Jacobian certificate, a named projection/truncation map, a
tail/operator-remainder bound, a Neumann-style defect interval, component-wise
radii-polynomial evidence, norm-divergence linkage metadata, and symbolic replay
inputs. A surviving candidate may now reach `closure_consistency_verified`, which
means the interval report and finite-dimensional closure arithmetic replay
consistently. It does not mean linearized invertibility has been proved in the
continuum Banach space, nor that a critical norm divergence has been linked
to an exact Navier-Stokes solution profile. Those remain explicit obligations in
the formal package and manifest.

The regularity route now has its own replayable certified-evidence layer:
`build_regularity_inequality_report()` consumes a `regularity_growth_law`
artifact, records coefficient intervals and continuation metadata, runs a
deterministic counterexample sweep over trace-space probes, and is replayed by
`verify_regularity_inequality_report()`. A passing report closes only the
machine-checkable evidence gate for the selected trace artifact; it still leaves
`all_smooth_finite_energy_data_proof` open until an analytic proof covers all
smooth finite-energy 3D data.

The theorem-grade attempt layer sits above those certified-evidence reports. It adds a
continuum Banach contract, theorem-grade operator/radii/norm/regularity
proof-attempt artifacts, symbolic replay via
`verify_theorem_grade_closure_attempt()`, and a strict `theorem_claim_gate()`.
Each route returns `proved_by_external_artifact`,
`blocked_with_precise_obligations`, or `falsified_with_counterexample`. Without
hash-checked external theorem-prover evidence, `unproven_claim` remains `False` even
when finite-dimensional certified-evidence certificates replay successfully.

The full proof-program artifact, `build_ns_proof_program_report()`, records all
remaining pieces needed for a computer-assisted proof attempt: exact equation
contracts, theorem-grade function-space definitions, candidate-family
implementation/falsification status, required interval/CAP backend capabilities,
blow-up and regularity lemma packages, Lean import metadata, and an external
review gate. Its replay verifier, `verify_ns_proof_program_report()`, checks
that the named open obligations and lemmas match the serialized evidence. This
answers what was proved, what failed, and exactly which lemma remains open; it
does not turn any theorem-grade boolean true by itself.

The gate-closure layer is evidence-driven. `proof_obligation_bundle()` records a
route, lemma ID, theorem statement, assumptions, dependencies, source artifact
hash, expected verifier, and obligation hash. `theorem_verifier_record()` records
which obligations an external verifier claims to discharge. The ingestion layer,
`ingest_theorem_verifier_bundle()`, accepts an obligation only when the theorem
name, obligation hash, source hash, verifier identity, verified status, and
freshness metadata all match. Partial evidence can close individual lemmas, but a
route remains blocked until every required lemma for that route has matching
evidence.

The remaining layers are explicit artifacts:

- **Gate and baseline hardening:** exact-flow tests now cover the primitive,
  pressure-Poisson, Leray, and inviscid vorticity-form surfaces. Promotion gates
  distinguish tail payloads from certified tail bounds.
- **True interval arithmetic layer:** interval operations cover coefficients,
  arithmetic, compactification maps, quadrature smoke checks, and residual
  report serialization through `ScalarInterval`.
- **Tail and continuum certification:** `build_axisymmetric_interval_report`
  emits derived coefficient-tail certificates, cellwise continuum residual
  envelopes, finite-energy tail certificates, and axis smoothness evidence from
  the finite polynomial basis.
- **Certified refinement:** `certified_candidate_refinement_report()` records
  whether the refined candidate survived the certified objectives and preserves
  coefficient lineage for falsification.
- **Analytic closure:** `build_blowup_closure_report()`,
  `build_axisymmetric_blowup_closure_report()`, and
  `build_regularity_inequality_report()` package the two regularity routes without
  claiming either is proved. The blow-up route now distinguishes
  finite-dimensional linearization, operator-theoretic remainder/Neumann
  evidence, component-wise radii closure, and norm-divergence linkage. The
  regularity route distinguishes selected inequality coefficients,
  counterexample sweeps, continuation metadata, and the remaining
  all-data proof obligation.
- **Theorem-grade attempt:** `build_theorem_grade_closure_attempt()` records the
  continuum function-space contract, Banach invertibility attempt, interval-root
  radii attempt, exact-profile norm-divergence attempt, and regularity all-data
  attempt. `theorem_claim_gate()` rejects every global-regularity claim unless all route
  obligations and external proof checks are closed.
- **Proof program:** `build_ns_proof_program_report()` composes the exact PDE
  contracts, route function spaces, interval backend requirements, candidate
  family status, lemma packages, Lean metadata, and external review gate into a
  single replayable status artifact. It also emits proof-obligation bundles and
  per-route verifier-ingestion results.
- **Formal packaging:** `build_formal_proof_package()` generates deterministic
  proof-obligation records for Lean/Isabelle/Coq or a written CAP package,
  including generated theorem-obligation hashes for theorem-grade reports.
- **External verification:** `build_certificate_manifest()` records artifact
  hashes, environment metadata, verification commands, open obligations, and the
  final claim boundary.

Current status: the implemented layers make the certified-evidence pipeline harder to
overclaim and easier to falsify. The current finding is that the
axisymmetric finite-basis artifact can now be wrapped in certified
finite-dimensional interval, tail, energy, and axis-regularity reports, but the
global-regularity-grade theorem is still blocked at analytic closure: no linearized
invertibility/radii-polynomial proof or universal a-priori inequality has been
established by the code.

## Constantin-Lax-Majda: Exact Whole-Line Blow-Up Certificates

The 1D Constantin-Lax-Majda (CLM) model `omega_t = omega H(omega)` is the
classical *exactly solvable* surrogate for 3D vortex stretching (`H` is the
whole-line Hilbert transform). Its solution

```
omega(x, t) = 4 omega0(x) / ((2 - t H omega0(x))^2 + (t omega0(x))^2)
```

blows up in finite time **iff** there is a point `x*` with `omega0(x*) = 0` and
`H omega0(x*) > 0`, at time `t = 2 / H omega0(x*)`. Taking the initial vorticity
in the verified conjugate-Poisson basis `omega0 = sum_i c_i q_{a_i}`
(`q_a(x) = x/(x^2+a^2)`, `a_i > 0`) makes the nonlocal operator **exact and
closed-form** (`H q_a = -p_a`), so every quantity is an outward-rounded interval
with *no quadrature and no truncated tail*.

`certified_clm_blowup()` certifies the origin zero (guaranteed by oddness) and
reports `2 / H omega0(0)` as an *upper bound* on the first singularity time.
`certified_clm_multizero_first_blowup()` strengthens this to the **earliest**
blow-up time over *all* zeros of `omega0`:

```python
from omnibias.pinn.certified import (
    certified_clm_multizero_first_blowup,
    certified_clm_multizero_first_blowup_schema_errors,
)

# Odd profile whose interior (non-origin) zero attains the largest H omega0.
cert = certified_clm_multizero_first_blowup(coeffs=[-0.59, 0.08], scales=[1.45, 0.41])

assert certified_clm_multizero_first_blowup_schema_errors(cert) == []
assert cert["completeness_certified"] is True          # all zeros found (exact Sturm)
assert cert["singularity_certified"] is True
print(cert["earliest_zero_location"]["midpoint"])      # ~0.368 -- not the origin
print(cert["first_blowup_time"])                        # two-sided certified interval
```

The "earliest" claim rests on three exact layers, none using quadrature:

- the non-origin zeros are `+-sqrt(u*)` for the positive roots `u*` of the
  degree `n-1` numerator polynomial `P(u) = sum_i c_i prod_{j!=i}(u + a_j^2)`,
  whose coefficients are exact `Fraction`s of the literal float inputs;
- **all** distinct positive roots are isolated by an exact Sturm sequence and
  enclosed by the interval-Newton test, so completeness (no missed zero with an
  even larger `H omega0`) is *certified*, not assumed;
- `H omega0` at every zero is an exact interval, so the maximum -- hence
  `T* = 2 / max H omega0` -- is a two-sided certified enclosure.

If completeness cannot be certified (e.g. a multiple root defeats the
uniqueness test) the time degrades honestly to the rigorous
`first_blowup_time_upper_bound` and `earliest_first_blowup_certified` is `False`.

An independent numpy-only twin,
`omnibias.symbolic.verify_clm_multizero_first_blowup()`, recomputes the
numerator polynomial, the zeros, and the maximum Hilbert value from scratch
(importing nothing from `omnibias.pinn.certified`) and checks they fall inside the
certificate's intervals; an `mpmath` principal-value quadrature independently
confirms the line Hilbert transform at the earliest zero.

!!! info "Claim boundary"
    This is a rigorous statement about the **1D CLM model only**. Every
    certificate hard-wires `honesty.unproven_claim`, `three_d_claim`, and
    `continuum_navier_stokes_claim` to `False`; it is a 1D model of vortex
    stretching, **not** a 3D Navier-Stokes/Euler blow-up and **not** a global-regularity
    result. See [Scope & guarantees § 3](../scope-and-guarantees.md#3-the-certified-pde-stacks-navierstokes-ccf).

## Near-Term Research Ladder

The practical sequence is:

1. certify known exact and manufactured Navier-Stokes flows;
2. add compactified-domain or Chebyshev/rational tails beyond the periodic
   sandbox;
3. search for regularity inequalities over energy, enstrophy, palinstrophy,
   strain, and vorticity features;
4. search for symmetry-reduced self-similar candidates, starting with
   axisymmetric classes;
5. turn surviving candidates into interval and proof-assistant obligations.

This is a certified-evidencearation program. Progress is measured by stronger
certificates and sharper falsification, not by a neural residual alone.

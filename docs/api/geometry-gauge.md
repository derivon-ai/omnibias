# omnibias.geometry.gauge — non-abelian gauge theory

> **Folded package.** This is the alpha `omnibias.geometry.gauge` submodule of
> `omnibias-geometry` (formerly the standalone `omnibias-gauge` package). Imports
> use `omnibias.geometry.gauge`; the numerics are unchanged.

Non-abelian gauge theory on flat `R^4`, built on the `omnibias-fields` substrate
with cross-backend (PyTorch + JAX) parity. This is the package that makes a
Yang-Mills connection *expressible*: Lie-algebra-valued forms, the field strength
`F = dA + g [A, A]`, the gauge-covariant derivative, the Yang-Mills operator
`D_mu F^{mu nu}`, the Bianchi identity, a flat signature-aware Hodge star, the
action and topological charge, and gauge transformations.

## What the `gauge` submodule adds over the rest of `omnibias-geometry`

`omnibias-geometry` is purely Riemannian and abelian: its `DifferentialForm`
components are scalar-valued, the only connection is Levi-Civita, and the metric
`signature` is never read. The `gauge` submodule adds the *connection-of-a-bundle*
objects that gauge theory needs:

- `LieAlgebra` for `u(1)`, `su(2)` (Pauli), `su(3)` (Gell-Mann), and general
  `su(N)`: generators `T^a`, structure constants `f^{abc}`, symmetric `d^{abc}`,
  normalization `tr(T^a T^b) = 1/2 delta^{ab}`.
- `LieAlgebraValuedForm`, generalizing the geometry form with an adjoint index.
- A flat, signature-aware Hodge star on `R^4` (Euclidean `(+,+,+,+)` and
  Minkowski `(-,+,+,+)`).

Field derivatives (`d_mu A_nu`, second partials for `D_mu F^{mu nu}`) come from
the omnibias **closed-form** activation-derivative tower via `FieldState`, not
autodiff or finite differences.

Gated Wilson-line holonomy band (theory 02-14) wraps these transport
primitives: [holonomy_band.md](holonomy_band.md). Closed form only
abelian + transverse-constant; open lines are gauge-dependent; **no**
Yang-Mills / mass-gap / continuum claim.

## Nonintegrable derivative: parallel transport, holonomy & the curvature commutator

The Wu-Yang *nonintegrable phase factor* is the finite integral partner of the
infinitesimal covariant derivative — exactly mirroring the
[Fundamental Theorem of Calculus](core.md#verified-backend-fundamental-theorem-of-calculus)
pairing of a local operator with its integral. `omnibias.geometry.gauge.{torch,jax}.ops`
expose both faces, each cross-checked against the closed-form `field_strength`:

- **Finite / integral** — `parallel_transport` computes the holonomy
  `U(C) = P exp(-i g ∫_C A_mu^a T^a dx^mu)` of a connection along a curve, and
  `wilson_loop` its gauge-invariant trace `(1/dim) Re tr U`. The connection is
  sampled along `curve(transport_nodes(...))` (the `line_integral` pre-evaluation
  convention); the representation defaults to the fundamental, and any irrep
  (`spin_matrices`, `adjoint_generators`, …) is selected by an explicit
  `generators=`. There is also a `parallel_transport_from_arrays` low-level form.
- **Local / differential** — `covariant_derivative_commutator` evaluates
  `([D_mu, D_nu] phi)^a` for an adjoint scalar the "derivative way" (nesting the
  covariant derivative), and `curvature_commutator_defect` returns its residual
  against the closed-form Ricci identity
  `[D_mu, D_nu] phi = g f^{abc} F_{mu nu}^b phi^c`. Both have `_from_arrays`
  variants; the `FieldState` variants take the adjoint-scalar component names.

The `-i g` exponent sign and the path ordering (earliest-parameter segment on the
left) are **fixed** so that an infinitesimal loop reproduces this package's
`field_strength`: the non-abelian Stokes cross-check
`(i / (g · area)) (U − I) → F_{mu nu}` converges first order in the loop radius
(error halves when the radius halves) against the strong BPST instanton.

**Honesty labels.** The connection and its closed-form derivatives are exact
(sigma tower); the covariant-derivative commutator identity is exact and
**bit-identical** across torch / jax (pure `einsum`). The parallel transport is a
midpoint ordered product converging as `substeps → ∞`; because the matrix
exponential is each backend's native primitive (`torch.linalg.matrix_exp` vs
`jax.scipy.linalg.expm`), the holonomy agrees torch ↔ jax to `rtol ≈ 1e-9`
(float64), not bit-for-bit. The small-loop → `F` check is `O(area)`-accurate
(validated by convergence), the finite-side analog of the FTC residual. Never a
open-problem claim.

## Characteristic classes (Chern-Weil)

The topology surface exposes Chern-Weil characteristic numbers built from the
closed-form field strength `F`:

- `second_chern_number` / `topological_charge` — the instanton number
  `c_2 = (1/8π²)∫ tr(F ∧ F)` (metric-independent Levi-Civita *symbol*).
- `first_chern_class` / `first_chern_number` — the abelian `U(1)` first Chern
  class `c_1 = F/(2π)` and its integral over a closed 2-surface.
- `pontryagin_density` / `pontryagin_number` — the first Pontryagin number
  `p_1 = -2 c_2` of an `SU(N)` bundle (textbook `p_1 = c_1² - 2c_2` with
  `c_1 = 0`).
- `euler_number_so2` — the Euler number of a rank-2 (`SO(2) ≅ U(1)`) bundle,
  equal to `c_1`. The Euler *characteristic of a surface* is
  `omnibias.geometry`'s `gauss_bonnet_euler`.

The densities are closed-form; the characteristic *numbers* are numerical
integrals whose **integer** value can be certified by an
`omnibias.core.verified.Interval` enclosure that brackets exactly one integer.
Validated on the Dirac monopole (`c_1 = n`, verified integer-quantisation) and
the BPST instanton (`c_2 = 1`, `p_1 = -2`). Higher-rank Euler classes (via the
Pfaffian) and index theorems are out of thesis.

## Representation theory (`su(N)` highest weights)

`omnibias.geometry.gauge._core.representation` is the pure-Python (numpy) highest-weight
calculus for the finite-dimensional irreducible representations of `su(N)`. An
`Irrep(n, dynkin)` is labelled by its non-negative **Dynkin labels**
`(a_1, ..., a_{N-1})`; the module then computes:

- **Exact invariants** — `dimension` (Weyl dimension formula), `quadratic_casimir`
  (`C_2`, physics normalization `tr_fund(T^a T^b) = 1/2 delta^{ab}`), and
  `dynkin_index` (`T(R) = C_2(R) dim(R)/dim(G)`). These are exact `Fraction` /
  `int` results — certified by construction, not sampled.
- **Weight system** — `weight_multiplicities` via the Freudenthal recursion
  (dominant weights) plus Weyl-orbit expansion.
- **Products & branching** — `tensor_product_decomposition` (the Racah–Speiser /
  Brauer–Klimyk algorithm) and `branching_to_subalgebra`
  (`su(N) -> su(N-1)`, Gelfand–Tsetlin interlacing).
- **Weyl character** — `character`, the Schur bialternant
  `det(x_i^{lambda_j+N-j}) / det(x_i^{N-j})`.
- **Explicit matrices** — `su2_spin_matrices` (the spin-`j` angular-momentum
  matrices), `adjoint_rep_matrices` (`(T^a_{adj})_{bc} = -i f^{abc}`), and the
  symmetric / antisymmetric tensor powers `symmetric_power_rep_matrices` (`Sym^k`,
  bosonic Fock formula) / `antisymmetric_power_rep_matrices` (`Λ^k`, fermionic
  Jordan–Wigner formula) of the fundamental — all Hermitian and algebra-closing
  by construction.

The torch / jax op surface (`spin_matrices`, `adjoint_generators`,
`symmetric_power_generators`, `antisymmetric_power_generators`,
`casimir_operator`, `casimir_eigenvalue`, `dynkin_index_value`) materializes the
numpy core as bit-identical tensors, so an irrep's generators plug straight into
the gauge-covariant-derivative surface. Three independent oracles pin every
claim: analytic textbook values (`su(2)` spin-`j`: `dim = 2j+1`, `C_2 = j(j+1)`;
`su(3)`: `C_2(3) = 4/3`, `3 ⊗ 3̄ = 8 + 1`, `8 ⊗ 8 = 27 + 10 + 10̄ + 8 + 8 + 1`),
a symbolic `sympy` character-product identity, and internal cross-checks
(weight multiplicities sum to `dim`; the adjoint Casimir equals the
`LieAlgebra.dual_coxeter_number`; `sum_a (T^a)^2 = C_2 I`; and
`tr(J_a J_b) = T(R) delta_{ab}`).

**Out of thesis (enforced).** Finite (non-Lie) group character tables and general
abstract-group theory are *not* implemented; an enforcement test fails if such a
surface is silently added. This is the Lie-algebra / highest-weight slice only.

## Validation

The test-suite validates against the analytic BPST instanton (`su(2)`,
regular gauge): self-duality defect at machine precision, zero Yang-Mills EOM
residual, topological charge `Q -> 1`, finite action `S -> 8 pi^2 / g^2`, the
Bianchi identity at machine precision, gauge invariance of the action, and
torch <-> jax parity to `rtol = 1e-9` in float64.

## Schemas

::: omnibias.geometry.gauge._core
    options:
      show_root_heading: false
      heading_level: 3

## Ops (torch)

::: omnibias.geometry.gauge.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.geometry.gauge.jax.ops`) is the bit-identical twin; the
cross-backend tests assert agreement to `rtol=1e-9` in float64.

## Stochastic quantisation (continuum)

`omnibias.geometry.gauge.{torch,jax}.ops` also expose the Parisi-Wu / DeTurck-gauged
gradient-flow right-hand side `gauge_flow_rhs` (the Yang-Mills gradient-flow
drift `-D_mu F^{mu nu}` plus an optional DeTurck gauge-fixing term) and a
`langevin_step` that adds scaled Gaussian noise to that drift. These are the
bit-identical twins of the lattice updater below and realize the
DeTurck-Zwanziger parabolic-flow picture as a tested numerical operator (they do
**not** constitute a rigorous stochastic-quantisation construction).

## Lattice Monte-Carlo (evidence)

The `omnibias.geometry.gauge.lattice` submodule provides an SU(2) Wilson lattice
Monte-Carlo driver at **fixed spacing** with **both** a torch backend (the
default, `omnibias.geometry.gauge.lattice`, also re-exported as
`omnibias.geometry.gauge.lattice.torch`) and a JAX backend
(`omnibias.geometry.gauge.lattice.jax`). The deterministic array math (staples,
plaquette / Wilson / Polyakov traces, APE smearing, correlators, GEVP) is shared
bit-identically through `omnibias.geometry.gauge.lattice._core.kernels`; the two backends
differ only in their RNG (`torch.Generator` vs `jax.random.PRNGKey`) and so agree
**statistically**, not to ULP, on full stochastic runs.

The updater combines an exact Kennedy-Pendleton **heat-bath**, **over-relaxation**
sweeps, and an optional geodesic (exponential-map) **Langevin** updater
(`langevin_sweep`). Unlike the closed-form field operators above, lattice MC is
**stochastic numerical evidence** — results depend on RNG seeds, thermalization,
and statistics. Every output carries `unproven_claim=False`; it does **not** prove a
continuum Yang-Mills mass gap.

::: omnibias.geometry.gauge.lattice.montecarlo.run_lattice_mc
    options:
      show_root_heading: true
      heading_level: 3

Observables returned in the output dict:

- `avg_plaquette` / `avg_plaquette_err` — mean plaquette with jackknife error.
- `avg_polyakov` / `avg_polyakov_err` — volume-averaged Polyakov-loop order
  parameter (confinement diagnostic) with jackknife error.
- `glueball_correlator` / `glueball_correlator_err` — connected smeared
  glueball-operator correlator `C(t)` ensemble mean and jackknife error.
- `wilson_loops` — ensemble of rectangular Wilson loop averages.
- `creutz_ratios` — Creutz ratios from Wilson loops (area-law diagnostics).
- `string_tension` — string-tension estimate from the largest accessible Creutz
  ratio (fixed spacing, not continuum-extrapolated).
- `gevp` — generalized eigenvalue problem (GEVP) ground-state mass estimate from
  multi-level APE smearing. The reported glueball mass is a single `(t0, dt)`
  estimate in lattice units, not a plateau fit.
- `gevp_plateau` — a `(t0, dt)` scan of the GEVP ground mass with a plateau
  estimate and a `stable` flag (a coarse plateau-quality diagnostic, still
  fixed-spacing).

The same driver is available in the JAX backend:

<!-- docs-test: slow -->
```python
import jax
jax.config.update("jax_enable_x64", True)
from omnibias.geometry.gauge.lattice.jax import run_lattice_mc as run_lattice_mc_jax

mc_jax = run_lattice_mc_jax(lattice_shape=(4, 4, 4, 4), beta=2.0, seed=0)
```

Gauge-orbit / holonomy helpers (`polyakov_loop`, `gauge_transform_links`,
`gauge_orbit_distance`) live alongside the observables in both backends.

<!-- docs-test: slow -->
```python
import torch
from omnibias.geometry.gauge.lattice import run_lattice_mc

mc = run_lattice_mc(
    gauge_group="su(2)",
    lattice_shape=(8, 8, 8, 16),
    beta=2.3,
    n_therm=200,
    n_meas=200,
    device="cpu",
    seed=12345,
)
assert mc["gauge_group"] == "su(2)"
assert len(mc["glueball_correlator"]) > 0
# evidence only — not a proof of global regularity
```

The correlator, its GEVP plateau scan, and the resulting effective-mass
estimate are lattice observables on a finite lattice at finite coupling. They
say nothing about the continuum theory.

## Certified transfer-matrix mass gap (`gauge.transfer`)

A **proof** about one fixed matrix, sitting beside the Monte-Carlo *evidence*
above. `gauge.transfer` builds a finite lattice transfer matrix whose entries are
outward-rounded intervals, then certifies a lower bound on its lattice-unit mass
gap `m a = -ln(|λ₁| / λ₀)` using the rigorous engines in
`omnibias.core.verified.eig`.

Read the scope note first: every result here is a statement about **one fixed
matrix, at one fixed spacing, in finite dimension**. Nothing in this section is a
continuum limit or a claim about the Yang-Mills mass gap.

### Building a matrix

| Constructor | Basis | Spectrum |
|---|---|---|
| `u1_heat_kernel_transfer` | `character` (diagonal) or `angle` (dense circulant) | `e^{-t n²}`, exactly; every `n ≠ 0` mode is doubly degenerate |
| `su2_heat_kernel_transfer` | `character` | `e^{-t C₂(a)}` with `C₂ = a(a+2)/4` an exact `Fraction`; non-degenerate |
| `su2_class_angle_transfer` | `angle` | the same `su(2)` spectrum, in a **dense, entrywise-positive** basis that can be sampled |
| `su3_heat_kernel_transfer` | `character` | `e^{-t C₂(p,q)}`; the conjugate pair `(p,q) ↔ (q,p)` makes the subdominant mode doubly degenerate |
| `su2_wilson_transfer` | `character` | `I_m(β)` via `besseli_iv`; the slowly-decaying tail the partner-chain deflation exists for |

Because the spectrum is known in closed form for all of them, a certified bound
can be checked against the exact answer rather than merely believed.

```python
from omnibias.geometry.gauge.transfer import (
    certified_transfer_matrix_gap,
    su2_heat_kernel_transfer,
)

transfer = su2_heat_kernel_transfer(0.8, max_dynkin=4)
gap = certified_transfer_matrix_gap(transfer)

# su(2): C2(1) - C2(0) = 3/4, so the exact lattice-unit gap is 3t/4 = 0.6.
assert gap.certified
assert abs(gap.spectral_gap_lower - 0.6) < 1e-9
assert gap.method == "symmetric_power_sum_partner_chain"
```

`certified_transfer_matrix_gap` dispatches to whichever engine is both applicable
and tighter: the symmetric power-sum engine with a partner chain (which deflates
the subdominant degeneracy and the tail behind it), or Birkhoff-Hopf projective
contraction (sound but deliberately conservative, and it needs an entrywise
positive matrix). Every candidate it considered is kept on the result, including
the losing ones.

### Sandwiching the true gap

`certified_effective_mass_curve` gives rigorous **upper** bounds from the
closed-form spectrum, decreasing toward the true gap, so the looseness of a lower
bound is measurable rather than a matter of opinion.
`certified_multistep_gap_refinement` sharpens a bound via `Tⁿ`, which helps most
when no partner chain is available.

```python
from omnibias.geometry.gauge.transfer import certified_effective_mass_curve

curve = certified_effective_mass_curve(transfer, taus=(1, 2, 4, 8))
assert curve.points[-1].upper >= gap.spectral_gap_lower  # a genuine sandwich
```

### A sealed, replayable, Lean-ready certificate

```python
from omnibias.core.proof import Conjecture
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.geometry.gauge.proofmachine import build_gauge_machine

machine = build_gauge_machine()
verdict = machine.evaluate(
    Conjecture(
        "su2-gap",
        "transfer_matrix_spectral_gap",
        {
            "parameters": {
                "builder": "su2_heat_kernel_transfer",
                "coupling": "4/5",
                "max_dynkin": 4,
                "lattice_spacing": 1.0,
            }
        },
    )
)

assert verdict.status == "PROVED"
assert verify_certificate_digest(verdict.certificate)
assert verdict.certificate["continuum_claim"] is False
assert verdict.certificate["honesty"]["yang_mills_claim"] is False
```

The certificate stores its matrix's *constructor arguments*, not just the
resulting numbers, so `replay_transfer_matrix_gap` rebuilds the matrix from
scratch and re-runs the gap engine — a sealed bound tighter than an independent
derivation supports is rejected. The rational `subdominant_ratio_upper` is the
obligation the Mathlib-free Lean kernel's `spectral_gap_pos` lemma discharges.

### Cross-checking against Monte Carlo

The certificate is about a fixed matrix; a lattice Monte Carlo is about an
ensemble. Rather than assume a correspondence, `certified_gap_versus_monte_carlo`
samples the path measure `∏ₜ T_{xₜ, xₜ₊₁}` that the matrix *itself* defines, so
both sides describe the same object. The sampler reads matrix entries only — it
never touches an eigenvalue — so the gap emerges from the decay of a sampled
autocorrelation.

<!-- docs-test: slow -->
```python
from omnibias.geometry.gauge.transfer import (
    certified_gap_versus_monte_carlo,
    su2_class_angle_transfer,
)

# The class-angle basis: same su(2) spectrum, but dense and positive, so the
# induced Markov chain actually moves.  (A diagonal matrix freezes it.)
sampled = su2_class_angle_transfer(0.8, max_dynkin=3)
check = certified_gap_versus_monte_carlo(sampled, seed=0)

assert check.consistent            # certified lower bound <= MC estimate
assert check.agrees_with_exact     # and the MC brackets the closed-form gap
```

`consistent` is one-sided and the estimator's residual bias helps it pass;
`agrees_with_exact` is the two-sided test with teeth. Both are **evidence** —
only the interval arithmetic is proof.

### Scaling across spacings, honestly

`heat_kernel_gap_scaling_report` collects certified bounds at several spacings.
That is a record of a trend, explicitly labelled evidence, and it is never an
extrapolation to the continuum.

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

## Gauge-covariant jet (coordinate trap)

`GaugeCovariantJet` is the only legal input for Yang-Mills singlet discovery.
It stores `F_{mu nu}^a` and `(D_rho F_{mu nu})^a` and drops `A` / `dA` / `ddA`
after construction. The searchable library is the frozen allowlist
`LEGAL_SINGLET_ATOMS` (`tr(F^2)`, `tr(F*Ftilde)`, `|D*F|^2`, `|Bianchi|^2`,
`|F-*F|^2`). Flattened adjoint components such as `F_01_2` are rejected:
they are still a color-basis coordinate chart.

`assert_library_gauge_legal` and `evaluate_gauge_law_gate` are fail-closed.
This is **not** a `FieldJet` of the connection components, not a Wilson /
Polyakov language, and not a continuum mass-gap claim.

## Gauge-invariant dictionary (search-space trap)

A raw 2-jet of `A_mu^a` in 4D SU(3) has 480 real components. STLSQ / MDL /
stability selection cannot survive that count: they sparsify inside a design
matrix, they do not create representation theory. `GaugeInvariantDictionary`
generates Weyl singlets of `F` and `D F`, truncated by **mass dimension**.
Default search atoms at mass dimension 4 are `tr(F^2)` and `tr(F*Ftilde)`.
Bianchi is an identity, not a feature. Flattened `F_01_2` / `D^k F` component
libraries raise. This is not a Hilbert-series completeness claim.

## Data paths (noise amplification)

High-order `partial^k` on a lattice or mesh is a high-pass filter. The legal
continuum path is closed-form `A, dA, ddA` (analytic / spectral) or a weak
Yang-Mills residual against a bank of smooth adjoint test 1-forms. Lattice
links stay links: the legal local atom is a plaquette, never `partial^k A`.
Interpolating links with a random-feature field and then reading a jet is
refused. 1-D Fredholm / Volterra columns are not the 4-D Yang-Mills weak
form. A typical Monte Carlo vacuum does not satisfy `D*F = 0`; this is not
a mass-gap claim.

## Loop language (language trap)

A finite-order jet of `A` -- even `F` and `D F` -- cannot see path-ordered
holonomy. Wilson / Polyakov traces are a **second language** on
`LatticeLinkField`, not an expansion of `GaugeCovariantJet`. Creutz is a
derived identity on those traces. GEVP and `gauge.transfer` stay certificates
(`yang_mills_claim` / `continuum_claim` remain false). This is not a continuum
mass-gap claim.

## Ensemble language (Path B)

Path B changes the object. Rows are ensemble statistics versus control
parameters -- `|P|`, `χ_P`, `C_P(r)`, Landau-gauge `G(p^2)`, a planted
spectral density -- not a jet of `A` and not a per-configuration loop table.
One `LatticeLinkField` is not an ensemble. GEVP / transfer-gap stay
certificates. The Euclidean `G → ρ` step is a **named regularized inverse**
of a finite Källén–Lehmann kernel; the only acceptance gate is planted-`ρ`
recovery. `yang_mills_claim` / `continuum_claim` remain false. This is not a
continuum mass-gap claim.

::: omnibias.geometry.gauge._core.ensemble_language
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.gauge._core.landau_gluon
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.gauge._core.spectral_density
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.gauge._core.loop_language
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.gauge._core.data_paths
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.gauge._core.weak_ym
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.gauge._core.invariants
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.gauge._core.jet_dimension
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.geometry.gauge._core.covariant_jet
    options:
      show_root_heading: false
      heading_level: 3

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
| `su3_wilson_transfer` | `character` | Cellwise interval-range Haar enclosures of the Wilson character coefficients on the SU(3) torus; the matrix **is** the `(p,q)≤3` truncation, not a product of ordinary `I_n` |

Heat-kernel and SU(2) Wilson spectra are known in closed form, so a certified
bound can be checked against the exact answer. SU(3) Wilson eigenvalues are
interval enclosures of a Haar integral at one coupling; a numerical sample of
each coefficient must lie in its enclosure. Neither constructor is 4-D
Yang-Mills.

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

Optional `trial=` feeds a holonomy trial space (characters of closed loops on
the dense `angle` / class-angle grid) into Lehmann–Maehly and the symmetric
engine. Character-basis heat-kernel matrices are already diagonal in this
basis, so the lever is the dense constructors. A badly conditioned Gram is
flagged rather than silently trusted. This is still one fixed matrix; the
continuum limit is not taken.

```python
from omnibias.geometry.gauge.transfer import (
    certified_transfer_matrix_gap,
    holonomy_trial_space,
    su2_class_angle_transfer,
)

dense = su2_class_angle_transfer(0.8, max_dynkin=4)
trial = holonomy_trial_space(dense)
holonomy_gap = certified_transfer_matrix_gap(dense, trial=trial)
assert holonomy_gap.certified
assert holonomy_gap.trial_flagged is False
assert holonomy_gap.trial_gram_condition is not None
# su(2) exact lattice-unit gap is 3t/4 = 0.6; a lower bound cannot exceed it.
assert holonomy_gap.spectral_gap_lower <= 0.6 + 1e-9
```

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

### Two-scale strong-coupling polymer bound

`certified_strong_coupling_glueball_bound` is a **self-contained** lower bound
on a 4-D SU(2) Wilson glueball mass at one fixed `β`, using the character
activity `u(β) = I₂(β)/I₁(β)` and a two-scale polymer remainder
`Σ N_n u^n ≤ u + A u² / (1 - B u)`. The first attachment is
`A = 4(2d-3)` (`20` in four dimensions); later generations use the
backtrack branching `B = 3(2d-3)` (`15` in four dimensions). Single-scale
`counting="backtrack"` (`C = 15`, **not** a bound on `N_2`) and
`counting="crude"` (`C = 24`), and `counting="cluster"` (explicit keep
plus a geometric tail) remain available. `certified=True` only when
the enclosed contraction ratio is strictly less than 1. Out of that
domain the leading activity is still returned and must not be sealed as proved.

This is counting at one coupling and one spacing. It is **not** a
continuum claim and **not** a formalization of Osterwalder-Seiler.

```python
from omnibias.geometry.gauge.transfer import (
    BETA_LOCK,
    certified_strong_coupling_glueball_bound,
    certified_wilson_character_gap,
)

polymer = certified_strong_coupling_glueball_bound(BETA_LOCK)
assert polymer.certified
assert polymer.spectral_gap_lower > 0.0
assert polymer.method == "two_scale_polymer_count"
assert polymer.first_step == 20
assert polymer.coordination == 15

# Infinite character-basis Wilson transfer (0+1-D, still not 4-D YM).
wilson = certified_wilson_character_gap(BETA_LOCK)
assert wilson.certified
assert wilson.spectral_gap_lower > polymer.spectral_gap_lower
```

### Two-plaquette Kogut–Susskind Hamiltonian

`su2_two_plaquette_hamiltonian` is a **multi-link** SU(2) operator on the
Gauss-law triples `|j1, j2, js⟩`. Its spectrum is not known in closed form,
so a holonomy trial space is not secretly the eigenbasis.
`certified_hamiltonian_gap` lower-bounds `λ1 - λ0` of **this** finite
matrix at one `g²` and one `j_max`. It is not 4-D Yang-Mills and the
continuum limit is not taken.

```python
from omnibias.geometry.gauge.transfer import (
    COUPLING_LOCK,
    certified_hamiltonian_gap,
    su2_two_plaquette_hamiltonian,
)

hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
h_gap = certified_hamiltonian_gap(hamiltonian)
assert h_gap.certified
assert h_gap.spectral_gap_lower > 0.0
```

Default magnetics are Racah 6j recoupling weights (`magnetic="sixj"`);
`magnetic="character"` keeps the older amplitude-1 operator. The
three-plaquette chain `su2_three_plaquette_hamiltonian` is the same
finite-matrix statement on `|j1,j2,j3,js12,js23⟩`.

### Spatial-strip transfer

`su2_spatial_strip_transfer` is a Euclidean-time transfer on a spatial
circle of SU(2) class angles (`2×4 → 16` in CI). Its spectrum is not
known in closed form. `su2_spatial_torus_transfer` is the finite 3+1-D
analogue on a `2×2` spatial torus (`n_angles=2 → 16` in CI).
`certified_strip_reflection_positivity` checks `⟨θv, T v⟩` on a locked
angle inversion; that is RP on **this** matrix, not Osterwalder–Seiler
reconstruction. `certified_strip_cluster_tail` encloses a geometric tail
of a locked spatial-bond correlator.

```python
from omnibias.geometry.gauge.transfer import (
    STRIP_COUPLING_LOCK,
    certified_strip_cluster_tail,
    certified_strip_reflection_positivity,
    certified_transfer_matrix_gap,
    su2_spatial_strip_transfer,
)

strip = su2_spatial_strip_transfer(STRIP_COUPLING_LOCK, n_sites=2, n_angles=4)
strip_gap = certified_transfer_matrix_gap(strip)
assert strip_gap.certified
rp = certified_strip_reflection_positivity(strip)
assert rp.certified
cluster = certified_strip_cluster_tail(strip, n_keep=2)
assert cluster.certified
assert cluster.tail.contains(cluster.sample)
```

### Scaling across spacings, honestly

`certified_gap_scaling_table` (alias of `heat_kernel_gap_scaling_report`)
collects certified bounds at several spacings. That is a record of a
trend, explicitly labelled evidence, and it is never an extrapolation to
the continuum. `continuum_claim` is hard-wired `False`.

```python
from omnibias.geometry.gauge.transfer import (
    certified_gap_scaling_table,
    su2_heat_kernel_transfer,
)

table = certified_gap_scaling_table(
    su2_heat_kernel_transfer,
    spacings=[1.0, 0.5, 0.25],
    couplings=[0.8, 0.4, 0.2],
    max_dynkin=4,
)
assert table.continuum_claim is False
assert len(table.points) == 3
```

### Polymer β-domain, and one sealed finite-gauge report

`certified_polymer_beta_domain` evaluates the polymer majorant on a
locked dyadic grid (`k/32` for `k = 1..16`). It records the largest
certifying grid point and the next grid failure. That is the
**majorant's** domain on that grid, not a physical critical coupling
and not `a -> 0`.

`certified_wilson_character_beta_domain` evaluates the infinite
character-basis Wilson gap on a wider locked grid
(`1/4, 1/2, 1, 2, 4`). It certifies at `1/4`, where the 4-D polymer
two-scale majorant already fails, and at larger points. That is a
0+1-D character-transfer statement on those points, not a physical
critical coupling and not 4-D Yang-Mills.

`finite_gauge_report` runs the existing engines on one named spec
(polymer plus cluster, the β-domain, the Wilson character gap and its
domain, Haar identities, a cellwise SU(3) Haar transfer whose gap is
required, the two-plaquette Hamiltonian with a measured G1 factor,
strip reflection positivity, and a three-point heat-kernel scaling
table). The report locks `n_cells=32`, the smallest of `{16, 32}` that
certifies a positive SU(3) gap after cellwise Haar. The bundle is
still a list of finite statements. `continuum_claim` and
`yang_mills_claim` stay false. It is not a staircase to Clay
existence.

```python
from fractions import Fraction

from omnibias.geometry.gauge.transfer import (
    certified_polymer_beta_domain,
    certified_wilson_character_beta_domain,
    finite_gauge_report,
)

domain = certified_polymer_beta_domain()
assert domain.certified
assert domain.beta_certified < domain.beta_outside
assert domain.continuum_claim is False

wilson_domain = certified_wilson_character_beta_domain()
assert wilson_domain.certified
assert wilson_domain.quarter_certified
assert wilson_domain.beta_certified > Fraction(1, 4)
assert wilson_domain.continuum_claim is False

pack = finite_gauge_report()
assert pack.certified
assert pack.continuum_claim is False
assert pack.yang_mills_claim is False
assert pack.g1.ge_generic
assert pack.g1.factor + 1e-12 >= 1.0
assert pack.haar.certified
assert pack.su3_gap.certified
assert pack.wilson_character_domain.certified
```

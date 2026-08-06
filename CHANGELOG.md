# Changelog

All notable changes to omnibias are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and each of the 42
distributions is versioned independently under semantic versioning.

## [Unreleased]

### Added — `omnibias-pinn`: conservative shock capturing (cage over partition)

Representing a shock wants a sharp seam; representing a conservation law wants
`div G = 0`. The two were separately available and had never been composed,
because the obvious composition is wrong: making each partition patch a
`FluxFormField` gives `div (sum_l w_l G_l) = sum_l grad w_l . G_l`, which
vanishes only where the gates are saturated — conservation breaks exactly at the
seam, the one place a shock needs it.

Inverting the nesting fixes it, and needed no new code. `FluxFormField` builds
`G^i = sum_j d_j P^{ij}` from an antisymmetric potential, so
`div G = d_i d_j P^{ij} = 0` by symmetry of mixed partials, for *any*
twice-differentiable `P` however sharp. Putting the partition inside, as the
potential, buys an arbitrarily sharp front at no cost to the conservation law.
That is also the right representation rather than a convenient one: with axes
`(t, x)` the cage gives `rho = d_x P`, so `P` is the cumulative mass, and the
exact viscous profile integrates to `P -> c0 x - (c0^2 + a^2) t/2 - a |x - c0 t|`
as `nu -> 0`. The potential of a shock is a **kink**, which is what a partition
of unity over smooth patches already represents well.

- `packages/omnibias-pinn/tests/partition/test_cage_over_partition.py` pins the
  composition: relative `|div G| < 1e-12` for `beta` from 2 to 200 (measured
  2e-16 to 3e-14), holding after training relocates the seam, and the rejected
  nesting is kept as a live falsifier at order-one divergence — without it the
  passing test would look like a property of cages in general.
- `docs/examples/pinn_burgers_shock.py` and
  `benchmarks/burgers_shock_conservation.py` (artifact:
  `docs/benchmarks/burgers_shock_conservation.json`) train the cage against a
  non-conservative arm at identical architecture, parameter count, seed and
  collocation budget, over 6 viscosities x 5 seeds.

Measured, and reported the way it came out. The conservative arm holds global
mass balance tighter at **every** viscosity, and the margin widens as the layer
goes under-resolved: 1.3x at `nu = 2e-2`, 2.9x at `nu = 2e-3`, winning 5 seeds out
of 5 once `nu <= 3e-3`. As `nu` falls tenfold the baseline's mass error grows
4.5x while the cage's grows 2.0x. On shock speed and relative L2 the ordering
**reverses** with resolution — the non-conservative arm is up to 16x better where
the layer is resolved — so only the mass-balance result is asserted in CI and the
rest is printed. That is the finite-volume result carried onto a mesh-free field:
a conservative scheme is not more accurate on a well-resolved smooth problem, it
is more robust when the feature is under-resolved.

Scope: `div G = 0` is structural (3.4e-15 worst over all 30 sweep cells, no
training, no quadrature, no tolerance); solution accuracy is optimised, not
proven. Conservation pins the Rankine-Hugoniot jump condition and does **not**
select the entropy solution. The `beta -> inf` seam sharpening is *temperature
collapse*, never the founding `delta -> 0` bias collapse.

Also adopts `docs/examples/partitioned_pinn.py`, which existed but appeared in no
`docs/examples.md` row, no `llms.txt` entry and no CI job, so nothing had been
running it.

### Fixed — docs: the published derivative-order ceiling understated the tower

`docs/scope-and-guarantees.md` sec 2 capped `huber`, `silu`/`swish`, `gelu`,
`relu` and `mish` at order 1, blaming a "Dirac at the kink". Every entry in that
row has had an all-orders fast path for some time, and three of the five have no
kink at all: `silu`, `gelu` and `mish` are smooth everywhere and carry exact
Leibniz towers over the `z f(z)` product, while `relu` and `huber` carry
all-orders almost-everywhere / regular-part towers. `smooth_sign` was likewise
listed at 2 despite being `tanh` rescaled, hence unbounded.

The table is labelled "reading guidance for AI agents", so a stale ceiling there
does not merely go out of date — it teaches every agent and reader the wrong
answer. `tests/test_doc_activation_orders.py` now parses the table and checks
every claim against the live registry on both backends, including a negative half
that requires each claimed cap to genuinely raise past its ceiling, so replacing
the table with "unbounded" everywhere would fail. Also corrects the stale header
tables in `omnibias.torch.activations.nqs` (`mish`, `smooth_sign`) and a section
comment in `omnibias.jax.activations`.

### Added — `omnibias-geometry`: certified lattice mass gap (`omnibias.geometry.gauge.transfer`)

The rigorous gap engines in `omnibias.core.verified.eig` were written *for* this
application — their docstrings name heat-kernel and Wilson transfer matrices, the
`+/- n` U(1) modes and the `(p,q) <-> (q,p)` SU(3) pairs — and were then never
connected to one. This connects them.

- **Transfer matrices** (`omnibias.geometry.gauge.transfer.matrices`):
  `u1_heat_kernel_transfer` (`character` diagonal or `angle` dense circulant),
  `su2_heat_kernel_transfer` / `su3_heat_kernel_transfer` (eigenvalues
  `exp(-t C2)` from the *exact* `Fraction` of `quadratic_casimir`),
  `su2_class_angle_transfer` (the `su(2)` spectrum in a dense, entrywise-positive
  basis that a Markov chain can actually move in), and `su2_wilson_transfer`
  (character expansion via the new `besseli_iv`). Entries are outward-rounded
  intervals; the closed-form spectrum is carried alongside, so a certified bound
  can be checked against truth rather than believed.
- **Certified gaps** (`.gap`): `certified_transfer_matrix_gap` dispatches to the
  symmetric power-sum engine with a partner chain, or Birkhoff-Hopf, whichever is
  applicable and tighter, keeping every candidate it considered.
  `certified_multistep_gap_refinement` sharpens via `T^n`;
  `certified_effective_mass_curve` supplies rigorous *upper* bounds so the true
  gap is genuinely sandwiched; `heat_kernel_gap_scaling_report` records bounds
  across spacings as evidence about a trend.
- **Certificates and a registry** (`.certificates`, `gauge/proofmachine.py`):
  sealed, tamper-evident `verified-transfer-matrix-gap-1` certificates carrying a
  top-level `subdominant_ratio_upper`, so the Mathlib-free Lean kernel's
  `spectral_gap_pos` lemma discharges the obligation with no new Lean. Replay
  rebuilds the matrix from its recorded *constructor arguments* and rejects a
  sealed bound tighter than an independent derivation supports. `gauge_provers()`
  / `build_gauge_machine()` mirror `omnibias.sos.proofmachine`, since
  `omnibias-pinn` does not depend on `omnibias-geometry`.
- **Monte-Carlo cross-check** (`.montecarlo`): rather than assume a matrix
  corresponds to an ensemble, `certified_gap_versus_monte_carlo` samples the path
  measure `prod_t T_{x_t, x_{t+1}}` that the matrix *itself* defines, reading
  matrix entries only. On `su(2)` the certified bound is exactly tight and the
  sampled effective mass brackets the closed-form gap.

Scope, unchanged and non-negotiable: every certificate is a statement about **one
fixed matrix at one fixed spacing in finite dimension**. `continuum_claim` is
hard-wired `False`, the scaling report is labelled evidence, and nothing here is a
claim about the Yang-Mills mass gap.

### Added — `omnibias-core`: `besseli_iv`

`omnibias.core.verified.besseli_iv` encloses the modified Bessel function
`I_n(x)`, following the existing `erf_iv` mpmath-bracket pattern so it inherits
`strict_backend()` / `libm_fallback_used()`. Because
`I_n(z) = sum_k (z/2)^(2k+n) / (k! (k+n)!)` is all-positive, the no-mpmath path is
a truncated series with a rigorous geometric tail bound — unconditionally sound,
not a ulp-inflated guess — and refuses arguments too large to bound soundly.

### Fixed — `omnibias-pinn`: `perron_spectral_gap` could never reach the Lean kernel

`_perron_certificate` never called `seal_certificate()`, while `check_certificate`
refuses unsealed certificates before emitting any Lean. `generate_obligation`
succeeded (the math was Lean-ready) but `verify_certificate_digest` was always
`False`, so the kind could never earn `theorem_prover_verified`.
`test_perron_lean_check_flag_mirrors_kernel` passed only because no runner had
Lean installed, and would have failed the moment one did. The certificate is now
sealed, its schema validates the digest, and a **generic guard test** asserts that
every default-machine prover whose certificate yields a non-`None`
`generate_obligation` also passes `verify_certificate_digest`, so this class of
bug cannot recur silently.

### Fixed — `omnibias-core`: the `Prover` protocol demanded a settable `name`

`Prover` declared `name: str`, which requires a *mutable* attribute, so the repo's
own frozen `FunctionProver` did not satisfy its own protocol under a strict type
check. It is now a read-only property, which accepts both plain attributes and
properties. No caller assigned through a `Prover`-typed reference, so this is
backwards-compatible.

### Added — `omnibias-pinn`: deep fields, multi-scale, balancing, conservation, decomposition

Before this change the only trainable free-form PINN field on the substrate was
`OneLayerVectorField` — a *single* hidden layer. `JetMLP`, `FourierFeatureMLP`,
and `make_siren` already existed in `omnibias.{torch,jax}.architectures` with
the closed-form tower intact, but returned raw tensors, so they could not reach
the field operators, the cages, or the prebuilt PDE residuals. They now can.

- **Deep fields** (`omnibias.pinn.{torch,jax}.fields`): `JetMLPVectorField`,
  `FourierFeatureVectorField`, `make_siren_vector_field`, and
  `build_jet_mlp_vector_field`, on a new `jet_mlp` dispatch tag. Every partial
  is read off **one** memoised multivariate jet (`state.extra`, keyed by total
  order) rather than recomputed per axis, so an order-2 Navier–Stokes residual
  costs one order-2 jet for the whole residual; `gradient_full` / `hessian_full`
  / `laplacian` take that fast path. A `depth=1` `JetMLPVectorField` reproduces
  `OneLayerVectorField` derivatives exactly.
- **Multi-scale** (`omnibias.{torch,jax}.architectures.multiscale` +
  `MscaleVectorField` / `AdaptiveJetMLPVectorField`): `AdaptiveActivation` is
  the Jagtap adaptive activation `sigma(n a z)` built from the new backend-
  neutral `omnibias.core.spec.tempered` combinator, so a *trainable* frequency
  still gets the whole tower `(n a)^k sigma^(k)(n a z)` for free rather than a
  hand-written derivative. `MscaleMLP` is the MscaleDNN band mixture
  `u = sum_j f_j(alpha_j x)`, one exact jet per band. `suggest_frequency_bands`
  closes the loop from the existing `power_spectrum_per_d` / `spectral_fidelity`
  diagnostics back into band selection.
- **Loss balancing** (`omnibias.pinn._core.weighting` +
  `omnibias.pinn.{torch,jax}.losses.weighting`): a *stateful* `LossWeighter`
  with EMA and update cadence, gradient-norm annealing, and self-adaptive
  pointwise weights trained by gradient **ascent**. Today's `ntk_balanced_loss`
  is stateless and recomputed from scratch; these carry state across steps.
- **Causal time-marching** (`omnibias.pinn._core.marching`):
  `TimeWindowSchedule` turns `causal_residual_loss` into real marching — time-
  binned collocation sampling, warm start of window `k+1` from window `k`, a
  causality-tolerance advance criterion, and epsilon annealing.
- **Conservation cages** (`omnibias.pinn.{torch,jax}.cage`):
  `IntegralConservationField` generalises `omnibias-qpinn`'s
  `NormConservationField` into a domain-neutral cage holding
  `int sum_c u_c^p dx = C` by quadrature rescaling — exact at every optimiser
  step, including step 0, to quadrature accuracy. `FluxFormField` writes a flux
  as `G^i = sum_j d_j A^ij` with `A` antisymmetric, making `div G = 0` an
  algebraic identity rather than a penalty.
- **Non-local field**: `AttentionVectorField` (and `AttentionJetMLP` in
  `omnibias.{torch,jax}.architectures`) — a softmax mixture over a trainable
  memory whose *coordinate* derivatives are closed form to arbitrary order.
  `omnibias.hopfield` differentiates the same block with respect to the
  **scores**; a PDE needs `d/dx`, which is what the new jet primitives supply.
- **Domain decomposition** (`omnibias.pinn._core.interface` +
  `omnibias.pinn.{torch,jax}.losses.interface`): `Interface` / `InterfaceSpec`
  geometry with `interface_points` and `split_by_interface`, and the XPINN /
  cPINN residuals `value_jump`, `flux_jump`, `normal_derivative`,
  `interface_residual`, `interface_loss`. The seam sampler draws points **on**
  the interface in its own tangent coordinates rather than near it — the
  hand-rolled `10.0 * interface` penalty the tests used before measures the jump
  plus a discretisation error, and no amount of training removes the second
  part. The flux condition carries the material pair `(k_+, k_-)`, so a
  conductivity contrast is expressible at all. Everything reads the field
  through `state.ops`, so either side may be any field type.
- **Heterogeneous patches**: `PartitionedField` (torch and jax) accepts a
  `Sequence[FieldBase]` of *different* field types and sizes instead of forcing
  identical `OneLayerVectorField`s, validating that they agree on the coordinate
  and component specs. `build_partitioned_field` takes per-region `hidden` /
  `base`, or a `subfield_factory(region) -> FieldBase` for full control. Both
  backends' `OneLayerVectorField` gained `forward_values`, the one-line contract
  a composite needs from a sub-solution.

### Added — `omnibias.pinn.solver`: stiff integrators

`omnibias.pinn.solver.{torch,jax}.stiff`, bit-identical twins. Until now the
only implicit step was `implicit_linear_step`, which handles a linear *diagonal*
Fourier symbol on a periodic 1-D grid; everything else was explicit and capped
by the fastest decaying mode. Four families now cover the rest, all written as
ordinary differentiable functions so a step composes inside a training graph:
`rosenbrock_step` (ROS2, L-stable, one LU and two solves),
`exponential_rosenbrock_step` (exact on an affine right-hand side, at any step
size), `imex_euler_step` / `imex_cnab2_step`, and `etdrk4_step`.

- `phi_diagonal` / `phi_matrix` evaluate `phi_k` by a scaled Taylor series and
  exact doubling identities rather than the defining quotient, which cancels
  every significant digit as `z -> 0`: `phi_1(1e-14)` comes out as `1 + 5e-15`
  where `(exp(z) - 1) / z` is wrong in the fourth digit. Complex symbols work,
  since a Fourier-space `L` is one.
- `closed_form_jacobian` reads a stiff step's linearisation off **one** order-1
  multivariate jet of the `(W, b, spec)` layer stack — no autodiff graph, no
  finite difference. `dense_jacobian` is the honest autodiff fallback for an
  arbitrary callable.
- `SemiDiscrete` now carries `nonlinear` alongside `symbol`, so the linear /
  nonlinear split an IMEX or ETD scheme needs is stated once and `method_of_lines`
  accepts `"etdrk4"`, `"imex_euler"`, and `"imex_cnab2"`.
  `kuramoto_sivashinsky_semidiscrete` is the canonical stiff case; `stiff_rollout`
  composes any step into a differentiable trajectory.

### Added — `omnibias-torch` / `omnibias-jax`: non-elementwise jet algebra

`jet_reciprocal`, `jet_exp`, `jet_softmax`, and `jet_attention` in
`omnibias.{torch,jax}.jet_mv`, bit-identical twins. Until now the multivariate
jet chain was elementwise-only; these close it under division, exponentiation,
and normalisation, which is what lets an attention block sit inside
`mlp_jet_mv` with the tower intact. `jet_exp` reuses the `exp` activation's
derivative tower rather than forking a second one, and `jet_softmax` is
max-shifted for stability with the shift cancelling exactly in every
higher-order coefficient.

### Fixed

- **Documented API that never existed.** `docs/api/pinn.md` listed 21
  `omnibias.pinn.certified` members with no implementation behind them — a
  candidate-discovery-sprint surface (`candidate_family_catalog`,
  `run_fast_candidate_sprint`, and the per-stage functions around them) and a
  certified transfer-matrix / heat-kernel spectral-gap surface. The
  `navier-stokes-certified` cookbook additionally *ran* the sprint API in a
  fenced block. Since the package genuinely does not ship those capabilities,
  the claims are withdrawn rather than back-filled: documenting an unimplemented
  Navier-Stokes blow-up candidate pipeline is exactly the kind of capability
  assertion the honesty gates elsewhere in that module exist to prevent. All of
  `docs/api/` is now audited to zero undefined members.
- **`omnibias-fields` tests were not hermetic in dtype.** `test_finite_strain`,
  `test_mhd`, and `test_kinetic` each called `torch.set_default_dtype(float64)`
  at *import* time. That is a process-global mutation applied during collection,
  so omnibias-torch's own autouse dtype fixture would restore float32 before the
  fields tests ran, and five elasticity tests failed on mixed-dtype `einsum`
  whenever the two suites shared a pytest session. The default now comes from an
  autouse fixture in the fields `conftest.py`, matching the omnibias-torch
  precedent, so it holds per test regardless of collection order.

## [0.4.0] - 2026-08-04 — initial public release

First public release. Everything below is new to the outside world; nothing was
published before this tag, so there is no upgrade path to describe and no
deprecation to observe.

> This file starts here on purpose. Development happened in a private tree, and
> a changelog of pre-release churn against changes nobody could have depended on
> is noise, not history. From this entry forward, every behavioural change is
> recorded.

### The primitive

omnibias computes the **closed-form n-th derivative of an activation**,
`sigma^(n)(z)`, for arbitrary `n`, with bit-stable accuracy and a **single
`sigma` evaluation regardless of order**. The polynomial coefficients come from
one shared pure-Python module, so PyTorch, JAX, and Keras 3 are **bit-identical
by construction** rather than by test.

The math is the Riccati identity (`sigmoid' = s(1-s)`, `tanh' = 1 - t^2`) and
the Eulerian / Legendre / Hermite recurrences it implies, extended to deep
compositions and mixed partials by Bell / Faà di Bruno combinatorics.

`OperatorBlock` dispatches six roles — `identity`, `grad`, `laplacian`,
`derivative`, `band`, and `integral`. The `integral` role is a genuine
closed-form antiderivative window, `S(z+b_hi) - S(z+b_lo)` with `S' = sigma`,
not a quadrature.

### Packages

42 distributions, versioned independently.

**Stable core** — `omnibias-core` 0.4.0 (pure-Python math: polynomial
coefficients, Bell / Faà di Bruno, multi-index jets, the rigorous
`omnibias.core.verified` substrate and the `omnibias.core.proof` certificate
format), `omnibias-torch` 0.4.0, `omnibias-jax` 0.4.0, `omnibias-ferminet`
0.2.0.

**Beta** — `omnibias-fields` 0.1.0 (the field substrate: `FieldState`, views,
`SigmaCache`, and the torch / jax differential-operator surface),
`omnibias-geometry` 0.2.0 (metric, Christoffel, covariant derivative,
Laplace–Beltrami, curvature, geodesics, exterior calculus, gauge theory),
`omnibias-pinn` 0.1.0, `omnibias-symbolic` 0.1.0 (neural-jet equation
discovery), `omnibias-fractional` 0.1.0.

**Alpha** — 33 further packages spanning quantum PINNs, curvature and
second-order optimisation, measure integration, score / SDE, variational
calculus, the `delta -> 0` difference register, q-calculus, time-scale calculus,
holonomic / D-finite methods, quantisation, Boolean algebra, spiking neurons,
modern Hopfield attention, certified verification, validated dynamics, SOS
positivity, the Mathlib-backed formal checker, differentiable discrete
optimisation (QUBO / submodular / structured DP / tabular / partition /
combinatorics / NP-hard families / routing / logic), convex programming,
spectral graph operators, control, shape fields, and the consumer agent-skill
library. `omnibias-keras` 0.0.1a1 and `omnibias-qpinn` 0.0.2a1 are the earliest.

Full inventory with maturity tiers: [`docs/packages.md`](docs/packages.md).

### Licensing

Two tiers, and they are not the same licence:

- **Tier P — `Apache-2.0`** (28 packages): the derivative tower and everything
  built directly on it. No copyleft, express patent grant, no commercial
  licence ever required.
- **Tier C — `AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial`**
  (14 packages): the certified-decision layer.

The split is recorded in `[tool.omnibias.license_tiers]` in the root
`pyproject.toml` and enforced by
`packages/omnibias-core/tests/test_license_consistency.py`, which fails the
build if a permissive package ever gains a copyleft dependency. Metadata
follows PEP 639 (`License-Expression`), and the repository follows the REUSE
Specification. See [`LICENSING.md`](LICENSING.md).

### Rigour and honesty

Claims are tiered and the tiers are enforced, not asserted:

- `omnibias.core.verified` produces **sound, outward-rounded enclosures** —
  intervals, affine zonotopes, Taylor models, QR-Lohner validated flow,
  Lehmann–Maehly–Goerisch eigenvalue lower bounds. Every enclosure is tested to
  contain both a dense deterministic grid and a random sample of true values.
- Certificates are canonical, hash-sealed JSON (format v1), tamper-evident via
  `verify_certificate_digest`.
- `theorem_prover_verified` is earned **only** by a genuine `lake build` pass
  against the Mathlib-free Lean kernel in `formal/omnibias-verified-kernel`.
  Asserting it without a pass blocks the verdict; with no Lean toolchain the
  bridge degrades gracefully. `mathlib_verified` is a separate, distinct tier.
- Methods are labelled honestly throughout: closed-form, forward-mode autodiff
  of an analytic quantity, and grid-based approximation are never conflated.
  `omnibias-fractional` is explicitly *not* closed form. Exact submodular
  minimisation is P-class and is not dressed up as more.
- The two senses of "collapse" are kept apart in every source, doc, and skill,
  and a terminology guard fails the build if they are conflated: the founding
  **bias collapse** is the `delta -> 0` limit in which `K` parallel hyperplanes
  coalesce into one carrying the derivative tower `sigma^(K-1)`, while
  **temperature collapse** is the `beta -> inf` limit in which a single
  hyperplane sharpens into a 0/1 feasibility indicator.

### Engineering

- Documentation is **executable**: CI runs every fenced Python block in the
  docs, and opting a block out requires a stated reason.
- Guard tests cover leakage (no absolute paths, scheduler tokens, vendor names,
  or secrets in any readable file), terminology, conceptual lineage, licence
  consistency, packaging hygiene, the public API surface, and agent-skill
  drift. Each guard self-tests its own blocklist so it cannot go vacuous.
- CI: 42 per-package test jobs, cross-backend parity, `ruff`, `mypy --strict`
  on the T1 workspace plus a growing curated-beta allowlist, `mkdocs build
  --strict`, wheel build with `twine check`, clean-venv import smoke, CodeQL,
  OpenSSF Scorecard, dependency review, and SBOM.
- Releases publish through PyPI trusted publishing (OIDC) with SLSA build
  provenance attestations; no long-lived credential exists.

[Unreleased]: https://github.com/derivon-ai/omnibias/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/derivon-ai/omnibias/releases/tag/v0.4.0

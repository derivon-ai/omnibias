# Changelog

All notable changes to omnibias are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and each of the 42
distributions is versioned independently under semantic versioning.

## [Unreleased]

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

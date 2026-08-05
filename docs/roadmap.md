# Roadmap (sanitized)

omnibias has a long backlog of derivative-tower paradigms it could
serve. This page lists the roadmap *categories* and the reserved
namespaces; per-paradigm proofs and de-risking arguments live in the
private enterprise backlog and ship to enterprise users
under NDA on request.

## Reserved namespaces

| package | scope | status |
|---|---|---|
| `omnibias-pinn` | Physics-informed NNs with closed-form Laplacian / biharmonic / curl / div / p-Laplacian operators. | **Beta (v0.1.0)** |
| `omnibias-fields` | Backend-agnostic field substrate (`FieldState`, `SigmaCache`, attribute-DSL views, op registry) + cross-backend (torch + jax) closed-form differential-operator surface (gradient, divergence, curl, laplacian, hessian, jacobian, integration, Sobolev norms, tensor divergence, Wirtinger). | **Beta (v0.1.0)** |
| `omnibias-geometry` | Differential geometry on manifolds: metric, Christoffel, covariant derivative, Laplace-Beltrami, Riemann / Ricci / scalar curvature, geodesics, exterior calculus (`d`, wedge, Hodge star), pullback metric `g = JᵀhJ` of a learned chart. | **Beta (v0.2.0)** |
| `omnibias-qpinn` | Quantum-physics PINNs (TISE / TDSE / NLS / Helmholtz / Klein-Gordon / Dirac) with norm / Bloch / Hermitian / nuclear-cusp cages, a closed-form molecular local-energy surface, and bit-parity torch + JAX backends. | **Alpha (v0.0.2a1)** |
| `omnibias-curvature` | Second-order optimization, Fisher matvecs, Riccati-Hessian KFAC, Sobolev / Jacobian regularization. | **Alpha (v0.1.0a1)** |
| `omnibias-symbolic` | Neural-jet equation discovery (library-free SINDy variant), AutoML surrogates, PDE operator-coefficient recovery, multivariate vector calculus & PDE discovery, differential geometry, exterior calculus, information theory, optimal transport, information geometry. Powers the [Discovery & Calculus Handbook](handbook/index.md). | **Alpha (v0.1.0)** |
| `omnibias.geometry.gauge` | Non-abelian gauge engine: compact Lie algebras, `F = dA + g[A, A]`, covariant divergence `D*F`, Bianchi, action density, topological charge, gauge flow / Langevin (torch + JAX), SU(2) lattice MC driver. | **Alpha (submodule of `omnibias-geometry`)** |
| `omnibias-score` | Score / SDE operators (score, Ito generator, Fokker-Planck) composed from the `omnibias-fields` closed-form primitives. | **Alpha (v0.1.0a1)** |
| `omnibias-fractional` | Fractional calculus (Grünwald–Letnikov / Riemann–Liouville / Caputo / spectral) on uniform / periodic grids. Non-local, grid-based — see [Scope & guarantees](scope-and-guarantees.md). | **Alpha (v0.1.0)** |
| `omnibias-keras` | Keras 3 unified backend: `OperatorBlock`, `cmbDense`, growable units running on TensorFlow / JAX / PyTorch. | **Alpha (v0.0.1a1)** |
| `omnibias-convex` | Certified LP / QP layers and differentiable + certified convex solvers built on `omnibias-fields` calculus. | **Alpha (v0.1.0a1)** |
| `omnibias-binary` | Binary / ternary / k-bit quantized NN training (replace STE with `tanh(beta * z)` whose every derivative is in closed form). | **Alpha (v0.1.0a1)** |
| `omnibias-boolean` | Differentiable Boolean algebra: exact ANF/Reed-Muller & Walsh spectra, Boolean differential calculus, reproductive equation solving (eliminant + GF(2) fast-path), and a beta-annealed propose-and-verify soft-gate solver. | **Alpha (v0.1.0a1)** |
| `omnibias-spiking` | Spiking NN training (LIF / IF, surrogate gradients = closed-form derivatives of smooth Heaviside approximations). | **Alpha (v0.1.0a1)** |
| `omnibias.score.flow` | Continuous normalizing flows with exact trace-of-Jacobian via the `omnibias-fields` divergence. | **Alpha (submodule of `omnibias-score`)** |
| `omnibias-hopfield` | Modern Hopfield / attention-as-operator with closed-form log-sum-exp Jacobian / Hessian. | **Alpha (v0.1.0a1)** |
| `omnibias-verify` | Certified NN verification: `TaylorModelMV` propagation, ReLU / GELU / max-pool enclosures, branch-and-bound, and robustness / Lipschitz / monotonicity / reachable-set certificates (tighter than IBP); torch + jax ingestion. | **Alpha (v0.1.0a1)** |
| `omnibias-dynamics` | Validated dynamics on the exact variational tower: QR-Lohner variational / monodromy flow, Poincaré-section enclosures, certified Lyapunov bounds, periodic-orbit proofs via the radii polynomial. | **Alpha (v0.1.0a1)** |
| `omnibias-graph` | Differentiable spectral graph operators (graph / normalized / random-walk Laplacians, spectral embedding, heat kernel, Rayleigh-Ritz cut relaxation) and temperature-controlled combinatorial relaxations (Sinkhorn, Gumbel-Sinkhorn, SoftSort, soft top-k); verified ring-graph eigenvalue certificate; exact NP-hard solving out of scope. | **Alpha (v0.1.0a1)** |
| `omnibias-struct` | Certified differentiable dynamic programming: soft Viterbi / shortest-path / CTC -- plus soft-DTW, sequence alignment (global / local Smith-Waterman / affine-gap Gotoh), soft value iteration / planning, monotonic alignment search, and structured attention -- via a `lse_beta` relaxation (`beta -> inf` temperature axis) differentiated exactly by the `delta -> 0` softplus/sigmoid tower. A **semiring / hypergraph driver** (`MaxPlus`/`Log`/`Counting`) generalises the substrate and reproduces the hand-written layers bit-for-bit, carrying the tree / grammar families -- **CKY inside-outside**, **Eisner projective dependency**, and the **matrix-tree** non-projective (exact Kirchhoff determinant) marginals -- plus distribution operators (path entropy, exact sampling, exact k-best). Closed-form `log(N)/beta` gap certificate self-checked against brute-force hard DP, higher-order `dp_value_jet` / `cky_lse_jet` / `eisner_lse_jet` + curvature bridge, verified interval enclosures, and sealed certified decoding (chain + DAG); bit-identical torch + jax twins. | **Alpha (v0.1.0a1)** |

The **formal loop** ships alongside these: `omnibias.core.verified` (rigorous
interval / affine / Taylor-model arithmetic, radii-polynomial existence, QR-Lohner
flow, certified eigenvalue bounds, and certified analytic-number-theory Dirichlet /
zeta / `L`-function enclosures on `Re(s) > 1` — no continuation, no Riemann
Hypothesis) and `omnibias.core.proof` (hash-sealed
certificate format v1 + the `lean_check` bridge) feed a Mathlib-free Lean 4 kernel
in `formal/omnibias-verified-kernel` that kernel-checks a certificate's finite
rational obligations and sets `theorem_prover_verified` on a genuine `lake` pass.

Implemented packages export real, tested APIs with their own `README.md`, CI job,
and torch/jax backends; any namespace still marked *Planning* ships an
`__init__.py` that raises with a clear pointer to this roadmap. Per-paradigm
derivations ship under NDA.

## Implementation philosophy

omnibias does not promise to ship every paradigm. The strict
rule for any new package being promoted out of the *Planning* status:

1. **Math is settled.** A peer-reviewable derivation in
   the private enterprise backlog shows why the closed-form derivative
   beats the naive autograd path.
2. **Three independent test cases.** At least three downstream call
   sites with regression tests pinned to bit-stable or 1-sigma
   tolerances.
3. **Cross-backend parity.** Every public API works on at least one of
   `omnibias-torch` / `omnibias-jax`, and high-priority APIs work on
   both with bit-identical polynomial coefficients.

## Carry-over from v0.1

v0.1 had several JAX-only experimental modules
(`lattice_attention`, `lattice_symmetry`, `slater_updates`,
`annealing`, `autoregressive`) that are *not* in the stability
contract -- they live in the archived internal research workspace and are free to break.
If any of them prove out across multiple research projects, the
*Planning* gate above is the path back into the public namespace.

## omnibias-fields (Beta promotion, v0.1.0)

`omnibias-fields` was extracted from `omnibias-pinn` as the shared backend-agnostic
substrate consumed by `omnibias-pinn`, `omnibias-geometry`, `omnibias-score`, and
the gauge / qpinn certified stacks. It ships at **Beta** because it meets all three
gate criteria:

1. **Math is settled.** The closed-form differential-operator surface
   (`gradient`, `divergence`, `curl`, `laplacian`, `hessian`, `jacobian`,
   tensor divergence, Wirtinger derivatives, integration, Sobolev norms) is a
   direct dispatch on the omnibias closed-form `sigma^(n)(z)` tower; the
   field-substrate value objects (`FieldState`, attribute-DSL views, lazy
   `SigmaCache`, `ops_registry`) carry no per-backend math.
2. **Multiple independent test cases.** The test suite under
   `packages/omnibias-fields/tests/` covers state schemas, the op registry,
   integration, dispatch, the torch ops, the jax ops, and cross-backend
   parity. Downstream packages (`omnibias-pinn`, `omnibias-geometry`,
   `omnibias-score`) re-exercise the substrate through hundreds of additional
   tests.
3. **Cross-backend parity.** Every torch op has a bit-identical JAX twin
   (`omnibias.fields.torch.ops.<X>` ↔ `omnibias.fields.jax.ops.<X>`) verified
   to float64 ULP tolerance.

The public surface is the `omnibias.fields._core` schemas, the `ops_registry`
extension point, and the `omnibias.fields.{torch,jax}.ops` operator surfaces.
Existing `omnibias.pinn._core` and `omnibias.pinn.<backend>.ops` imports keep
working unchanged through back-compat shims.

## omnibias-geometry (Beta promotion, v0.2.0)

`omnibias-geometry` was promoted from Alpha to **Beta** in v0.2.0. It meets
all three gate criteria:

1. **Math is settled.** Two exact mechanisms — closed-form sigma-tower for
   field-function derivatives, exact forward-mode autodiff for metric
   derivatives — both validated against a sympy symbolic reference and
   against the analytic round sphere (`scalar curvature = 2/R²`,
   `Ricci = g/R²`, `Δ_{S²} cos(θ) = -2 cos(θ)`). Index conventions and
   per-operator derivations are in
   [`GEOMETRY_DERIVATIONS.md`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-geometry/GEOMETRY_DERIVATIONS.md).
   The "two exact mechanisms" framing is enshrined in
   [Scope & guarantees § 1](scope-and-guarantees.md#1-the-three-kinds-of-exact).
2. **Multiple independent test cases.** The test suite under
   `packages/omnibias-geometry/tests/` covers the metric / Christoffel /
   covariant derivative / Laplace-Beltrami / Riemann / Ricci / scalar
   curvature / geodesics, exterior calculus (`d`, wedge, Hodge star,
   codifferential), and the pullback metric `g = JᵀhJ` of a learned chart,
   plus a torch ↔ jax cross-backend bit-parity test.
3. **Cross-backend parity.** Torch and JAX ops agree to `rtol = 1e-9` in
   float64 (verified by
   `packages/omnibias-geometry/tests/test_cross_backend_parity.py`).

The public surface is `ManifoldSpec`, `MetricSpec`, `ChartSpec`,
`metric_spec_from_chart`, `pullback_metric`, and the
`omnibias.geometry.{torch,jax}.ops` op surfaces (`metric`, `christoffel`,
`covariant_derivative`, `laplace_beltrami`, `riemann_tensor`,
`ricci_tensor`, `scalar_curvature`, `geodesic_rhs`, plus the exterior
calculus operators).

## omnibias-pinn (Beta promotion, v0.1.0)

`omnibias-pinn` was promoted from *Planning* to **Beta** in v0.1.0
(May 2026). Beta status means the package satisfies all three of
the gate criteria:

1. **Math is settled.** Derivations live in
   `docs/pinn-derivations.md`; structural-constraint
   derivations live in `docs/theory.md` Section 9.
2. **Three independent test cases.** 16 integration tests in
   `packages/omnibias-pinn/tests/integration/` verify residual /
   loss parity with the in-tree research code for 2D Navier-Stokes,
   Kuramoto-Sivashinsky, and Cahn-Hilliard. Plus a 3D Navier-Stokes
   sweep in the internal benchmark archive (TGV smoke +
   Kolmogorov Re=100 / 1000); see [`benchmarks.md`](benchmarks.md).
3. **Cross-backend parity.** 620 / 620 cross-backend tests pass,
   verifying bit-identical numerics between
   `omnibias.pinn.torch.*` and `omnibias.pinn.jax.*` to float64-ULP
   tolerance.

Per the stability matrix in `docs/stability.md`, *most* of the
omnibias-pinn surface is `stable` for v0.1.0; the experimental
parts are `ChebyshevVectorField`, the Helmholtz / hard-boundary /
mass-flux cages, and `diagnostics.field_stability`.

The discontinuity-capturing `omnibias.pinn.partition` bridge (an optional
`[partition]` extra on the alpha `omnibias-partition` keystone) now ships
**both** a torch and a JAX `PartitionedField` twin, checked bit-for-bit on
identical parameters (`tests/partition/test_partitioned_field_jax.py`). Its
derivatives use the autodiff product rule in both backends (the closed-form
`sigma`-tower does not cover products of sigmoids).

## omnibias-qpinn (Alpha, v0.0.2a1)

`omnibias-qpinn` is **Alpha** at v0.0.2a1 (May 2026). Alpha
status reflects that the package satisfies the *math is settled*
criterion and has cross-backend parity, but full-fidelity benchmark
numbers still need to be locked down on a GPU cluster:

1. **Math is settled.** Per-equation split-real derivations live in
   `packages/omnibias-qpinn/QPINN_DERIVATIONS.md`, covering
   time-independent / time-dependent / nonlinear Schrodinger,
   Helmholtz, Klein-Gordon, and Dirac residuals plus the
   norm / Bloch / Hermitian cages, and (sections 9-11) the molecular
   Born-Oppenheimer local energy, the nuclear-cusp cage, and the
   closed-form Pade-Jastrow correlation factor.
2. **Three independent smoke cases.** Smoke integration tests in
   `packages/omnibias-qpinn/tests/integration/` verify training
   convergence for QHO ground state (TISE), Gaussian wavepacket
   (TDSE), and dark soliton (NLS); the three relativistic
   equations have their own smoke cases (Helmholtz scattering,
   `phi^4` KG kink, Dirac plane wave). Full-fidelity numbers will
   remain in the internal benchmark archive (see
   [`benchmarks.md`](benchmarks.md)) before package promotion.
3. **Cross-backend parity.** All public residuals and the
   norm / Bloch / Hermitian cages pass `rtol=1e-9, atol=1e-12`
   bit-parity tests in
   `packages/omnibias-qpinn/tests/cross_backend/`.

Until a stable release the surface is *not* part of the API stability
contract. Expect refinements around the Bloch cage (currently only up
to 2nd-order single-axis derivatives), the Hermitian helpers, and the
spinor DSL between alpha releases.

The direct **Galerkin eigensolvers** (`omnibias.qpinn.torch.eigensolvers`) are
**torch only** in the alpha: the Galerkin assembly is tied to the torch field
basis + closed-form Laplacian and `scipy.linalg.eigh`. A JAX twin
(`jax.scipy.linalg.eigh` over an `omnibias.pinn.jax` basis) is a tracked
roadmap item; `omnibias.qpinn.jax` ships no partial `eigensolvers` submodule
until it lands. The equation / cage / diagnostics residuals stay bit-identical
across both backends.

## omnibias-curvature (Alpha, v0.1.0a1)

`omnibias-curvature` has moved beyond namespace reservation. The current
alpha surface provides closed-form one-layer parameter gradients,
Hessians, Gauss-Newton Fisher matrices, Newton steps, and KFAC factors
for Riccati-class fields. It remains outside the stable workspace because
the Torch port, multi-layer assembly, and `kfac_jax` integration are not
yet complete.

## omnibias-symbolic (Alpha, v0.1.0)

`omnibias-symbolic` is **Alpha** at v0.1.0. It promotes the equation-discovery
research engine into the public namespace:

1. **Math is settled.** The discoverer treats every column of the activation
   derivative jet `y, dy, d2y, ...` as an exact closed-form quantity, then fits a
   sparse implicit relation over generic jet coordinates (a library-free SINDy
   variant). Because the jets are exact, the recovered relations are identities,
   not approximations.
2. **Multiple independent test cases.** The test suite in
   `packages/omnibias-symbolic/tests/` covers activation identities
   (`dy=y`, `d2y=-y`, `dy=1-y^2`), AutoML surrogate recovery, heat-equation
   operator-coefficient recovery, high-dimensional sparse recovery, the Blasius
   boundary layer, and a real-world tabular surrogate; the applied demos under
   `examples/symbolic_discovery/` add their own suites (battery, turbofan,
   finance, synthetic features, latent ODEs, dimensional groups, causal terms,
   joint operator regressor). Reproduced numbers are pinned in
   `examples/symbolic_discovery/PARITY.md`.
3. **Backend.** The jet engine uses the `omnibias-jax` closed-form activation
   fastpaths; the sparse / AutoML / PDE / analytic-Blasius paths are pure numpy.

Until a stable release the surface is *not* part of the API stability contract.

## Closed-form NN primitives (Alpha promotion, v0.1.0a1)

`omnibias-binary`, `omnibias-spiking`, `omnibias.score.flow`, and `omnibias-hopfield`
were promoted from reserved-namespace stubs to implemented **Alpha** packages.
Each ships closed-form derivative math reusing the shared `omnibias-core`
coefficients (no per-backend fork) with bit-checked torch + jax twins:

1. **Math is settled.** `binary` uses the `tanh(beta z)` Riccati derivative
   (`tanh' = 1 - tanh^2`); `spiking` uses exact surrogate gradients
   (fast-sigmoid / Gaussian) from the activation dictionary; `flow` computes the
   exact CNF trace-of-Jacobian through the `omnibias-fields` divergence;
   `hopfield` uses the closed-form log-sum-exp Jacobian (`softmax`) and Hessian.
2. **Tests.** Each package ships forward / backward correctness and
   cross-backend parity tests (float64-pinned) under its `tests/` tree.
3. **Cross-backend parity.** Torch (`autograd.Function` / module) and jax
   (`custom_vjp` / functional) backends agree to `rtol=1e-9` in float64 where
   both apply.

Until a stable release these surfaces are *not* part of the API stability
contract.

## Multi-layer jet composition (core / jax / torch, v0.3.0)

The stable backends gained an exact **multi-layer directional jet** primitive.
`omnibias.core.bell` adds the pure-Python Bell-polynomial / Faà di Bruno
combinatorics, and `omnibias.jax.jet` / `omnibias.torch.jet` (bit-identical
twins) propagate a truncated Taylor jet along a line `x(t) = x0 + t v` through a
deep network. Because the per-layer `sigma^(k)` are the closed-form omnibias
fast paths, the resulting directional derivative tower is exact (matching
`jax.experimental.jet` and nested autodiff in float64) rather than an autodiff
or finite-difference approximation, and high orders stay at machine precision
where finite differences fail. Deep Hessians follow by polarization of
directional 2-jets.

## Multivariate (multi-index) jet composition (core / jax / torch, v0.4.0)

The directional primitive was generalised to the **full multivariate
(multi-index) Faà di Bruno** jet. `omnibias.core.multi_index` adds the
pure-Python multi-index ordering and truncated Cauchy-product table, and
`omnibias.jax.jet_mv` / `omnibias.torch.jet_mv` (bit-identical twins) carry the
entire truncated Taylor expansion of a deep network around `x0`. A single
forward pass therefore yields *every* mixed partial derivative up to total order
`N` -- the gradient, the full Hessian, and higher-order tensors -- exact in
float64 (matching nested autodiff and tied back to the directional kernel by
restriction). The directional jet is now exactly the 1-D restriction of this
primitive.

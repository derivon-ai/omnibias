# Package index

omnibias ships **42 distributions** from a single [uv](https://github.com/astral-sh/uv)
workspace monorepo. Each package's version is the single source of truth in its
own `pyproject.toml`, and maturity is the package's own `Development Status`
classifier. **Track** -- the *curated public core* versus the *extended set* --
is a separate release decision; see the
[API-stability contract](stability.md).

## Curated public core (8)

Published first and held to the [API-stability contract](stability.md).

| Package | Version | Status | Scope |
|---|---|---|---|
| omnibias-core | 0.4.0 | Beta | Pure-Python closed-form n-th derivative core: Eulerian / Legendre / Hermite coefficient generators, the backend-agnostic `ActivationSpec`, plus gated Wave-1 algebra (`MultiPackSpec`, `BankSpec`) and Wave-3 `MollifierSpec` / `BandPlan` / `FrameSpec` / locus / conjugate Hilbert / ladder / transfer / tanh-method. |
| omnibias-torch | 0.4.0 | Beta | PyTorch backend: OMBU, operator-typed blocks, closed-form activation-derivative kernels, reference PINN / CmbNet / CvxLayer architectures; gated `MultiPackUnit` / `BiasScan` / `ScanNet` / `JetKAN` / `LadderNet` / `EquivariantScan` / hierarchical scan. |
| omnibias-jax | 0.4.0 | Beta | JAX backend: closed-form n-th derivative kernels, neural-field Laplacian / Hessian, Born-Oppenheimer derivative tools for VMC; gated `init_multipack` / `bias_scan` / Scan-Net / Jet-KAN / ladder / equivariant-scan twins. |
| omnibias-ferminet | 0.2.0 | Beta | FermiNet bridge: folx-compatible Laplacian, restricted Tier-2 ansatz, analytic nuclear Hessian / Born-Oppenheimer primitives. |
| omnibias-fields | 0.1.0 | Beta | Backend-agnostic field substrate (`FieldState`, attribute-DSL views, sigma^(n) cache) and the closed-form differential-operator surface (grad / div / curl / laplacian / hessian / jacobian, integration, Sobolev norms); gated weak-form VPINN (`omnibias.fields.weak`) and equality-locus layer (`omnibias.fields.locus`). |
| omnibias-pinn | 0.1.0 | Beta | Physics-informed neural networks: typed fields, hard-conservation cages, PDE residuals, diagnostics; alpha `train` / `domain` / `operator` / `solver` (four-gap gated); gated `omnibias.pinn.interface` transmission PINN (not XPINN `_core.interface`) plus gated travelling / layered / BEM / linearizing-transform submodules. |
| omnibias-geometry | 0.2.0 | Beta | Differential geometry: metric, Christoffel, covariant derivative, Laplace-Beltrami, Riemann / Ricci / scalar curvature, geodesics, exterior calculus, learned-chart pullback metric; gated chart scan and Wilson-line holonomy band. |
| omnibias-keras | 0.0.1a1 | Alpha | Keras 3 unified backend: OMBU, operator blocks, and drop-in `cmbDense` / `cmbConv` layers on TensorFlow / JAX / PyTorch. |

## Extended set (34, Alpha)

Real, tested torch/jax math with a dedicated CI job each, but **not** under the
API-stability contract -- the public surface may shift between alpha releases.

### Physics, fields & calculus registers

| Package | Version | Status | Scope |
|---|---|---|---|
| omnibias-qpinn | 0.0.2a1 | Alpha | Quantum PINN residuals and conservation cages for Schrodinger / Gross-Pitaevskii / Helmholtz / Klein-Gordon / Dirac (torch + jax). |
| omnibias-fractional | 0.1.0 | Alpha | Fractional calculus in two honest classes: grid/spectral GL / RL / Caputo (non-local, numerical) and a closed-form analytic fractional derivative on analytic functions. |
| omnibias-measure | 0.1.0a1 | Alpha | Autograd-native measure integration: a `Measure` abstraction (pushforward / product / importance reweighting) and the measure integral `int f dmu`, layer-cake / distribution-function, importance-sampling and simple-function primitives, with trainable torch / jax layers. |
| omnibias-score | 0.1.0a1 | Alpha | Score / SDE operators: closed-form score (grad log p), the Ito generator, and the Fokker-Planck adjoint, composed from the fields grad / Hessian primitives. |
| omnibias-variational | 0.1.0a1 | Alpha | Least-action / variational calculus: action integrals, Euler-Lagrange / Euler-Poisson functional derivatives, Hamiltonian / Noether, symplectic integrators, rigorous action enclosures. |
| omnibias-difference | 0.1.0a1 | Alpha | The founding delta->0 register: certified finite-difference -> derivative extraction, umbral / Sheffer calculus, asymptotic-coefficient reading (Stirling / Bernoulli / Euler), gated exact-Q irregular Birkhoff stencils (01-04). |
| omnibias-qcalculus | 0.1.0a1 | Alpha | Quantum / q-calculus: q-numbers, Gaussian q-binomials, the Jackson q-derivative / q-integral, q-exponentials; the q->1 limit recovers ordinary calculus. |
| omnibias-timescale | 0.1.0a1 | Alpha | Time-scale (Hilger) calculus unifying the continuous and discrete registers via delta / nabla derivatives; graininess mu->0 recovers the derivative tower. |
| omnibias-holonomic | 0.1.0a1 | Alpha | D-finite / holonomic engine: Ore (skew-polynomial) algebra, Gosper + creative telescoping, and Lean-certified binomial identities. |

### Curvature & second-order optimization

| Package | Version | Status | Scope |
|---|---|---|---|
| omnibias-curvature | 0.1.0a1 | Alpha | Closed-form parameter Hessian / Gauss-Newton Fisher / KFAC factors for one-hidden-layer Riccati fields. |

### Differentiable + certified optimization

| Package | Version | Status | Scope |
|---|---|---|---|
| omnibias-discrete | 0.1.0a1 | Alpha | Shared discrete-optimization substrate: the DiscreteProblem seam, annealed sigmoid relaxation, rounding + k-flip decoder, and a Lasserre / moment-SOS optimality-gap certificate; ships a MaxSAT front-end. |
| omnibias-qubo | 0.1.0a1 | Alpha | Differentiable + certified QUBO / Ising: annealed relaxation, 1-flip decoder, brute-force oracle, and a spectral / SOS-Lasserre gap certificate; max-cut / MIS front-ends. |
| omnibias-submodular | 0.1.0a1 | Alpha | Differentiable + certified submodular optimization: multilinear extension + continuous greedy, pipage / swap rounding, a (1 - 1/e) / curvature guarantee + gap sandwich, and exact P-class minimization. |
| omnibias-struct | 0.1.0a1 | Alpha | Certified differentiable dynamic programming: soft Viterbi / shortest-path / CTC via logsumexp_beta, differentiated exactly by the softplus / sigmoid tower, with a gap certificate vs hard DP; gated tropical homotopy (`omnibias.struct._core.tropical`). |
| omnibias-combinatorics | 0.1.0a1 | Alpha | Exact differentiable matching / flow / matroid layers: entropic relaxations onto integral polytopes with a tight LP-dual optimality-gap certificate. |
| omnibias-nphard | 0.1.0a1 | Alpha | Differentiable certified heuristics for named NP-hard families (QAP / GAP / scheduling) on omnibias-qubo, with an MCTS search track and honest (non-tight) gap certificates. |
| omnibias-routing | 0.1.0a1 | Alpha | Certified + differentiable routing: a poly-size TSP relaxation + 2-opt decoder + Neumaier-Shcherbina LP gap certificate; decision-focused predict-then-optimize. |
| omnibias-convex | 0.1.0a1 | Alpha | Differentiable + certified convex LP / QP: a closed-form-Hessian log-barrier interior-point solver, KKT implicit-function gradients, and verified optimality enclosures. |
| omnibias-sos | 0.1.0a1 | Alpha | Certified positivity: Sum-of-Squares / Positivstellensatz decompositions with a rigorous interval LDL^T PSD certificate that can earn `theorem_prover_verified`. |
| omnibias-graph | 0.1.0a1 | Alpha | Differentiable spectral graph operators (Laplacians, spectral embedding, heat kernel) and combinatorial relaxations (Gumbel-Sinkhorn, SoftSort, soft top-k); gated Face-Net on a sampled arrangement subgraph. |
| omnibias-logic | 0.1.0a1 | Alpha | Differentiable + certified Boolean logic: weighted MaxSAT plus (weighted) #SAT / model counting with inclusion-exclusion count enclosures. |
| omnibias-control | 0.1.0a1 | Alpha | Differentiable control with a model-relative safety certificate: a batched CBF-QP safety filter and a recoverable-set certificate. |
| omnibias-tab | 0.1.0a1 | Alpha | Differentiable, exactly second-order-trained, certified soft decision-tree ensembles for tabular data; benchmarked against LightGBM. |
| omnibias-partition | 0.1.0a1 | Alpha | Certified soft partition-of-unity primitive: soft-split gates hardening as beta->inf, a sound membership-gap certificate, a shared region-model registry, and gated arrangement geometry (`omnibias.partition.arrangement`). |
| omnibias-shape | 0.1.0a1 | Alpha | Differentiable soft shape / occupancy fields and soft-coverage (soft-OR / log-sum-exp union) operators with a closed-form derivative tower. |

### Learning primitives

| Package | Version | Status | Scope |
|---|---|---|---|
| omnibias-binary | 0.1.0a1 | Alpha | Closed-form quantization gradients for binary / ternary / k-bit training via the tanh-beta Riccati derivative. |
| omnibias-boolean | 0.1.0a1 | Alpha | Differentiable Boolean algebra: exact ANF / Reed-Muller and Walsh spectra, Boolean differential calculus, reproductive equation solving, and a soft-gate solver. |
| omnibias-spiking | 0.1.0a1 | Alpha | Spiking LIF / IF primitives with exact closed-form surrogate gradients. |
| omnibias-hopfield | 0.1.0a1 | Alpha | Modern Hopfield networks and attention-as-operator with a closed-form log-sum-exp Jacobian / Hessian. |
| omnibias-symbolic | 0.1.0 | Alpha | Neural-jet equation discovery (library-free SINDy), AutoML surrogates, PDE operator-coefficient recovery, and Blasius surrogates. |

### Verification, formal & dynamics

| Package | Version | Status | Scope |
|---|---|---|---|
| omnibias-verify | 0.1.0a1 | Alpha | Certified neural-network verification: Taylor-model / interval propagation (smooth + ReLU / GELU / max-pool) with branch-and-bound, yielding robustness / Lipschitz / monotonicity / reachable-set certificates. |
| omnibias-dynamics | 0.1.0a1 | Alpha | Computer-assisted dynamics: validated variational / monodromy flows, Poincare-section enclosures, certified Lyapunov bounds, and radii-polynomial periodic-orbit proofs. |
| omnibias-formal | 0.1.0a1 | Alpha | Mathlib-backed formal checker: drives the `formal/omnibias-analytic` Lean project to discharge a certificate's rational obligations, reporting a `mathlib_verified` tier. |

### Tooling

| Package | Version | Status | Scope |
|---|---|---|---|
| omnibias-skills | 0.1.0a1 | Alpha | Agent-skill library for building on omnibias: bundled Cursor / Claude Code Agent Skills (backends, fields/PINN, frontier, geometry, curvature, verify, symbolic) and an idempotent installer CLI. |

## Folded modules (not separate distributions)

Six names are frequently mistaken for standalone packages. They ship *inside*
their parent distribution and are guarded by
`packages/omnibias-core/tests/test_package_registry.py`:


- `omnibias.score.flow` -- continuous normalizing flows (in `omnibias-score`).
- `omnibias.pinn.solver` -- the PDE solver (in `omnibias-pinn`).
- `omnibias.pinn.operator` -- neural operator learning: DeepONet closed-form trunk jet through order 4 (KS residual unchanged) + FNO baseline + multi-head conditioning (in `omnibias-pinn`).
- `omnibias.pinn.train` -- causal marching drivers + causality / trivial-solution diagnostics + spectral band scheduler (in `omnibias-pinn`).
- `omnibias.pinn.domain` -- SDF / R-function geometry + distance-constrained hard BCs (in `omnibias-pinn`).
- `omnibias.geometry.gauge` -- the non-abelian gauge engine (in `omnibias-geometry`).


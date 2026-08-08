# AGENTS.md

Guidance for AI coding agents working in the **omnibias** repository.
Human contributors should read [`CONTRIBUTING.md`](CONTRIBUTING.md);
this file is the machine-oriented companion.

## What this project is

omnibias computes the **closed-form n-th derivative** of an activation,
`sigma^(n)(z)`, for arbitrary `n`, with bit-stable accuracy and a single
`sigma` evaluation regardless of order. The same primitive is exposed on
PyTorch, JAX, and Keras 3. The polynomial coefficients come from one
shared pure-Python module, so every backend is **bit-identical by
construction**.

The core math is the Riccati identity (`sigmoid'(z) = s(1-s)`,
`tanh'(z) = 1 - t^2`) plus the Eulerian / Legendre / Hermite polynomial
recurrences it implies.

## Repository layout

This is a [uv](https://github.com/astral-sh/uv) workspace monorepo.

```
packages/
  omnibias-core/        # pure-Python math (polynomials, Bell/Faa di Bruno).
                        #   NO torch/jax/keras imports. (stable)
  omnibias-torch/       # PyTorch backend; closed-form derivatives + jet ops (stable)
  omnibias-jax/         # JAX backend; closed-form derivatives + jet ops (stable)
  omnibias-ferminet/    # FermiNet bridge, depends on omnibias-jax (stable)
  omnibias-keras/       # Keras 3 unified backend: TF / JAX / torch (alpha)
  omnibias-fields/      # foundational field substrate: FieldState, views,
                        #   SigmaCache, ops_registry + torch/jax field ops
                        #   (grad/div/curl/lap/hess/jacobian, integration,
                        #   norms, tensor divergence, Wirtinger). (beta)
  omnibias-pinn/        # physics-informed NNs; builds on omnibias-fields,
                        #   re-exports the moved substrate via shims (beta).
                        #   Hosts omnibias.pinn.solver (mesh-free PDE solver),
                        #   omnibias.pinn.operator (DeepONet / FNO + conditioning),
                        #   omnibias.pinn.train (causal marching), and
                        #   omnibias.pinn.domain (SDF / hard curved BCs); all alpha.
  omnibias-qpinn/       # quantum PINNs (alpha)
  omnibias-curvature/   # closed-form Hessian / Fisher / KFAC (alpha)
  omnibias-geometry/    # differential geometry: metric, Christoffel, covariant
                        #   derivative, Laplace-Beltrami, curvature, geodesics,
                        #   exterior calculus, pullback metric of a learned chart
                        #   (g = J^T h J). Builds on omnibias-fields. (beta, v0.2.0)
                        #   Hosts omnibias.geometry.gauge: non-abelian gauge
                        #   theory, alpha (folded from omnibias-gauge).
  omnibias-fractional/  # fractional calculus (GL/RL/Caputo/spectral). NON-local,
                        #   grid-based approximations -- NOT closed form. (alpha)
  omnibias-measure/     # autograd-native measure integration: a Measure
                        #   abstraction (pushforward / product / importance
                        #   reweighting), the measure integral int f dmu, and
                        #   layer-cake / simple-function primitives, with
                        #   trainable torch / jax layers. (alpha)
  omnibias-score/       # score / SDE: score, Ito generator, Fokker-Planck,
                        #   composed from omnibias-fields ops. (alpha) Hosts
                        #   omnibias.score.flow: continuous normalizing flows,
                        #   alpha (folded from omnibias-flow).
  omnibias-variational/ # least action / variational calculus: action integrals,
                        #   Euler-Lagrange residuals, arbitrary-order Euler-Poisson
                        #   functional derivative + first (Gateaux) variation,
                        #   constrained (holonomic / isoperimetric) calculus,
                        #   Hamiltonian / Noether, classical field theory,
                        #   geodesics-as-least-action, symplectic integrators,
                        #   rigorous action enclosures. Builds on omnibias-fields
                        #   (+ optional geometry). (alpha)
  omnibias-difference/  # the founding delta->0 register: certified finite-
                        #   difference -> derivative extraction, umbral / Sheffer
                        #   calculus, asymptotic-coefficient reading (Stirling /
                        #   Bernoulli / Euler). (alpha)
  omnibias-qcalculus/   # quantum / q-calculus: q-numbers, Gaussian q-binomials,
                        #   the Jackson q-derivative / q-integral, q-exponentials;
                        #   the q->1 limit recovers ordinary calculus. (alpha)
  omnibias-timescale/   # time-scale (Hilger) calculus unifying the continuous and
                        #   discrete registers via delta / nabla derivatives;
                        #   graininess mu->0 recovers the derivative tower. (alpha)
  omnibias-holonomic/   # D-finite / holonomic engine: Ore (skew-polynomial)
                        #   algebra, Gosper + creative telescoping, and
                        #   Lean-certified binomial identities. (alpha)
  omnibias-symbolic/    # neural-jet equation discovery: library-free SINDy,
                        #   AutoML surrogates, PDE coeff recovery, Blasius.
                        #   Uses omnibias-jax fastpaths. (alpha)
  omnibias-binary/      # binary / ternary / k-bit quantization; closed-form
                        #   tanh-beta Riccati backward (torch + jax). (alpha)
  omnibias-boolean/     # differentiable Boolean algebra: exact ANF/Reed-Muller
                        #   + Walsh spectra, Boolean differential calculus,
                        #   reproductive equation solving (eliminant + GF(2)),
                        #   beta-annealed soft-gate solver. Builds on
                        #   omnibias-binary. (alpha)
  omnibias-spiking/     # spiking LIF / IF neurons; closed-form surrogate
                        #   gradients from the activation dictionary. (alpha)
  omnibias-hopfield/    # modern Hopfield / attention-as-operator; closed-form
                        #   log-sum-exp Jacobian / Hessian (torch + jax). (alpha)
  omnibias-verify/      # certified NN verification: TaylorModelMV propagation,
                        #   ReLU/GELU/max-pool enclosures, branch-and-bound, and
                        #   robustness / Lipschitz / monotonicity / reachable-set
                        #   certificates; torch + jax weight ingestion. (alpha)
  omnibias-dynamics/    # validated dynamics: QR-Lohner variational/monodromy
                        #   flow, Poincare-section enclosures, certified Lyapunov
                        #   bounds, periodic-orbit proofs (radii polynomial).
                        #   Pure-Python; builds on omnibias.core.verified. (alpha)
  omnibias-sos/         # certified positivity: Sum-of-Squares / Positivstellensatz
                        #   decompositions with a rigorous interval LDL^T PSD
                        #   certificate that can earn theorem_prover_verified.
                        #   (alpha)
  omnibias-formal/      # Mathlib-backed formal checker: drives the
                        #   formal/omnibias-analytic Lean project to discharge a
                        #   certificate's rational obligations, reporting a
                        #   mathlib_verified tier (distinct from the Mathlib-free
                        #   theorem_prover_verified kernel). (alpha)
  omnibias-discrete/    # shared differentiable + certified discrete-optimization
                        #   substrate: the DiscreteProblem seam, AnnealSchedule,
                        #   anneal_descent relaxation (torch + jax twins), rounding /
                        #   k-flip decoder + brute-force oracle, Lasserre/SOS + trivial
                        #   negative-coeff lower bounds, certify_gap. Extracted from
                        #   omnibias-qubo (which re-exports it unchanged); ships the
                        #   omnibias.discrete.maxsat consumer (weighted CNF -> a
                        #   pseudo-Boolean MaxSATProblem). (alpha)
  omnibias-qubo/        # differentiable + certified QUBO / Ising optimization:
                        #   annealed sigmoid relaxation (temperature collapse,
                        #   unrolled for backprop) + rounding/1-flip decoder +
                        #   brute-force oracle + certified optimality-gap sandwich
                        #   (SOS/Lasserre lower bound via omnibias-sos, or spectral /
                        #   box-QP seal via omnibias-convex); max_cut /
                        #   max_independent_set front-ends (torch + jax). Builds on
                        #   omnibias-discrete. (alpha)
  omnibias-submodular/  # differentiable + certified submodular optimization:
                        #   multilinear extension F(p)=E_{x~p}[f] + continuous
                        #   greedy (Frank-Wolfe over a matroid polytope; soft LP
                        #   oracle unrolled for backprop) + pipage/swap rounding
                        #   with the a-priori (1 - 1/e) / curvature-sharpened
                        #   guarantee + a min-of-bounds gap certificate
                        #   f(S) <= OPT <= U(S) self-checked vs brute_force_max;
                        #   accelerated (lazy/stochastic) + knapsack + non-monotone
                        #   (double/measured) + streaming maximizers; Coverage /
                        #   FacilityLocation / BudgetAdditive / LogDeterminant /
                        #   GraphCut families + Sum/Scaled/Saturated algebra over
                        #   uniform / partition / laminar / graphic / transversal /
                        #   intersection matroids and knapsack budgets (torch +
                        #   jax). Ships honest P-class EXACT submodular
                        #   minimization (Lovász + Fujishige-Wolfe) -- not a
                        #   P=NP claim. Builds on omnibias-discrete. (alpha)
  omnibias-struct/      # certified differentiable dynamic programming: soft
                        #   Viterbi / shortest-path / CTC (+ soft-DTW, alignment
                        #   global/local-SmithWaterman/affine-Gotoh, planning / soft
                        #   value iteration, MAS, structured attention) whose
                        #   logsumexp_beta relaxation (beta->inf temperature axis) is
                        #   differentiated exactly by the delta->0 softplus/sigmoid
                        #   tower via omnibias.{torch,jax}.jet. A semiring / hypergraph
                        #   driver (MaxPlus/Log/Counting) generalises the substrate,
                        #   reproduces the hand-written layers bit-for-bit, and carries
                        #   the tree/grammar families -- CKY inside-outside, Eisner
                        #   projective dependency, matrix-tree non-projective (exact
                        #   Kirchhoff determinant) -- plus distribution operators
                        #   (path entropy, exact sampling, exact k-best). Closed-form
                        #   log(N)/beta gap certificate vs brute-force hard DP,
                        #   higher-order dp_value_jet / cky_lse_jet / eisner_lse_jet +
                        #   curvature bridge, verified interval enclosures
                        #   (struct.verified) + sealed certified decoding chain+DAG
                        #   (struct.decode) (torch + jax). (alpha)
  omnibias-tab/         # differentiable + certified soft decision-tree ensembles
                        #   for tabular data: oblique soft-split gates sigmoid(beta
                        #   (w.x - t)) annealed beta->inf (temperature collapse,
                        #   the feasibility sense) from soft toward hard trees; a depth-1
                        #   additive (sum-of-sigmoids, certifiable) tier + a depth>=2
                        #   multiplicative (oblivious) tier; exact second-order joint
                        #   training over omnibias.torch.optim + a stagewise Newton-
                        #   boosting driver (closed-form loss curvature, GBM-mirror);
                        #   sound certificates (output bounds / Lipschitz / per-feature
                        #   monotonicity / sealed min via omnibias-verify + a certified
                        #   soft->hard rounding gap); bit-identical torch/jax forward;
                        #   benchmarked vs LightGBM. Builds on omnibias-discrete /
                        #   -curvature / -verify. (alpha)
  omnibias-partition/   # light, certified soft partition-of-unity keystone: depth
                        #   oblique split gates sigmoid(beta (w.x - t)) route into
                        #   2**depth regions with weights that are non-negative, sum
                        #   to one, and harden to a crisp partition as beta->inf
                        #   (temperature collapse, the feasibility sense, NOT the
                        #   founding delta->0 collapse). numpy partition_weights +
                        #   bit-identical torch/jax twins, hard_assignment /
                        #   hardened_rules, a SOUND soft->hard gap certificate
                        #   (Interval weight enclosure + log(n_regions)/beta), and a
                        #   RegionModels registry whose one combine() engine drives
                        #   four bridges: omnibias.pinn.partition (discontinuity PINN),
                        #   omnibias.geometry.atlas (region-wise metric),
                        #   omnibias.symbolic.piecewise (per-region SINDy), and
                        #   omnibias.struct.decision + omnibias.tab.decision (certified
                        #   decision layer). Depends only on omnibias-core (+ torch/
                        #   jax). (alpha)
  omnibias-combinatorics/
                        # exact differentiable matching / flow / matroid layers:
                        #   entropic relaxations onto integral polytopes with a
                        #   tight LP-dual optimality-gap certificate. (alpha)
  omnibias-nphard/      # differentiable certified heuristics for named NP-hard
                        #   families (QAP / GAP / scheduling) on omnibias-qubo,
                        #   with an MCTS search track and honest (non-tight) gap
                        #   certificates. (alpha)
  omnibias-routing/     # certified + differentiable routing: a poly-size TSP
                        #   relaxation + 2-opt decoder + Neumaier-Shcherbina LP gap
                        #   certificate; decision-focused predict-then-optimize.
                        #   (alpha)
  omnibias-convex/      # differentiable + certified convex LP / QP: a closed-form-
                        #   Hessian log-barrier interior-point solver, KKT
                        #   implicit-function gradients, and verified optimality
                        #   enclosures. (alpha)
  omnibias-graph/       # differentiable spectral graph operators (Laplacians,
                        #   spectral embedding, heat kernel) and combinatorial
                        #   relaxations (Gumbel-Sinkhorn, SoftSort, soft top-k).
                        #   (alpha)
  omnibias-logic/       # differentiable + certified Boolean logic: weighted MaxSAT
                        #   plus (weighted) #SAT / model counting with
                        #   inclusion-exclusion count enclosures. (alpha)
  omnibias-control/     # differentiable control with a model-relative safety
                        #   certificate: a batched CBF-QP safety filter and a
                        #   recoverable-set certificate. (alpha)
  omnibias-shape/       # differentiable soft shape / occupancy fields and
                        #   soft-coverage (soft-OR / log-sum-exp union) operators
                        #   with a closed-form derivative tower. (alpha)
  omnibias-skills/      # consumer agent-skill library: bundled Cursor / Claude
                        #   Code skills + an installer CLI, for BUILDING ON
                        #   omnibias; see "Agent tooling" below. (alpha)
formal/
  omnibias-verified-kernel/ # Mathlib-free Lean 4 kernel: sound ZInterval algebra
                            #   + finite rational obligation checker (sorry-free).
  omnibias-analytic/        # Mathlib-backed finite rational obligation checker
                            #   feeding the mathlib_verified tier (sorry-free).
examples/symbolic_discovery/  # recovered applied discovery experiments (synthetic
                        #   / battery / cmapss / financial / latent-ODE /
                        #   dimensional-groups / causal-term / joint)
tests/                  # cross-backend parity tests (torch <-> jax <-> keras)
docs/                   # mkdocs-material site
```

The **stable workspace** is `core` + `torch` + `jax` + `ferminet`. The
extension and keras packages ship their own `pyproject.toml` and are
installed per package.

## Build / test / lint commands

```bash
# Materialise the workspace
uv sync --all-extras --dev

# Run the installed workspace tests
uv run pytest

# A single package
python -m pytest packages/omnibias-core/tests -q

# Lint + strict typing (T1 packages)
uv run ruff check packages tests
uv run mypy --strict packages/omnibias-core/src packages/omnibias-torch/src \
  packages/omnibias-jax/src packages/omnibias-ferminet/src

# Docs (must be clean in strict mode)
mkdocs build --strict

# Docs are executable: every fenced ```python block is run
python -m pytest tests/test_docs_snippets.py -q
python tests/test_docs_snippets.py --all --only docs/cookbook   # triage report

# Keras 3 unified backend: pick the backend explicitly
KERAS_BACKEND=jax        python -m pytest packages/omnibias-keras/tests -q
KERAS_BACKEND=tensorflow python -m pytest packages/omnibias-keras/tests -q
KERAS_BACKEND=torch      python -m pytest packages/omnibias-keras/tests -q
```

## The derivative-tower contract (read before touching math)

- All polynomial coefficients live in
  `packages/omnibias-core/src/omnibias/core/polynomials.py`
  (`sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`,
  `hermite_coeffs`). Every backend imports these. Do **not** reimplement
  them per backend.
- Bell polynomials / Faà di Bruno combinatorics live in
  `packages/omnibias-core/src/omnibias/core/bell.py` (pure Python). The
  multi-layer **jet** kernels (`compose_jet`, `affine_jet`, `layer_jet`,
  `mlp_jet`, `tower_to_jet`/`jet_to_tower`) are bit-identical twins in
  `omnibias.jax.jet` and `omnibias.torch.jet`; they propagate exact directional
  Taylor jets through deep compositions using the closed-form `sigma^(k)`.
- The **multivariate** generalisation lives in `omnibias.core.multi_index`
  (pure-Python multi-index ordering + Cauchy-product table) and the bit-identical
  `omnibias.jax.jet_mv` / `omnibias.torch.jet_mv` kernels (`mlp_jet_mv`,
  `layer_jet_mv`, `compose_jet_mv`, `identity_jet`, `jet_partials` /
  `jet_gradient` / `jet_hessian`). One forward pass yields every mixed partial up
  to total order `N`; the directional kernel is its 1-D restriction.
- An activation is described by an `ActivationSpec`
  (`omnibias.core.spec`). Backends specialise the tensor type but share
  the metadata.
- A fastpath kernel computes `sigma^(n)(z)` directly. For `n < 0` it must
  raise `ValueError`. Orders that are genuinely unimplemented raise
  `NotImplementedError`.
- `OperatorBlock` dispatches on the op tag:
  `identity | grad | laplacian | derivative | band | integral`.

## Do

- Add a regression test for **every** behavioral change.
- Keep numerics bit-identical across backends; run the parity tests in
  `tests/` and `packages/*/tests/`.
- Default new tensors to the framework default dtype
  (`torch.get_default_dtype()` / `keras.config.floatx()`), never a
  hardcoded `float32`.
- Regenerate the `__all__` block at the bottom of an `__init__.py` when
  you add or remove a public symbol.
- Use vendor-neutral language in any tracked file: "GPU job" / "GPU
  cluster", never a specific scheduler, vendor, or local filesystem path. That
  also rules out the EDA-shop synonym for a compute cluster, scheduler commands
  and environment variables, developer usernames, and site scratch conventions --
  artifacts go to `$OMNIBIAS_SCRATCH`, defaulting to a repo-relative
  `artifacts/`. `packages/omnibias-core/tests/test_no_leakage.py` is the
  enforcing guard: it lists every blocked token, scans the whole readable
  surface (including `.github/`, `notebooks/`, and `formal/`), and self-tests
  its own blocklist so it can never go vacuous.
- Write documentation snippets that **actually run**: CI executes every fenced
  Python block in the docs (`tests/test_docs_snippets.py`), and a block may only
  rely on names defined earlier in the *same document*. Verify a call against the
  real signature before documenting it, rather than writing a plausible-looking
  one. Opting a block out needs a directive with a stated reason -- `signature`,
  `skip reason="..."`, `slow`, `raises=...`, or document-level `file-skip` -- so
  every exemption is reviewable.
- Make a new package **earn independent existence** before creating it: it must
  have a distinct *domain*, a distinct *dependency / maturity tier*, or a distinct
  *audience*. If it fails that test, ship it as a submodule of an existing package
  and promote it to its own distribution only once it earns independence. Folding
  a submodule back out later is cheap; un-shipping a premature distribution is not.
  See the `omnibias-dev-new-package` skill.

## Don't

- Don't import `torch`, `jax`, `tensorflow`, or `keras` from
  `omnibias.core`. The core is pure Python.
- Don't fork the polynomial coefficients per backend.
- Don't add scheduler-specific submission commands, internal hostnames,
  or absolute local paths to tracked files. The reproduction scripts
  live in the separate, private `omnibias_experiments` project (extracted
  from the formerly gitignored `internal/` tree).
- Don't bump package versions except where a task explicitly says to.
- Don't create stub / premature packages. A distribution that only re-composes
  another package's ops, or whose top-level module is a bare `__version__` shim
  with no distinct domain, should be a submodule instead. `omnibias-score` and
  `omnibias-qpinn` are the cautionary stubs that inflated the tree; `pde`, `gauge`,
  and `flow` were folded back into `pinn` / `geometry` / `score` for this reason.
  A folded package is de-wired everywhere at once (workspace exclude, CI job,
  `mkdocs.yml` nav + `paths`, `docs/api`, `llms.txt`, `CHANGELOG.md`, `AGENTS.md`);
  `test_package_registry` enforces the workspace / folded-name / Python-floor
  consistency, while CI matrix coverage, `mkdocs build --strict`, and the docs
  path guard cover the CI / docs / llms wiring separately.
- Don't restore the archived `research/`, `results/`, or `paper/` trees;
  they are intentionally not shipped.

## Calculus & differential-geometry layer

- The **field substrate** (`FieldState`, attribute-DSL views, `SigmaCache`,
  `ops_registry`, and the torch/jax field-operator surface) lives in
  `omnibias-fields` (`omnibias.fields._core` + `omnibias.fields.{torch,jax}`).
  It was extracted from `omnibias-pinn`; `omnibias.pinn._core` and
  `omnibias.pinn.<backend>.ops` are transparent re-export shims, so existing
  imports keep working and `FieldState` is a single class object.
- Backend ops select the closed-form sigma-tower path via the
  `_omnibias_dispatch` class marker (name = `omnibias.fields._core.DISPATCH_ATTR`)
  rather than importing concrete field classes -- this is why the foundational
  package never imports a downstream package.
- **Manifold ops** (`omnibias-geometry`): field-function derivatives are exact
  closed form; metric derivatives are exact forward-mode autodiff of the analytic
  metric (labelled honestly, not "closed form").
- **Fractional** (`omnibias-fractional`) is non-local / grid-based and is
  explicitly NOT closed form. **Score / SDE** (`omnibias-score`) is a pure
  composition of the `omnibias-fields` gradient / Hessian ops.
- New field-level ops are torch + jax only (matching `omnibias-pinn`); keras
  users still get bit-identical activation-level math via `OperatorBlock`.
- Typing tier: the rule, rather than a list that rots. Only the T1 workspace
  (`core` / `torch` / `jax` / `ferminet`) is on the blanket `mypy --strict` CI
  gate and in `[tool.mypy].mypy_path`. **Every other package is
  extension-tier** and is not strict-gated, because the bulk-copied torch/jax
  field ops carry systematic `no-any-return` / `type-arg` findings. Two curated
  beta packages (`omnibias-fields`, `omnibias-geometry`) plus the authored-strict
  `omnibias.pinn.operator` surface additionally gate an *incremental, growing*
  authored-strict surface through `scripts/mypy_strict_allowlist.txt` (the
  `mypy_curated_beta` job); add a module there once
  `mypy --strict --follow-imports=silent <file>` is clean.
  Newly authored modules are written strict-clean regardless of tier, and each
  package ships its own CI test job.

## Rigorous numerics & the formal loop

omnibias runs *one* derivative tower in three registers -- **differentiable**
(`omnibias.{torch,jax}`), **rigorous** (`omnibias.core.verified`), and **formal**
(`formal/`). The verified substrate and the formal bridge are pure-Python /
Lean-core and never import a backend.

- **Rigorous core** (`omnibias.core.verified`): `Interval` (outward-rounded),
  `affine` zonotopes (dependency cancellation), `TaylorModel`/`TaylorModelMV`,
  `sequence_space` (geometric-decay tail bounds), `kantorovich` (radii-polynomial
  existence), `lohner` (QR-Lohner / TM validated flow), `eig_operator`
  (Lehmann-Maehly-Goerisch eigenvalue *lower* bounds), `dirichlet` (Dirichlet /
  zeta / `L` / Jacobi-theta enclosures on `Re(s) > 1` only; continuation and the
  Riemann Hypothesis are recorded external obligations, never inferred). Every
  enclosure must contain a dense deterministic grid **and** a random sample of
  true values.
- **Certificate format v1** (`omnibias.core.proof.certificate`): canonical,
  hash-sealed JSON for `Interval`/`TaylorModel` enclosures; tamper-evident
  (`verify_certificate_digest`). `Verdict` carries `certificate_schema_version`
  and the kernel-earned `theorem_prover_verified`.
- **Formal loop** (`omnibias.core.proof.lean_check` + `formal/omnibias-verified-kernel`):
  the bridge extracts a certificate's *finite, rational* obligation (spectral-gap
  positivity, enclosed-quantity sign), emits Lean that chains the kernel's proven
  `ZInterval` soundness lemmas, and runs `lake build`. `theorem_prover_verified`
  is set **only** on a genuine kernel pass and can never be forged by the
  certificate; asserting the claim without a pass blocks the verdict. No Lean
  toolchain present -> the bridge degrades gracefully (flag stays `False`).
- The Lean kernel is deliberately **Mathlib-free** (Lean 4 core only) so CI
  kernel-checks it cheaply. Both Lean projects are **`sorry`-free**, and their
  scope is *finite, rational* obligations. Infinite analytic obligations --
  limits, continuum statements, asymptotics -- are out of scope and are not
  expressed in Lean at all, so they can never be silently discharged. Do not
  claim otherwise.

## Agent tooling (skills & rules)

Two persona-scoped agent-skill libraries ship with the repo, mirrored to Cursor
(`.cursor/skills/`) and Claude Code (`.claude/skills/`):

- **Consumer skills** (`omnibias-*`) teach an assistant how to *use* omnibias.
  They are the shippable `omnibias-skills` package -- canonical source in
  `packages/omnibias-skills/src/omnibias/skills/_bundled/skills/`; downstream
  users run `omnibias-skills install`. The repo's copies are installed + committed
  and a CI drift check (`omnibias-skills install --check`) keeps them
  byte-identical to the bundle. **Do not hand-edit `.cursor/skills/omnibias-*`** --
  edit the bundle and re-run the installer.
- **Maintainer skills** (`omnibias-dev-*`) teach an assistant how to *develop*
  omnibias. They are hand-authored **canonically in `.cursor/skills/`** and
  mirrored to `.claude/skills/` by `python scripts/sync_skills.py` (with `--check`
  in CI). Edit the `.cursor` copy, then re-run the sync.
- **Rules** (`.cursor/rules/`): one always-apply `omnibias.md` (universal
  invariants) plus path-scoped `.mdc` rules (`jax-tracing`, `verified-enclosures`,
  `formal-lean`). Root `CLAUDE.md` points Claude Code at this file.

## Where to look

- Full package inventory (all 42, with versions + maturity): [`docs/packages.md`](docs/packages.md).
- Activation math: `omnibias.core.polynomials`, `omnibias.core.spec`.
- Field substrate + field ops: `omnibias.fields`.
- Manifold geometry / exterior calculus: `omnibias.geometry`.
- Fractional / score-SDE: `omnibias.fractional`, `omnibias.score`.
- Boolean algebra (exact `_core` + differentiable torch/jax): `omnibias.boolean`;
  verified spectra + S-box cryptanalysis in `omnibias.boolean.{_core.verified,cipher}`.
- Rigorous numerics: `omnibias.core.verified`; certificates + Lean bridge in
  `omnibias.core.proof`; the Lean kernel in `formal/omnibias-verified-kernel`.
- Certified NN verification: `omnibias.verify`. Validated dynamics: `omnibias.dynamics`.
- PyTorch layers: `omnibias.torch.unit`, `.blocks`, `.growable`.
- JAX closed-form fields: `omnibias.jax.laplacian`.
- Keras 3 unified backend: `omnibias.keras`.
- Runnable examples: [`docs/examples/`](docs/examples/).
- Agent skills & rules: `.cursor/skills`, `.claude/skills`, `.cursor/rules`; the consumer package is `omnibias.skills` (`packages/omnibias-skills`).
- Public docs: [`docs/index.md`](docs/index.md).
- Benchmarks (vendor-neutral): [`docs/benchmarks.md`](docs/benchmarks.md).

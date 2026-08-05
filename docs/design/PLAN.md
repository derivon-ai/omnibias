# PLAN.md — calculus & differential-geometry layer

Tracking checklist for adding the missing calculus and differential-geometry
primitives to omnibias. The full design (layout, API, derivations, validation)
lives in [`calculus-geometry-plan.md`](calculus-geometry-plan.md).

## Status

- **Phases 0-8 implemented; last-mile CI/packaging fixes applied.**
- Four new packages ship: `omnibias-fields` (substrate + field ops),
  `omnibias-geometry` (manifold + exterior calculus), `omnibias-fractional`
  (grid-based, non-closed-form), and `omnibias-score` (Ito / Fokker-Planck).
  `omnibias-pinn` re-exports the moved substrate via transparent shims.
- Tests green: pinn+qpinn unchanged (866 passed / 7 skipped); new packages
  fields 25, geometry 12, fractional 5, score 4; cross-backend bit-parity in
  float64; `tests/test_reserved_stubs.py` updated (score is no longer a stub).
- Gates: `ruff check` clean on all new files (repo-wide count below the prior
  baseline); `mkdocs build --strict` clean; leakage grep clean; SPDX headers on
  every new source file. Each new package has a dedicated CI job.
- **Notebooks delivered:** `notebooks/09_fields_quadrature_norms`,
  `10_geometry_sphere_laplace_beltrami`, `11_fractional_diffusion`,
  `12_score_ou_generator` (CPU, executed clean via nbclient, output-stripped),
  plus the shared `notebooks/_fields.py` analytic-field helper.
- **Deferred (honest):** GPU-grid benchmarks live in the separate, private
  `omnibias_experiments` project. New field-layer packages stay at the extension typing tier (not
  on the `mypy --strict` gate), matching `pinn`/`qpinn`/`curvature`.
- No git commit has been made (awaiting explicit request).

## Conventions for every phase

Each phase ends only when **all** of the following are true:

- [x] Closed-form (or honestly-labelled non-closed-form) implementations on
      torch + jax.
- [x] Analytic + symbolic + autodiff + cross-backend parity + pinned regression
      tests, float64, documented tolerances.
- [x] `ruff check packages tests` clean on new files; `mkdocs build --strict`
      clean. (`mypy --strict` stays scoped to the T1 set; the new field-layer
      packages are extension-tier, like `pinn`/`qpinn`/`curvature`.)
- [x] Leakage grep clean; SPDX dual-license header on every new source file.
- [x] Docstrings + `docs/api/*.md` + derivations doc + cookbook updated.
- [x] `CHANGELOG.md` entry under `[Unreleased]`. (No commit yet — awaiting
      explicit request.)

---

## Phase 0 — research & design (STOP for review)

- [x] Explore the repo (core, pinn ops, FieldState, conventions).
- [x] Write [`calculus-geometry-plan.md`](calculus-geometry-plan.md).
- [x] Write this `PLAN.md` checklist.
- [x] Human review of both Phase 0 deliverables (signed off).

## Phase 1 — substrate extraction → `omnibias-fields`

- [x] Create `packages/omnibias-fields/` (pyproject, LICENSE, README, py.typed).
- [x] Move `omnibias.pinn._core.{state,view,components,coords,field_base,sigma_cache,ops_registry}`
      → `omnibias.fields._core.*`.
- [x] Move `omnibias.pinn.{torch,jax}.ops.*` + `_ops_dispatch` + `ops.registry`
      → `omnibias.fields.{torch,jax}.*`; `FieldState.ops` points at the new dispatch.
- [x] Keep `pde.py` / `registry.py` / `diagnostics.py` and the PINN field
      constructors in `omnibias-pinn`.
- [x] Add back-compat re-export shims in `omnibias.pinn._core` and
      `omnibias.pinn.{torch,jax}.ops` (transparent `sys.modules` aliases).
- [x] Add `omnibias-fields/src` to mkdocstrings paths. (Strict-mypy enrollment
      intentionally deferred — extension tier; see the design doc, question (d).)
- [x] Verify: existing pinn / qpinn / ferminet tests green unchanged; cross-backend
      parity + SigmaCache reuse preserved (marker-based dispatch, no import cycle).

## Phase 2 — integration, inner product, norms, tensor divergence

- [x] `omnibias.fields._core.quadrature` (`QuadratureSpec` + rule builders).
- [x] `ops.integral.integrate` (+ `quadrature_nodes`).
- [x] `ops.norms` (`inner_product`, `l2_norm`, `sobolev_norm`).
- [x] `ops.tensor.tensor_divergence` (flat Cartesian ∇·σ).
- [x] View sugar + `ops_registry` wiring; `FIELDS_DERIVATIONS.md`; tests.

## Phase 3 — differential geometry core (`omnibias-geometry`)

- [x] Create `packages/omnibias-geometry/`; `_core.manifold`
      (`MetricSpec`, `ManifoldSpec`). (`ChartSpec`/atlases deferred — single chart.)
- [x] Ops: `metric`, `inverse_metric`, `sqrt_det_metric`, `christoffel`,
      `covariant_derivative` (scalar/vector/one-form), `laplace_beltrami`,
      `geodesic_rhs`, `riemann_tensor`, `ricci_tensor`, `scalar_curvature`.
- [x] Sphere analytic + sympy symbolic + flat-metric parity + cross-backend
      validation; `GEOMETRY_DERIVATIONS.md`.

## Phase 4 — exterior calculus

- [x] `_core.forms.DifferentialForm`; ops `exterior_derivative`, `wedge`,
      `hodge_star`, `codifferential` / `hodge_laplacian_scalar`.
- [x] de Rham regression tests (`d²=0`; Hodge involution; `δd f = -Δ_g f`).

## Phase 5 — complex / Wirtinger calculus

- [x] `omnibias.fields.{torch,jax}.ops.complex` (`dz`, `dzbar`) over the
      qpinn split-real convention.
- [x] Cauchy–Riemann regression for the holomorphic `exp(z)`.

## Phase 6 — fractional calculus (`omnibias-fractional`)

- [x] Create `packages/omnibias-fractional/`; `_core.kernels` (GL/RL/Caputo/spectral).
- [x] Ops `caputo`, `riemann_liouville`, `grunwald_letnikov`,
      `spectral_fractional` — **honestly labelled non-closed-form**.
- [x] `D^α x^p` analytic checks; spectral integer/semigroup + cross-backend
      (`rtol=1e-9`); `FRACTIONAL_DERIVATIONS.md`.

## Phase 7 — stochastic generators (`omnibias-score`)

- [x] `score`, `ito_generator`, `fokker_planck` as compositions of the
      `omnibias-fields` gradient/Hessian ops.
- [x] Ornstein–Uhlenbeck stationary-density validation (`L* p_inf = 0`).

## Phase 8 — cross-cutting finish

- [x] Cross-backend parity + regression sweep across all new ops.
- [x] Docs: `docs/api/*.md` pages + mkdocs nav; geometry cookbook page.
      (Executable notebooks deferred — cookbook page for now.)
- [x] Update `AGENTS.md`, `.cursor/rules/`, `llms.txt`, `docs/llms.txt`,
      per-package READMEs.
- [ ] Heavy grid benchmarks (off-band, in the separate `omnibias_experiments` project) — deferred.
- [x] Final gates green. (No commit — awaiting explicit request.)

## Last-mile fixes (post-review)

- [x] Declare `omnibias-fields` dependency on `omnibias-pinn` (+ extras).
- [x] CI: install `omnibias-fields` in pinn/qpinn/docs jobs; add dedicated
      fields/geometry/fractional/score test jobs; drop `score` from the
      reserved-stub install loop.
- [x] Update `tests/test_reserved_stubs.py` (score is no longer a stub).
- [x] Add a curved-space vector `covariant_derivative` analytic test.

## Last-mile fixes (post-review, round 2)

- [x] Reconcile the design doc's strict-mypy section (§3/§8) with the resolved
      extension-tier decision (§13(d)); update the status banner.
- [x] Add torus + conformally-flat 2-metric symbolic (sympy) curvature tests and
      a torus analytic scalar-curvature check, plus their cross-backend parity.
- [x] Add a pinned/golden regression for the metric→curvature pipeline
      (`tests/data/curvature_sphere_golden.npz`, `rtol=1e-12, atol=1e-14`).
- [x] Deliver executable notebooks 09–12 (+ `notebooks/_fields.py`); indexed in
      `notebooks/README.md`.

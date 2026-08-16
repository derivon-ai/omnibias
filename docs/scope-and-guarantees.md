# Scope & guarantees

> One canonical page for what omnibias **does** guarantee, what it **doesn't**,
> and how every page on the site uses those words consistently. Cookbook pages
> link here rather than repeating the same disclaimer panel.

## 1. The three kinds of "exact"

omnibias is exact-by-construction wherever the math permits. The site uses
three precisely-defined labels:

| Label | What it means | Where it shows up |
|---|---|---|
| **Closed-form (sigma-tower)** | The output is produced by a polynomial recurrence on the activation. No autodiff graph, no finite differences. One forward pass at any derivative order. Bit-identical across backends because the [`omnibias.core.polynomials`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-core/src/omnibias/core/polynomials.py) recurrences are shared pure Python. | `omnibias.core`, `omnibias.{torch,jax,keras}`, `omnibias-ferminet`, `omnibias-pinn` field-function derivatives, `omnibias-symbolic` jets. |
| **Autodiff-exact** | The output is produced by forward-mode automatic differentiation of an analytic expression. Exact to machine precision — *not* a finite-difference approximation — just realised by a different exact mechanism than the sigma-tower. | `omnibias-geometry` metric derivatives (Christoffel / Riemann / Ricci / curvature). |
| **Numerical** | The output is a controlled numerical approximation (a sum, an FFT multiplier, a Monte-Carlo estimator). Accuracy is set by grid/spectrum/sample size and is documented per op. | `omnibias-fractional` (Grünwald–Letnikov / Caputo / spectral), the CCF Hilbert transform (periodic spectral multiplier), `omnibias.geometry.gauge` lattice Monte-Carlo. |

**Reading guidance for AI agents:** assume the result is closed-form unless the
docstring or this table says otherwise. Three labels, no surprises.

## 2. The derivative-order ceiling

`sigma^(n)(z)` is closed-form for every `n ≥ 0` when `sigma` is in the Riccati
class (`σ' = P(σ)`) or is the eigenfunction of `d/dz`. The classification is
**tight**: a theorem in [`docs/theory.md`](theory.md) shows the iff.

| Family | Max closed-form order `n` | Why |
|---|---|---|
| `sigmoid`, `tanh`, `softplus`, `exp`, `gaussian`, `sin`, `cos`, `sinh`, `cosh`, `smooth_sign`, `soft_relu`, `soft_step`, `soft_sign` | **unbounded** | Riccati or `d/dz` eigenfunction. |
| `silu`, `swish`, `gelu`, `mish` | **unbounded** | Analytic product `z · f(z)`: Leibniz over a Riccati / Hermite tower. |
| `relu`, `huber`, `elu`, `selu`, `celu`, `leaky_relu`, `prelu`, `relu6`, `hardtanh`, `hardsigmoid`, `hardswish`, `softshrink`, `hardshrink`, `threshold`, `softsign`, `abs`, `sign`, `step` | **unbounded** (a.e.) | Exact tower on each open piece. The singular part at the kink (a Dirac) is dropped by the almost-everywhere / regular-part convention. |
| `tan`, `cot`, `coth`, `sech`, `log_cosh` | 3 | Truncated Riccati table. |
| `arctan`, `log1pu2`, `softabs` | 2 | Mixed-class proximal. |

The a.e. row is the one to read carefully: the value returned is the
**regular part**, so `relu''` is `0`, not a delta. That is a modelling
convention, chosen because a float is what the caller can use — it is not a
claim that the distributional derivative vanishes.

`n < 0` raises `ValueError`. Genuinely unimplemented orders raise
`NotImplementedError` with a specific message — the
derivative-tower contract is enforced by every backend.
[`tests/test_doc_activation_orders.py`](https://github.com/derivon-ai/omnibias/blob/main/tests/test_doc_activation_orders.py)
parses this table and checks every row against the live registry on both
backends, so it cannot drift from the code.

## 3. The certified PDE stacks (Navier–Stokes / CCF)

The `omnibias.pinn.certified.*` modules are a **research infrastructure**: they
generate, schema-validate, independently replay, and machine-check certificates
for *finite, discrete* reduced models. They are not — and structurally cannot be
— a proof of global regularity for the continuum equations.

This is encoded in code, not just docs:

- Every artifact carries a `honesty: HonestyLabels` field. Defaults set
  `unproven_claim = False`, `three_d_claim = False`,
  `continuum_navier_stokes_claim = False`, `interval_verified = False`,
  `theorem_prover_verified = False`, `finite_energy_verified = False`.
- The schema validators (`ns_cap_schema_errors`, `cap_schema_errors`)
  **reject** any bundle that sets `unproven_claim=True` without all
  external-verification flags present.
- The numpy-only twin `omnibias.symbolic.navier_stokes` imports **nothing**
  from the `omnibias.pinn.certified` modules and re-checks every honesty flag
  independently. Tampered bundles fail the replay.

What this gives you in practice:

- **Reproducible numerical evidence**, independently replayed.
- **Falsifiable schemas** that reject overclaim by construction.
- **A precise, machine-checked enumeration** of exactly what remains to be
  proved.

What it does not give you: a proof of the continuum problem.

### 3.2 A-posteriori neural PDE certificates

The `pinn_aposteriori_error` path is a **model-problem certificate** for trained
or explicitly supplied neural ansatz layers. It rigorously encloses interior and
boundary residuals with the certified multivariate jet, combines them with an
explicit stability estimate, seals the result with the v1 certificate format, and
routes it through the proof machine's schema / replay / honesty gates.

The stability constants remain a recorded proof obligation unless a future helper
supplies and proves them for a named PDE/domain pair. Optional Lean checking only
discharges finite numerical inequalities such as `error_bound <= threshold`; it
does not formalize the analytic PDE theorem.

The certified jet's **function class** is the rigorous (interval) derivative tower:
`tanh`, `sigmoid`, `gaussian`, and the trigonometric pair `sin` / `cos`. The trig
pair lets a closed-form **Fourier-mode / plane-wave** ansatz be residual-*certified*
— e.g. a single `cos` layer `u = cos(w·x)` is enclosed as an exact Helmholtz
eigenfunction (`Δu + |w|²u = 0`), with the enclosure tightening toward zero under
box subdivision (`splits`). This is a *rigorous interval enclosure over the whole
cell*, not a new analytic solution and not a continuum/global-regularity statement; it is a
strict subset of the differentiable tower's activation set (§2), and certifying a
field never asserts that the field is a true PDE solution beyond the sealed
residual/stability bound. See the
[proof-carrying PDE cookbook](cookbook/proof-carrying-pde.md).

### 3.3 Periodic Navier–Stokes residual certificates

The `navier_stokes_periodic_residual` path
(`omnibias.pinn.certified.fluid.certified_periodic_flow_residual`) is the
**nonlinear** companion to 3.2. Because Navier–Stokes is nonlinear it does *not*
reuse the linear a-posteriori theorem; it instead seals the **evidence** that a
sampled periodic field satisfies the incompressible momentum, continuity and
pressure-Poisson equations to a stated tolerance, plus energy / enstrophy /
palinstrophy diagnostics and a regenerable fixture descriptor.

The boundary is deliberately narrow and encoded in the honesty flags:

- Residuals are computed by **periodic spectral (FFT) sampling on a grid**, not
  by interval enclosure, so `interval_verified = False`. This is *certified-evidence
  evidence*, not a continuum theorem. The certificate now also records a
  **band-limited spectral L1 bound** (`spectral_l1_residual_bound`): for the
  truncated trigonometric polynomial actually sampled, `sum |coeff|` dominates the
  sup *between* grid nodes, so the FFT sup is no longer an unguarded node-only
  number. The recorded resolution diagnostic `velocity_spectral_tail_ratio`
  exposes under-resolution. A genuine interval / Taylor-model enclosure of the
  residual is the path in §3.4.
- These FFT residual numbers are **not bit-reproducible across platforms** (FFT
  ordering / BLAS), unlike the closed-form derivative tower in `omnibias.core`,
  which is bit-identical by construction. "Bit-stable" in this stack refers to the
  derivative coefficients, not to a float sup printed by a backend FFT.
- The shipped fixtures — the 2-D **Taylor–Green** vortex, the 2-D **Kolmogorov**
  forced base state and the exact **3-D Arnold–Beltrami–Childress (ABC)** decaying
  flow (`beltrami_abc_flow`) — are **exact analytic** Navier–Stokes solutions, so
  `exact_solution_claim` is a finite-grid, finite-time statement about a known
  flow. The 3-D ABC fixture is the honest answer to "does this do 3-D
  Navier–Stokes?": yes, for an exact analytic 3-D solution; no, not global regularity
  global-regularity problem (`unproven_claim = False`).
- `unproven_claim`, `continuum_navier_stokes_claim`, `chaotic_tracking_claim`,
  `perfect_weather_claim` and `turbulence_closure_claim` are all `False`. The
  schema validator rejects any certificate that flips them on; it also recomputes
  the body digest (`periodic_residual_digest_ok`) so an edited payload fails the
  schema gate, and the numpy-only twin
  `omnibias.symbolic.fluid.verify_periodic_flow_residual` regenerates the flow
  from the descriptor with **two independent methods** — a spectral re-derivation
  *and* a `np.roll` central finite-difference cross-check
  (`finite_difference_consistent`) — so a tampered bundle fails the replay gate.

This is the honest version of "chaotic / turbulent fluid dynamics": certified
residual and divergence bounds for periodic model flows, not perfect weather and
not high-Reynolds turbulence tracking. See the
[proof-carrying fluid dynamics cookbook](cookbook/proof-carrying-fluid-dynamics.md).

### 3.4 Rigorous interval streamfunction residuals

`navier_stokes_streamfunction_residual`
(`omnibias.pinn.certified.fluid_rigorous.certified_streamfunction_residual`)
**discharges** the open obligation left by §3.3: it is a genuine **interval
enclosure** (`interval_verified = True`), valid on the *whole* periodic cell, not
just at sample nodes. Incompressibility is enforced *by construction* via a
**streamfunction cage** `u = ∇⊥ψ`, so `div u = ψ_xy − ψ_yx` encloses `0`
structurally; the certified quantity is the steady 2-D **vorticity-transport**
residual `(u·∇)ω − νΔω − f_ω`, built from the verified `tanh` Cauchy-product jet
in `omnibias.core.verified`.

- A `y`-only `tanh` streamfunction is a **rigorously exact** steady-Euler shear:
  the interval residual encloses machine zero over the entire domain.
- A general `tanh` streamfunction yields a rigorous, *finite* residual enclosure
  that **tightens under subdivision** — the standard interval-analysis signature.
- The independent twin `verify_streamfunction_residual` recomputes the residual
  by an **explicit analytic `tanh`-tower chain rule** (a genuinely different
  algorithm from the interval jet), so a forged too-small residual fails replay.
- This certifies a residual *enclosure* for a constructed neural field; it is
  **not** a continuum existence/uniqueness or global-regularity statement (`unproven_claim =
  False`).

### 3.5 Genuine pseudo-spectral rollout diagnostics

`navier_stokes_rollout_diagnostics`
(`omnibias.pinn.certified.fluid_rollout.certified_rollout_diagnostics`) replaces
the earlier "rollout" over *analytic snapshots* with a **real time integration**:
a pseudo-spectral 2-D vorticity–streamfunction solver (integrating-factor RK2,
2/3-rule dealiasing) advanced forward from a Fourier initial condition. What is
sealed are honest **window diagnostics**, not pointwise truth:

- **incompressibility is maintained** — `max_divergence` stays at machine zero
  (spectral velocity is divergence-free by construction);
- for the **inviscid, unforced** case the conserved invariants (energy,
  enstrophy) drift by a *reported, finite* amount — a measurable statement about
  how little numerical diffusion the scheme adds — and for the **viscous** case
  energy is reported as monotonically dissipating;
- `chaotic_tracking_claim` and `perfect_weather_claim` remain `False`; the twin
  `verify_rollout_diagnostics` re-integrates independently in numpy and matches
  the recorded drift / divergence.

This is the honest meaning of "explicitly not perfect weather": we certify
*statistical / conservation* window diagnostics of a real rollout, never
butterfly-exact pointwise long-horizon tracking.

### 3.1 The prove/disprove machine: verdict semantics

`omnibias.core.proof.ProofMachine` (assembled with every built-in prover by
`omnibias.pinn.certified.build_default_machine`) gives all of these certificates
one front door: a `Conjecture` goes in and a `Verdict` comes out. The status is
deliberately three-valued, and the words mean exactly this:

| Status | Meaning | What backs it |
|---|---|---|
| **PROVED** | The asserted model statement is certified **and** survives every gate. | A certificate whose schema validates, whose independent replay (where a twin exists) agrees, and whose honesty flags support any asserted claim. |
| **DISPROVED** | The negation is certified (the statement is rigorously false for this datum). | E.g. the CLM multi-zero prover enumerates *all* zeros (exact Sturm) and certifies `max H ω₀ < 0`, so no finite-time singularity can exist. |
| **BLOCKED** | Neither proved nor disproved with the current machinery. | Schema invalid, replay disagreed, an honesty claim was unsupported, no prover handles the kind, or the certificate simply did not close. The unmet obligations are listed. |

Three gates run on top of every prover's intrinsic result, and **any** of them
downgrades a `PROVED`/`DISPROVED` to `BLOCKED`:

1. **Schema gate** — the certificate must pass its own `*_schema_errors`
   validator.
2. **Replay gate** — when an independent numpy twin exists (e.g.
   `verify_clm_multizero_first_blowup`), it must agree; disagreement blocks.
3. **Honesty gate** — a `Conjecture` may *assert* honesty claims (e.g.
   `claims={"unproven_claim": True}`). The verdict only stands if the certificate's
   own `honesty` flags back every asserted claim. Because every model
   certificate hard-wires `unproven_claim=False`, **asserting a global-regularity claim is always
   downgraded to `BLOCKED`** unless an external verifier has attached genuine
   evidence — the same boundary as the per-certificate schema gates above.
4. **Formal (Lean) gate** — evaluating with `lean_check=True`, or asserting the
   reserved `theorem_prover_verified` claim, routes a certificate's *finite,
   rational* obligation through the Mathlib-free Lean kernel
   (`formal/omnibias-verified-kernel`) via `omnibias.core.proof.lean_check`.
   `Verdict.theorem_prover_verified` is set **only** on a genuine `lake` pass and
   can never be forged by a certificate flag; asserting the claim without a pass
   downgrades to `BLOCKED`. With no Lean toolchain the gate degrades gracefully
   (the flag stays `False`; an unclaimed verdict is unaffected). The kernel only
   discharges finite obligations — infinite analytic obligations (limits,
   continuum statements, asymptotics) are out of scope and are not expressed in
   the Lean projects at all.

So the machine never manufactures a global-regularity claim. `PROVED` always means "this
**model** statement is certified", never "this global-regularity problem is solved". See the
[prove/disprove machine cookbook](cookbook/proof-machine.md) for worked verdicts.

## 4. Per-package contract levels

| Tier | Packages | Contract |
|---|---|---|
| **Stable** | `omnibias-core`, `omnibias-torch`, `omnibias-jax`, `omnibias-ferminet` | Public surface frozen; two-step deprecation policy. |
| **Beta** | `omnibias-pinn`, `omnibias-fields`, `omnibias-geometry` | Public surface frozen; heavy off-band GPU / large-scale production runs may live outside the public tree, but **CPU smoke and multi-seed acceptance JSON for the PINN four-gap suite are public** under [`docs/benchmarks/`](benchmarks.md) (see §6). |
| **Alpha** | `omnibias-qpinn`, `omnibias-curvature`, `omnibias-symbolic`, `omnibias-score`, `omnibias-fractional`, `omnibias-binary`, `omnibias-boolean`, `omnibias-spiking`, `omnibias-hopfield`, `omnibias-keras`, `omnibias-convex`, `omnibias-control`, `omnibias-routing`, `omnibias-verify`, `omnibias-dynamics`, `omnibias-graph`, `omnibias-struct` | API may shift between alpha releases; per-package CI gate. |

Each tier ships with its own breaking-change policy
([`docs/stability.md`](stability.md)) and per-package CI job. The roadmap
([`docs/roadmap.md`](roadmap.md)) records the three promotion criteria (math
settled, three independent test cases, cross-backend parity) and the evidence
attached to each promotion.

## 5. Cross-backend bit-identity

Every public closed-form kernel that exists on more than one backend is
**float64-ULP-equal** across backends — torch, jax, and keras 3 all import the
same `omnibias.core.polynomials` recurrence. This is verified per release by
`tests/test_jax_parity.py`, `tests/test_keras_parity.py`, and the
package-level cross-backend suites. Bit-identity is what makes
backend-portable certified-numerics work in the first place.

On float32 Keras backends, agreement drops to float32 tolerance — the kernel
is identical, the precision is the user's choice.

## 6. Domain & numerics scopes that bite

A few specific scope notes that an AI agent should know before reaching for
the relevant op:

- **Empirical PINN four-gap gates** (`omnibias.pinn.train` / `.domain` /
  `.operator`, plus FBPINN / one-shot `lstsq`) are **public, regenerable CPU
  artifacts**, not an internal-archive-only claim. Scripts in `benchmarks/`
  write smoke (`*_smoke.json`, CI) by default and `--full` multi-seed acceptance
  JSON under `docs/benchmarks/`; every artifact emits a `gates` block via
  `benchmarks/_gates.py`. Status matrix:
  [`benchmarks/pinn_four_gap_matrix.md`](benchmarks/pinn_four_gap_matrix.md).
  Smoke is a wiring gate — do not quote it as a multi-seed result.
- **`omnibias-fractional`** is non-local and **grid-based**. Accuracy is
  controlled by `h`; see the per-op error budget in
  [`FRACTIONAL_DERIVATIONS.md`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-fractional/FRACTIONAL_DERIVATIONS.md).
- **CCF residuals** are on a **periodic torus** with a spectral Hilbert
  transform; the periodic toy model is *not expected* to reproduce the
  line-domain self-similar exponents (that is a documented domain mismatch,
  not a bug).
- **`omnibias.geometry.gauge` lattice MC** is currently SU(2) at **fixed spacing** on a
  finite-volume periodic lattice (heat-bath / over-relaxation / Langevin
  / Wilson loops / Creutz ratios / GEVP glueball). Stochastic numerical
  evidence, not a continuum claim.
- **`omnibias.geometry.gauge.transfer` certified mass gap** is a **sound lower
  bound** on `m a = -ln(|λ₁| / λ₀)` for **one fixed finite transfer matrix at one
  fixed spacing**, by interval arithmetic over the rigorous engines in
  `omnibias.core.verified.eig`. It is *proof* about that matrix, and the sealed
  certificate's rational obligation is Lean-checkable via `spectral_gap_pos`.
  It is **not** a statement about the continuum limit (`continuum_claim` is
  hard-wired `False`), **not** a statement about a lattice ensemble, and **not**
  the Yang-Mills mass gap. `heat_kernel_gap_scaling_report` collects such bounds
  across spacings as **evidence about a trend**, never an extrapolation.
  `certified_gap_versus_monte_carlo` cross-checks the bound against a Monte Carlo
  of the *same* matrix's path measure — an independent oracle, still evidence
  rather than proof; only the interval arithmetic is proof.
  Optional holonomy `trial=` spaces are characters of closed loops on a dense
  angle / class-angle grid; they may tighten a variational bound for that
  matrix. They do not take a continuum limit.
- **`certified_strong_coupling_glueball_bound`** is a **polymer-count**
  lower bound for SU(2) Wilson at **one fixed `β`** and spacing. The
  default is the two-scale remainder `u + A u² / (1 - B u)` with
  first-step `A = 20` and subsequent `B = 15` in 4-D. `certified=True`
  only inside the interval-certified domain (enclosed ratio `< 1`). It
  is **not** a continuum claim, **not** a uniform-in-spacing bound, and
  **not** the Yang-Mills mass gap. The default method tag is
  `two_scale_polymer_count`. `counting="cluster"` keeps the first terms
  explicitly. Single-scale `C = 15` is **not** a bound on `N_2`.
- **`su3_wilson_transfer`** encloses SU(3) Wilson character coefficients
  by midpoint-plus-Lipschitz Haar quadrature on the maximal torus at
  **one fixed `β`**. The matrix is the `(p, q) ≤ 3` truncation: locked
  trigonometric characters through `(2,2)`, Clebsch recurrence for
  `p, q = 3`. It is **not** a product of ordinary Bessel functions,
  **not** 4-D SU(3) Yang-Mills, and **not** a continuum claim.
- **`certified_hamiltonian_gap`** is a sound lower bound on `λ1 - λ0` of
  **one finite two- or three-plaquette SU(2) Kogut–Susskind Hamiltonian**
  at one coupling and one spin truncation. Default magnetics are Racah
  6j weights. It is **not** a continuum claim, **not** a
  uniform-in-spacing bound, and **not** the Yang-Mills mass gap.
- **`su2_spatial_strip_transfer`** is one finite Euclidean-time transfer
  on a spatial circle of SU(2) class angles. **`su2_spatial_torus_transfer`**
  is the finite 3+1-D analogue on a 2×2 spatial torus. Reflection
  positivity is checked on **that** matrix; that is not
  Osterwalder–Seiler reconstruction and not a continuum claim.
- **`certified_gap_scaling_table`** is a table of independent
  fixed-spacing certificates. `continuum_claim` is hard-wired `False`.
  A trend across rows is not an extrapolation.
- **`omnibias-qpinn` Bloch cage** supports orders ≤ 2 on a **single axis** in
  v0.0.1; multi-axis mixed partials need the Leibniz expansion (planned).
- **Random-feature fields** are accurate only inside the support of their
  training points; `field_*` operators on extrapolation points return the
  exact derivative of the *fitted* field, which can diverge from the true
  function.
- **Differentiable info-theory kernels** assume `sum(p) == 1`. The certified
  enclosures renormalise and widen the bracket instead.
- **`omnibias-geometry` / `omnibias.geometry.gauge` topology** realise only the
  **de Rham / Chern-Weil** (differential-form) slice of algebraic topology:
  the Hodge Laplacian, harmonic-form Betti numbers, map degree / winding, the
  Gauss-Bonnet Euler characteristic, and the Chern / Pontryagin / `SO(2)` Euler
  *characteristic numbers* of a connection. These are smooth-manifold integrals
  (numerical quadrature + a nullity count, certifiable by an interval
  enclosure). **Out of thesis** (never claimed, guarded by enforcement tests):
  homotopy groups `π_n`, persistent homology / TDA, simplicial `ℤ`-homology,
  Smith normal form, and higher-rank Euler classes via the Pfaffian. The
  `k ≥ 1` Hodge Laplacian is exact only on a **constant (flat) metric**; a
  curved-metric `k`-form Laplacian (Weitzenböck term) raises `NotImplementedError`.
- **`omnibias.geometry.gauge` representation theory** realises the **`su(N)`
  highest-weight / Lie-algebra** slice only: exact dimension, quadratic Casimir,
  and Dynkin index (rational, certified by construction), the Freudenthal weight
  system, the Racah–Speiser tensor-product decomposition, Gelfand–Tsetlin
  `su(N) -> su(N-1)` branching, the Weyl (Schur) character, and explicit spin-`j`
  / adjoint generator matrices. **Out of thesis** (never claimed, guarded by an
  enforcement test): finite (non-Lie) group character tables, conjugacy-class /
  Burnside / Molien machinery, and general abstract-group theory. Exceptional and
  non-compact real forms beyond the `su(N)` / `u(1)` generators are future work.
- **`omnibias-qpinn` / `omnibias-ferminet` molecular QM** ships the **closed-form**
  pieces of fixed-nuclei (Born-Oppenheimer) electronic structure: the drift-form
  local kinetic energy `T_L = -1/2 (∇²log|ψ| + ‖∇log|ψ|‖²)` (the exact multivariate
  jet Laplacian of `log|ψ|`), the bare Coulomb potential (e-n / e-e / n-n), the
  symmetric Padé-Jastrow correlation factor's closed-form value / gradient /
  Laplacian, and the hard `NuclearCuspField` cage enforcing the Kato electron-nucleus
  cusp `u'(0) = -Z`. The hydrogen-atom oracle `E_L = -Z²/2` and the harmonic-trap
  oracle `E_L = D·ω/2` are reproduced exactly (zero local-energy variance). **Not
  closed-form / out of scope** (iterative or stochastic numerics, never claimed,
  guarded by an enforcement test): VMC Monte-Carlo sampling, SCF / Hartree-Fock /
  CI / coupled-cluster self-consistency, and Gaussian-basis electron-repulsion
  integrals (ERI). The direct Galerkin eigensolver is a **numerical** Rayleigh-Ritz
  quotient (quadrature + `scipy.linalg.eigh`); it is bit-exact only in its
  analytic-basis limit -- a single unit-width Gaussian `exp(-x²/2)` is the exact
  1-D SHO ground state, so a `K = 1` solve returns `E₀ = 1/2` with no fitting error.
- **`omnibias-graph` differentiable graph ops** ship two families: **exact** spectral
  linear algebra (combinatorial / normalized / random-walk Laplacians, Laplacian-eigenmaps
  `spectral_embedding`, the heat kernel `exp(-tL)`, and the Rayleigh-Ritz cut relaxation)
  and **differentiable relaxations** of discrete objects (`sinkhorn_normalize`,
  `gumbel_sinkhorn`, `soft_sort`, `soft_top_k`), each with a temperature that recovers the
  hard object only as `tau -> 0`. The ring-graph spectrum `2 - 2cos(2 pi k / n)` is certified
  by an `omnibias.core.verified` interval eigenvalue enclosure that brackets a true
  eigenvalue. **Yes-if, not out of scope:** a relaxed cut / tour value is a rigorous *lower
  bound* on the discrete optimum, and combinatorial **routing** *is* supported end to end
  as a **certified differentiable relaxation + decoder + reported optimality gap** in
  [`omnibias-routing`](api/routing.md) (`lower <= optimum <= tour_cost`, never a zero-gap
  claim). The **one honest limit** (a theorem, kept): no poly-time *differentiable* map
  returns the *exact* NP-hard optimum -- that would imply `P = NP`, and the exact argmin's
  gradient is a.e. zero, so an "exact differentiable TSP" is ill-posed for learning.
  `omnibias-graph` itself ships **only** relaxations and exact spectral algebra -- never an
  exact NP-hard solver (guarded by an enforcement test); see the
  [certified combinatorial-optimization cookbook](cookbook/graph-limitation.md).
- **`omnibias.core.verified.dirichlet` analytic number theory** ships a **verified-interval**
  slice of Dirichlet series in the half-plane of absolute convergence `Re(s) > 1`:
  `zeta_enclosure`, `l_function_enclosure` (Dirichlet `L` / beta), the general
  `certified_dirichlet_series` (caller-proved tail majorant, same contract as
  `certified_geometric_series_sum`), the `n^{-s}` term, the `p`-series
  `p_series_tail_bound`, and the public Jacobi `theta_enclosure`. Every enclosure
  *provably contains* the `mpmath` value (`zeta(2) = π²/6`, `zeta(4) = π⁴/90`,
  `beta(2) = Catalan`). **Mandatory majorant / boundary:** the tail bound is the
  integral-test `p`-series majorant, valid **only** for `Re(s) > 1`; the constructor
  raises for `Re(s) ≤ 1`. Analytic continuation past `Re(s) = 1` -- the functional
  equation, the critical strip, and the **Riemann Hypothesis** -- is a recorded
  *external* proof obligation, never inferred, and nothing here makes any statement
  about zeros of `zeta` / `L`.
- **Number theory & cryptography -- centralized out-of-scope boundary.** omnibias is an
  *analytic / differential* engine, not a discrete-hardness solver. The following are
  **out of scope and never claimed** (guarded by enforcement tests and the
  [RSA-limitation](cookbook/rsa-limitation.md) cookbook): integer **factoring**, the
  **discrete-logarithm** problem, **RSA** / **ECC** / Diffie-Hellman or any key-recovery
  attack, **primality proving** / general sieving, and any claim about the **Riemann
  Hypothesis** or the location of non-trivial zeros. These have no analytic / relaxation
  handle -- unlike combinatorial **routing**, which *is* supported with a certified gap
  (see [`omnibias-routing`](api/routing.md) and the
  [certified combinatorial-optimization cookbook](cookbook/graph-limitation.md)).
  Relatedly, `omnibias.boolean.cipher` is **S-box figure-of-merit analysis**
  (nonlinearity, SAC, differential-uniformity, Walsh-spectrum descriptors), **not**
  cryptanalysis: it scores design metrics, it does not break ciphers.

## 7. Where to look next

- The handbook [AI quickstart](handbook/ai-quickstart.md) carries the gotchas
  table in a copy-paste-friendly form.
- The complexity page ([`docs/complexity.md`](complexity.md)) has the measured
  asymptotic wins (`O(1)` in `D`, polylaplacian flat in `k`).
- The API reference per package
  ([`docs/api/`](https://github.com/derivon-ai/omnibias/tree/main/docs/api))
  links into the autodoc with these labels attached.
- [References](references.md) collects the external literature for the models
  the certified stacks validate against.

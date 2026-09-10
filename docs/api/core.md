# omnibias-core

The pure-Python mathematical core: polynomial coefficient generators
and the backend-agnostic ActivationSpec. Named collapses besides the
founding three senses live in [collapse.md](collapse.md). The
certificate router over those collapses is
[proof_engine.md](proof_engine.md).

## Polynomials

::: omnibias.core.polynomials
    options:
      show_root_heading: false
      heading_level: 3

## Bell polynomials / Faà di Bruno

::: omnibias.core.bell
    options:
      show_root_heading: false
      heading_level: 3

## Multi-index combinatorics

Canonical multi-index ordering and the truncated Cauchy-product table powering
the *multivariate* (multi-index) jet kernels.

::: omnibias.core.multi_index
    options:
      show_root_heading: false
      heading_level: 3

## Multi-pack Birkhoff support

Pure-Python support algebra for heterogeneous multi-pack collapse (theory
01-01). Documented in full at [multipack.md](multipack.md).

## Bias-scan bank

Offset / scale bank algebra for the transverse bias scan (theory 01-02).
Documented in full at [scan.md](scan.md).

## Causal transverse taps

Designed causal `sigma^(n)` FIR taps (theory 05-02 G5). Order 0 is the
logistic tail. Documented in full at [sequence.md](sequence.md).

## Mollifier calculus

Pack-as-mollifier algebra with certified exponential tails (theory 01-05).
Analytic bases are not compactly supported; higher-order kernels take
negative values. Documented in full at [mollifier.md](mollifier.md).

## Spectral design

Order-as-frequency band plans (theory 01-07). Pack order is a band
selector, not a Littlewood-Paley completeness claim. Frames:
[frames.md](frames.md). Documented in full at
[spectral_design.md](spectral_design.md).

## OMBU frames

Wavelet-frame reading of a bias-scan bank (theory 01-06). `sigma'` is
not admissible; frames are not orthonormal and not compactly supported.
Documented in full at [frames.md](frames.md).

## Equality locus

Constraint-manifold Newton / Krawczyk (theory 01-09). Not a general
PDE solver. Documented in full at [locus.md](locus.md).

## Jet-bundle vocabulary

Contact residual / holonomy test (theory 01-10). A dictionary, not a
discovery. Documented in full at [jets.md](jets.md) and
[theory-jets.md](../theory-jets.md).

## Conjugate Hilbert tower

Line Hilbert permutation of the dictionary (theory 01-12). G5 is a
projection defect, not a stretch-gate clearing. Documented in full at
[conjugate.md](conjugate.md).

## Recommended trainer stack

08-01 stack on a one-layer closed-form loss jet: Newton, jet line
search, optional Kantorovich, sharpness damping. Documented in full at
[train_stack.md](train_stack.md).

## Weight-space loss jet

Closed-form directional derivatives of a one-layer Riccati MSE loss
along a weight direction. Leibniz assembly from `σ^(n)`; not a deep-net
polynomial in every weight. Documented in full at
[weight_loss_jet.md](weight_loss_jet.md).

## Exact jet line search

Taylor-polynomial line search (theory 03-12). Certified Lagrange
radius when a `|phi^(N+1)|` bound is supplied; `verify=True` is the
never-worse backstop. G4/G5 are recorded, not in CI `all_passed`.
Documented in full at [line_search.md](line_search.md).

## Adaptive pack refinement

Birth / growth / death of tempered packs (theory 03-13). Birth and
growth are bit-identical; death reports a bound. G4 is recorded, not
in CI `all_passed`. Documented in full at [refine.md](refine.md).

## Composed-curvature joint Newton

Order-2 chain rule on consecutive layers (theory 08-02). Escape is from
a slice critical point when the joint block is indefinite. G1–G4 CI.
Not a global min and not CCF stretch. Documented in full at
[composed_curvature.md](composed_curvature.md).

## Sharpness-scheduled step

Map a Ritz `lambda_max` from exact HVPs to cubic `sigma` or a
learning rate (theory 08-06). Sharpness is a step-size signal, not a
generalization claim. Documented in full at
[sharpness_schedule.md](sharpness_schedule.md).

## Block exact search

03-12 line search on one last-linear / OMBU-bias / arrangement block
(theory 08-07). A coordinate sweep, not a global solver. Documented in
full at [block_exact_search.md](block_exact_search.md).

## Depth-causal local jet

Named local Gauss–Newton plus a compressed `k`-direction `layer_jet`
(theory 08-03). Greedy warm start, not a global min. Documented in
full at [local_jet.md](local_jet.md).

## Implicit / DEQ Newton

Fixed point `u = sigma(W u + x)` plus an exact-`sigma'` IFT VJP
(theory 08-08). One linear solve, not unrolled BPTT. Documented in
full at [implicit.md](implicit.md).

## Hierarchical pack tree

1-D near/far split (theory 02-07). `eta=0` is bit-identical to the
dense sum. Documented in full at [hierarchy.md](hierarchy.md).

## Tanh-method / ladder / transfer / linearizing maps

Gated core algebra for remaining Group 02: [travelling.md](travelling.md),
[ladder.md](ladder.md), [layered.md](layered.md),
[transforms_pde.md](transforms_pde.md).

## ActivationSpec

::: omnibias.core.spec
    options:
      show_root_heading: false
      heading_level: 3

## Integral-transform identities

The single source of truth for *which* Laplace / Fourier / Mellin transforms of
the activation dictionary omnibias ships in closed form, what each one equals,
where it converges, and -- just as importantly -- why each gap is a gap. Pure
Python: strings, floats and frozen dataclasses, no tensor library, exactly like
`polynomials` is the shared source of the derivative-tower coefficients. The
backend twins `omnibias.torch.transforms` and `omnibias.jax.transforms` are thin
tensor evaluations of this table, and a coverage test walks the table against
both registries so code and documentation cannot drift.

Conventions are fixed once here: `L[s] = int_0^inf sigma(z) e^{-sz} dz`,
`F[xi] = int_R sigma(z) e^{-i xi z} dz` (non-unitary angular frequency), and
`M[s] = int_0^inf sigma(z) z^{s-1} dz`.

Gaps carry a reason code rather than silence -- **divergent** (the integral does
not converge for the activation *as omnibias registers it*), **distributional**
(the transform exists only as Dirac masses or principal values, which no tensor
kernel can return), **complementary** (the convergent classical identity belongs
to the activation's complement), **conditional** (convergent only as an Abel
limit, so no quadrature can validate it), or **unavailable** (a closed form
exists but calls a special function neither backend ships).

::: omnibias.core.transforms
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: certified series & spectral gap

Rigorous (outward-rounded `Interval`) building blocks used by the certified-evidence
stack. `series` is the verified `Σ`: a truncated sum plus a geometric tail
majorant becomes a theorem-grade bracket. `eig` is the verified spectral-gap
operator: a Rayleigh / residual eigenvalue enclosure and the **Birkhoff–Hopf**
certified subdominant-ratio bound `τ = (√κ − 1)/(√κ + 1)` for an
entrywise-positive matrix, giving a rigorous mass-gap lower bound
`m a ≥ −ln(τ)` at fixed lattice spacing.

For a real **symmetric** matrix `certified_symmetric_spectral_gap` gives a
much tighter certificate via power sums: a Rayleigh / Collatz–Wielandt lower
bound on the Perron root `λ₀` together with the Schur inequality
`|λ₁| ≤ √(tr(A²) − λ₀²)` on the subdominant *modulus* (since
`Σ λᵢ² = tr(A²) = Σ aᵢⱼ²`). Whenever the subdominant eigenvalue dominates the
remaining tail — as for rapidly decaying heat-kernel spectra — this recovers
nearly the full gap rather than the fraction Birkhoff–Hopf yields. When the
remaining tail is *not* negligible — a degenerate `λ₁` (inflating the plain bound
by `√(multiplicity)`) or a slowly-decaying spectrum — passing the top eigenvectors
as `subdominant_vectors` deflates a **chain** of rigorous Courant–Fischer lower
bounds: each nested frame `[perron, v₁, …, v_k]` gives `ℓ_k ≤ λ_k`, and since the
`λ_k` are distinct power-sum terms `Σ_k ℓ_k² ≤ Σ_{i≥2} λ_i²`, so
`λ₁² ≤ tr(A²) − λ₀² − Σ_k ℓ_k²` collapses onto the exact `λ₁` as more partners are
supplied (rigorous for any input vectors; a rank-deficient frame deflates nothing).
The same hints yield a rigorous **upper** bound on the gap (`λ₀` above by
`min(√(tr A²), Gershgorin)`, `λ₁` below by the smallest Ritz value of
`[perron, v₂]`), so the certificate **brackets** `m a ∈ [lower, upper]`; the
bracket collapsing to a point certifies the gap essentially exactly.

`certified_block_operator_gap` lifts a *finite* computation to an
infinite-dimensional **coercivity** statement. Split a self-adjoint operator by an
orthogonal projection into a finite block (smallest eigenvalue `a`, computed here
by Gershgorin), a coupling `b ≥ ‖B‖`, and a **tail block whose gap `d` is an
explicit hypothesis**; the Schur / `2×2` bound
`λ_min(S) ≥ ½[(a+d) − √((a−d)² + 4b²)]` is then a rigorous lower bound *given* `d`.
Coercivity holds iff `d > threshold_tail_gap = b²/a` — the single scalar inequality
that a conditional spectral-gap program (e.g. the linearised rescaled SQG operator
in a weighted norm) must still close. It never claims the tail bound
(`tail_is_hypothesis` is always `True`) and makes no continuum / blow-up claim.

::: omnibias.core.verified.series
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.eig
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: certified invariant subspaces (Davis–Kahan)

`eig` and `eig_operator` certify *scalar* spectral facts: one eigenvalue lower
bound, an exact eigenvalue count (with multiplicity) below a threshold, or a
gap between two named eigenvalues. Neither certifies that a whole **cluster**
of eigenvalues — a genuinely degenerate ground state, or a near-degenerate
representation multiplet — has been resolved as a *subspace*, without
requiring the eigenvalues inside the cluster to be individually separated
from each other. `invariant_subspace` closes that gap with a single
certificate, `certified_invariant_subspace`, implementing the classical
**Davis–Kahan `sin Θ` theorem** via its residual/Sylvester-equation proof, on
top of the *existing* eigenvalue machinery (no new eigenvalue solver): the
candidate `V`'s Ritz block `H = VᵀAV` and Gram `G = VᵀV` give a certified
cluster bracket `[ritz_lower, ritz_upper]` via
`generalized_eigenvalue_enclosure` on the pencil `(H, G)`; inertia bisection
(`count_eigenvalues_below`) on `A` itself assigns the cluster to eigenvalue
indices `p+1 .. p+k` and certifies its separation `gap` from *every other*
eigenvalue of `A`; and a rigorously re-orthonormalized `Ṽ` (via the interval
`LDLᵀ` factor of `G`) gives a certified **Frobenius**-norm residual
`‖R‖_F = ‖AṼ - ṼH̃‖_F`. The bound `sin Θ(span(V), S) ≤ min(‖R‖_F / gap, √k)`
then follows from the Sylvester equation `A₂Y - YH̃ = P_{S⊥}R` the residual
identity induces on the orthogonal complement `S⊥` of the true invariant
subspace `S`. Works for an **exactly degenerate** cluster and any `k ≥ 1`
without requiring the `k` Ritz values to be separated from each other — only
the cluster as a whole from the rest of the spectrum. When `G` cannot be
certified positive definite (the columns of `V` are not certified
independent) or the separation gap cannot be certified positive, the
certificate honestly reports `certified=False` with the unresolved fields
`None`, matching `count_eigenvalues_below`'s returns-`None`-on-failure idiom,
rather than fabricating a bound.

```python
from omnibias.core.verified.invariant_subspace import certified_invariant_subspace

# A = diag(2, 2, 5): eigenvalue 2 has an exact 2-D eigenspace, well separated from 5.
a = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]]
v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # the exact 2-D eigenspace, as two basis vectors

cert = certified_invariant_subspace(a, v)
assert cert.certified
assert cert.cluster_start_index == 1
assert cert.gap_lower is not None and abs(cert.gap_lower - 3.0) < 1e-9
assert cert.sin_theta_upper is not None and cert.sin_theta_upper < 1e-9  # exact eigenspace
```

::: omnibias.core.verified.invariant_subspace
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: certified conditioning (the `ε→0` collapse)

`conditioning` is the rigorous register of the **`ε→0` rank/regularization
collapse**: Tikhonov solving `(A + εI)⁻¹b` collapses onto the Moore–Penrose /
minimum-norm solution `A⁺b` as `ε→0`, and this module certifies the quantities
that govern that limit — `certified_min_eigenvalue` / `certified_max_eigenvalue`
(inertia-bisection enclosures of `λ_min` / `λ_max`), `certified_condition_number`
(`κ = λ_max/λ_min`, upper endpoint `+∞` when positive-definiteness cannot be
certified — the honest rank-deficient signal), `certified_damping` (the smallest
`ε` provably giving `κ(A + εI) ≤ T`), `certified_regularization_error` (a sound
`‖x_ε − A⁺b‖` bound on `range(A)`), and `conditioning_certificate` (a sealed v1
certificate carrying the `λ_min > 0` `LDLᵀ` pivots for the Lean bridge). The
differentiable min-norm / Tikhonov solver that consumes these lives in
[`omnibias.curvature.regularize`](curvature.md).

!!! note "Honesty labels"
    The enclosures are **verified** (outward-rounded on top of `eig_operator`); the
    regularized solve itself is a **numerical** (LAPACK-class) operation done by the
    consumer, never "closed-form". The `ε→0` limit is a *distinct* collapse from the
    founding `δ→0` derivative limit and the `β→∞` feasibility penalty — never
    conflated. `certified_regularization_error` is sound only on `range(A)`; a
    null-space right-hand-side component makes the naive Tikhonov solve diverge as
    `ε→0` (the blow-up the collapse avoids) and is out of scope.

::: omnibias.core.verified.conditioning
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: analytic number theory (Dirichlet series)

`dirichlet` applies the verified `Σ` to Dirichlet series `D(s) = Σ aₙ n⁻ˢ` in the
half-plane of absolute convergence `Re(s) > 1`: `zeta_enclosure`,
`l_function_enclosure` (Dirichlet `L` / beta), the general
`certified_dirichlet_series` (caller-proved tail majorant), the `n⁻ˢ` term, the
integral-test `p_series_tail_bound`, and the public Jacobi `theta_enclosure`. Every
enclosure provably contains the true value (cross-checked against `mpmath`:
`ζ(2) = π²/6`, `ζ(4) = π⁴/90`, `β(2) = Catalan`).

**Exact special values** (`closed-form`, only inexactness is the certified `π`
enclosure `PI_IV`): `zeta_even(m) = ζ(2m)` as a rational multiple of `π²ᵐ` off the
`tanh`-tower Bernoulli numbers, `zeta_negative_odd(m) = ζ(1−2m) = −B₂ₘ/2m`, and
`dirichlet_beta_odd(m) = β(2m+1)` as a rational multiple of `π²ᵐ⁺¹` off the
`sech`-tower Euler numbers. **Attempted continuation** (`numerical`):
`zeta_euler_maclaurin(s)` pushes an enclosure of `ζ(s)` into the critical strip via
the Euler–Maclaurin engine with a rigorous remainder — it *encloses* the true value
but proves nothing about the *location* of zeros.

!!! warning "Continuation is numerical only — no RH"
    The absolutely-convergent tail majorant is valid **only** for `Re(s) > 1`;
    `zeta_enclosure` and friends refuse `Re(s) ≤ 1`. `zeta_euler_maclaurin` supplies
    a *numerical* critical-strip enclosure, `zeta_via_functional_equation` encloses
    the continued *value* for `Re(s) < 0` by multiplying `chi(s)` into a right
    half-plane series, and `zeta_approximate_functional_equation` is an AFE on a
    named compact (`|t| ≤ T_MAX`). None of these is a continuation theorem. The
    Riemann Hypothesis remains a recorded *external* proof obligation —
    never inferred. A small enclosed magnitude near a putative zero is **not** a
    claim that `ζ` vanishes there. Nothing here makes a statement about the location
    of zeros of `ζ` / `L`, primality, factoring, or any cryptographic hardness
    assumption (see [scope & guarantees](../scope-and-guarantees.md) §6).

::: omnibias.core.verified.dirichlet
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: complex Gamma, xi, and a finite evaluator

`gamma_complex` encloses `log`, `exp`, `sin` / `cos`, and `Gamma` on complex
rectangles (Stirling in a right sector, multiplicative reflection on the left).
`xi` encloses `chi(s)` and `xi(s)`, dispatches `zeta_continued`, and records a
functional-equation residual `xi(s) - xi(1-s)` as Enclosure Collapse of a finite
obligation (not an identity over `Q`). A rectangle in `Re(s) > 1` has winding
`0`; a strip contour is allowed to return BLOCKED and is never a zero
certificate. `riemann_siegel` adds a Hardy–Littlewood AFE with a cited
Titchmarsh §4.13 remainder majorant on a locked compact and **refuses**
outside it.

::: omnibias.core.verified.gamma_complex
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.xi
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.riemann_siegel
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: Euler–Maclaurin summation

`euler_maclaurin` turns the closed-form derivative tower into a *certified*
summation / continuation engine. `euler_maclaurin_sum(name, a, b, …)` encloses
`Σ_{k=a}^{b} f(k)` for an activation `f` by pairing the exact Bernoulli numbers
(`bernoulli_number_exact`) with derivative enclosures from `sigma_tower_interval`
and a **rigorous remainder** (`B_{2K}` × a box enclosure of `|f^{(2K)}|`), the same
Lagrange-shape bound as the finite-difference remainder. On top of it sit certified
`log_gamma_iv` and `digamma_iv` (Stirling series with an argument shift and enclosed
tail) and slow `ζ`-type partial sums. The engine beats a naive partial sum by orders
of magnitude for the same term budget and is the substrate behind the
critical-strip `zeta_euler_maclaurin` above.

::: omnibias.core.verified.euler_maclaurin
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: certified quadrature

`quadrature` turns the certified derivative tower into a **rigorous numerical
integration** engine: `simpson_integral` (4th-order Peano remainder),
`gauss_legendre_integral` (closed-form nodes/weights + a derivative-bound remainder),
`romberg_integral` (Richardson extrapolation with a certified error term),
`euler_maclaurin_quadrature` (reusing the summation engine above),
`clenshaw_curtis_integral`, and the `tanh_sinh_estimate` double-exponential rule for
endpoint singularities, alongside the sound `trapezoid_integral` / `midpoint_integral`
baselines. Every remainder is derived from a `TaylorModel` derivative enclosure — no fudge
factor — so the returned `Interval` provably contains `∫_a^b f`. At an equal node budget
the Gauss / Romberg rules beat the fixed-node trapezoid baseline; cross-checked against
`mpmath.quad`.

::: omnibias.core.verified.quadrature
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: Hurwitz zeta & polylogarithm

`hurwitz` encloses the Hurwitz zeta `ζ(s, a)` via the Euler–Maclaurin engine, with the
**exact** negative-integer values `ζ(−n, a) = −B_{n+1}(a)/(n+1)` (`closed-form`, off the
exact Bernoulli *polynomials*) and a `numerical` continuation elsewhere. `polylog`
encloses the polylogarithm `Li_s(z)` and the Lerch transcendent `Φ(z, s, a)` on their
domain of convergence with a certified geometric / ratio tail, cross-checked against
`mpmath`. The `dirichlet` module's `dirichlet_L(s, χ)` and its exact `L(1−n, χ)` from
generalized Bernoulli numbers extend the analytic-number-theory surface — the same honesty
applies: exact special values are `closed-form`, continuation is `numerical`, and **GRH
stays an external obligation**, never inferred.

::: omnibias.core.verified.hurwitz
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.polylog
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: Taylor models

A degree-`d` `TaylorModel` encloses a function over a *whole* interval cell as a
polynomial (in the relative variable `x - center`) with `Interval` coefficients
plus an `Interval` remainder that rigorously absorbs everything the polynomial
omits. Keeping the function's *shape* avoids the wrapping blow-up of naive
interval evaluation, so products, powers, `reciprocal` (a certified `1/(1+g)`
geometric series with an analytic tail) and `sqrt` (a certified `√(1+g)` binomial
series with a Lagrange-remainder tail) stay tight. This is the substrate that
discharges the CCF
[between-node residual obligation](../cookbook/ccf-line-calibration.md) (via
`reciprocal`) and the 2-D
[SQG steady vortex](../cookbook/sqg-vortex.md) half-power norm sups (via `sqrt`):
the residual / magnitude is enclosed per cell, so its sup is certified rather than
sampled.

::: omnibias.core.verified.taylor_model
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: 2-D Riesz / Leray

The planar analogue of the line-Hilbert pair, for 2-D model equations
(2-D Euler / SQG). On a basis of **radial blobs that are the Laplacian of an
explicit Newtonian potential** (`f_a = ΔN_a = a²/(π D²)`, `D = x²+y²+a²`), the
second-order Riesz composite `R_iR_k f_a = −∂_i∂_k N_a` and the Leray projection
`P = I − ∇Δ⁻¹∇·` are **elementary** (the Calderón–Zygmund building blocks the
Leray projection needs), so they are exact outward-rounded `Interval`s. The
divergence of a Leray-projected vector blob is certified to enclose `0` by a
closed-form residual, and `blob_gradient` supplies `∇f_a` (radial). A far-field
`riesz_tail_bound` upgrades a finite-basis evaluation to a rigorous full-plane
statement. This is the substrate of the
[2-D Euler steady-vortex certificate](../cookbook/euler2d-vortex.md). Note the
*single* Riesz transform of a radial blob is **not** elementary (it carries the
half-Laplacian `|ξ|⁻¹`), which is why genuine SQG velocity is a recorded open
obligation rather than a closed form here.

::: omnibias.core.verified.riesz
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: 2-D SQG (single Riesz / Poisson blob)

The single Riesz transform that 2-D Euler's `riesz` basis *cannot* express in
closed form becomes elementary on the **2-D Poisson kernel** basis
`θ_a = a/(2π D^{3/2})`, whose Fourier symbol is `e^{−a|ξ|}` (so every half-power
of `−Δ` is a plain multiplication). The stream function
`ψ_a = (−Δ)^{−1/2}θ_a = 1/(2π D^{1/2})`, the **single Riesz transform**
`R_jθ_a = ∂_jψ_a = −x_j/(2π D^{3/2})` and the SQG velocity
`u = R^⊥θ_a = (y,−x)/(2π D^{3/2})` are all exact outward-rounded `Interval`s (the
half-power `D^{1/2}` via `Interval.sqrt`). `sqg_blob_gradient` is radial and
`sqg_velocity_divergence_residual` certifies `∇·u` encloses `0`. The whole-plane
`L²` inner product `sqg_blob_l2_inner(a, b) = 1/(2π(a+b)²)` diagonalises the
profile norm and powers the **self-similar obstruction** certificate
(`‖(y+R^⊥θ)·∇θ‖₂ ≥ ‖θ‖₂ > 0`). The closed forms are checked against an
independent `mpmath` Hankel transform of the symbol (and an `mpmath` quadrature for
the inner product). This is the substrate of the
[2-D SQG steady-vortex certificate](../cookbook/sqg-vortex.md) and discharges the
single-Riesz / half-Laplacian open obligation recorded by the Euler one.

::: omnibias.core.verified.sqg
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: the certified derivative tower (`sigma` / `transcend`)

`sigma_tower_interval(name, z, order)` is the rigorous twin of the float
fast-path kernels: given an `Interval` enclosure of `z` it returns guaranteed
enclosures of `(σ(z), σ'(z), …, σ^(order)(z))` from a *single* transcendental
evaluation (`transcend`) followed by exact-integer polynomial Horner steps, so the
tower stays tight and needs only one transcendental enclosure regardless of order.

Supported activations are `tanh`, `sigmoid`, `gaussian`, and the trigonometric
pair **`sin` / `cos`**. The trig pair closes the verified tower under classical
**Fourier-mode / plane-wave** fields: `sin`/`cos` are their own fourth derivative,
so the whole tower is the 4-cycle of phase shifts `(cos, −sin, −cos, sin)` /
`(sin, cos, −sin, −cos)` built from the two interval-range enclosures `cos_iv` /
`sin_iv`. Because `sin`/`cos` are **non-monotone**, those enclosures take the hull
of the endpoint brackets and saturate to `±1` only when an extremum (an even / odd
multiple of `π`, detected through an outward enclosure of `x/π`) lies inside the
argument interval. That membership test has **no false negatives**, so an interior
extremum is never missed and the enclosure is always sound.

!!! note "Differentiable vs. rigorous tower"
    The *differentiable* tower (`omnibias.{torch,jax,keras}`) has long supported
    `sin` / `cos` / `sinh` / `cosh` (see the [scope table](../scope-and-guarantees.md)
    §2). This entry is specifically about the **rigorous interval** tower in
    `omnibias.core.verified`, whose *certified* function class now matches a subset
    of the differentiable one — so a closed-form trigonometric solution can be
    residual-*certified*, not merely differentiated.

::: omnibias.core.verified.sigma
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.transcend
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: multivariate certified jet

The rigorous twin of the float multi-index jet kernels. Replaying the exact
`mlp_jet_mv` operation sequence in `Interval` arithmetic — with the input jet's
constant row seeded as the whole box — makes every intermediate an
inclusion-isotonic extension, so each row *encloses* `Dᵅu(x₀)/α!` for **every**
`x₀` in the box. Layers may use any tower activation
(`tanh` / `sigmoid` / `gaussian` / `sin` / `cos`), so closed-form Fourier-mode /
plane-wave fields are certifiable too. `certified_partials` / `jet_gradient` /
`jet_hessian` / `jet_laplacian` read out raw derivative enclosures;
`certified_residual_bound` turns them into a sup-norm PDE-residual bracket
(tightened by box subdivision).

::: omnibias.core.verified.jet_mv
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: Fundamental Theorem of Calculus

The Taylor-jet tower is **two-sided**. `antiderivative_jet` integrates a jet
term-by-term (`A₀ = constant`, `Aₘ = aₘ₋₁ / m`; the jet grows one order) and
`derivative_jet` differentiates it (`(k+1)·aₖ₊₁`; shrinks one order), with
`derivative_jet(antiderivative_jet(a)) == a` — **FTC part 1**, lossless. This is
exact term-by-term integration at the jet level (valid to any order for any
activation, integration constant free), *not* a pointwise `σ^(-n)` fast path:
repeated antiderivatives of an activation are non-elementary. The `torch` / `jax`
twins (`omnibias.torch.jet` / `omnibias.jax.jet`) are bit-identical.

`omnibias.core.verified.ftc` seals the **FTC identity**
`∫ₐᵇ σ^(k)(z) dz = σ^(k-1)(b) − σ^(k-1)(a)` (default `k=1`) as a tamper-evident v1
certificate for every verified-tower activation. The two sides are computed
independently — the left by integrating a rigorous `TaylorModel` of `σ^(k)`, the
right by pointwise endpoint towers — so the residual enclosing `0` is a genuine
cross-check, never a open-problem claim.

::: omnibias.core.verified.jet
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - antiderivative_jet
        - derivative_jet

::: omnibias.core.verified.ftc
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: complex intervals & validated Fourier series

`ComplexInterval` is a rigorous rectangular enclosure `[re] + i[im]` of a complex
quantity (sound `+ − × ÷`, `conj`, modulus). On top of it,
`ValidatedFourierSeries` is the `d`-dimensional weighted `ℓ¹_ν` Banach **algebra**
of Fourier coefficients: a finite block over the box `‖k‖∞ ≤ N` plus a tail
radius bounding `Σ_{‖k‖∞>N}|a_k|ν^{‖k‖₁}`. Convolution (`*`) keeps the kept block
exact and folds overflow rigorously into the tail; the **bounded** nonlocal
multipliers — Riesz `R_j = ik_j/|k|`, the coordinate Hilbert transform
`−i·sign(k_j)`, and Leray `P_{ab} = δ_{ab} − k_a k_b/|k|²` — act coefficient-wise
(tail factor `1`), so the SQG velocity `u = (−R₂, R₁)θ` and the gSQG family
become rigorous diagonal operators. `ν ≥ 1` is the *tight* sound boundary for
this two-sided algebra: complexifying each coordinate `x_d → x_d + is_d` shows
`ν` is `e^h` for a strip of half-width `h = ln ν` around the real torus, and
`ν < 1` is not a looser "formal" regime (unlike the one-sided
`ValidatedSeries`) because it makes submultiplicativity provably fail on
cancelling wavevectors like `i=(1,)`, `j=(−1,)`. `ν` also accepts a
length-`d` sequence of **anisotropic** per-axis weights `(ν₁, …, ν_d)`
(each `≥ 1`) for a direction-dependent strip — a strict superset of the
scalar case, since every bounded-multiplier proof only uses that the weight
is non-negative, never submultiplicativity itself.

::: omnibias.core.verified.complex_interval
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.fourier
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: self-dual Hermite-function basis

`hermite_basis` is a rigorous 1-D **physicists'** Hermite-function basis
`ψ_n(x) = H_n(x)·e^{-x²/2}`, evaluated by an interval Horner pass on the
*exact-integer* coefficients of `H_n` (`hermite_poly_coeffs_exact`) times a
rigorous `exp(-x²/2)` enclosure (`gaussian_weight`). Under the unitary Fourier
transform `F[f](k) = (2π)^{-1/2}∫f(x)e^{-ikx}dx`, the Hermite functions are
*exact* eigenfunctions, `F[ψ_n] = (−i)^n ψ_n`; because `(−i)^n` cycles through
the four exactly-representable values `{1,−i,−1,i}`,
`HermiteExpansion.fourier_transform_exact` applies it by exact component
permutation/sign-flip — bit-for-bit, not an outward-rounded product — and
since `|(−i)^n|=1` the same tail bound (`tail_bound`, reusing
`geometric_tail_bound` under a caller-supplied geometric coefficient-decay
hypothesis and a caller-supplied uniform bound on `|ψ_n(x)|`, e.g. the
classical Cramér inequality `sup_x|ψ̂_n(x)| ≤ π^{-1/4}` for the
`L²`-normalized basis) holds unchanged before and after the transform. A
1-D Cohn–Elkies packing bound ([cohn_elkies.md](cohn_elkies.md)) uses exactly
this: candidate test functions built from finitely many Hermite modes get an
*exact* (not FFT-truncated) Fourier transform as the same coefficient vector
up to the diagonal `(−i)^n` phases, turning the LP's simultaneous sign
constraints on `f` and `f̂` into constraints on one finite coefficient vector.
The Laguerre sibling ([laguerre_basis.md](laguerre_basis.md)) is orthogonal on
`[0, inf)` and is **not** Fourier self-dual.

```python
from omnibias.core.verified.hermite_basis import HermiteExpansion, CRAMER_UNIFORM_BOUND

expansion = HermiteExpansion.from_coeffs([1.0, 0.5, -0.25])
transformed = expansion.fourier_transform_exact()  # c_n -> (-i)^n c_n, exact
assert transformed.get(1).im.mid == -0.5  # (-i)^1 * 0.5 = -0.5i

bound = expansion.tail_bound(coeff_bound=1.0, ratio=0.5, psi_bound=CRAMER_UNIFORM_BOUND)
assert bound.lo >= 0.0
```

::: omnibias.core.verified.hermite_basis
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: Newton–Kantorovich / radii polynomial

The computer-assisted-proof workhorses: from an approximate zero `x̄` of `F` and an
approximate inverse `A` of `DF(x̄)`, prove a **true** zero exists (and is unique)
in an explicit ball. `radii_polynomial_certificate` verifies the scalar radii
polynomial `p(r) = Z₂r² − (1 − Z₀ − Z₁)r + Y₀`; `krawczyk_certificate` is the
finite-dimensional Krawczyk test; `newton_kantorovich_bounds` assembles
`(Y₀, Z₀, Z₁, Z₂)` for a map with a Lipschitz Jacobian.
`kantorovich_accept_step` (theory 08-04) is the optimizer policy: accept a
trial only when that unique-zero ball is nonempty. Empty is a valid reject.
The sealed payload records `continuum_pde_claim: false`. See
[Kantorovich-accepted Newton](kantorovich_newton.md).

::: omnibias.core.verified.kantorovich
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: PDE-residual certificates (network → certificate)

The end-to-end wiring that turns a trained network (or a spectral ansatz) into a
sealed certificate. A `LinearPDE` (with ready-mades `laplace` / `poisson` /
`helmholtz` / `screened_poisson` / `advection_diffusion`) becomes a residual
functional on the certified jet; `certified_interior_residual` and
`certified_boundary_residual` give rigorous sup-norm brackets, and
`aposteriori_error_certificate` combines them with a caller-supplied stability
constant into a certified bound `‖u_NN − u_true‖∞ ≤ C_Ω·R_int + C_∂·R_bnd`.
`StabilityEstimate` records where those constants came from,
`adaptive_certified_interior_residual` tightens the residual by subdivision, and
`structural_invariant` records algebraic identities such as hard constraints or
cage-enforced invariants in the sealed payload. Experimental nonlinear extensions
enter through `certified_custom_residual` / `certified_quadratic_reaction_residual`.
`radii_polynomial_residual_certificate` feeds a certified residual as the
Newton-Kantorovich defect `Y₀`, and `spectral_residual_norm` assembles the
residual `F(a) = La + Q(a) − rhs` of a band-limited Fourier ansatz with the
validated algebra. (Extract `(W, b, name)` layers from a trained `JetMLP` with
`omnibias.verify.verified_layers`, or run the convenience path
`omnibias.verify.certify_pinn_aposteriori`.) See the
[proof-carrying PDE cookbook](../cookbook/proof-carrying-pde.md) for the
end-to-end example.

::: omnibias.core.verified.pde_certificate
    options:
      show_root_heading: false
      heading_level: 3

## Verified backend: spectral existence proofs (radii polynomial in ℓ¹_ν)

A self-contained **computer-assisted existence proof** for a periodic solution of
the quadratic spectral problem `F(a) = ℓ·a + Q(a, a) − f = 0`, posed in the weighted
Fourier algebra `ℓ¹_ν`. From an approximate zero `ā` (a finite trigonometric
polynomial) `quadratic_radii_certificate` builds the **split** approximate inverse
`A` — a numerical inverse `A_N` of the finite Jacobian block on `‖k‖∞ ≤ N`, and the
exact diagonal `1/ℓ(k)` on the tail (bounded by `μ`) — assembles the rigorous
radii-polynomial bounds `(Y₀, Z₀, Z₁, Z₂)`, and (when a contracting radius exists)
returns a sealed certificate proving a **true** zero `a*` with `‖a* − ā‖_ν ≤ r`,
unique in that ball. `laplacian_symbol` / `laplacian_tail_inverse_bound` supply a
coercive diagonal linear part `ℓ(k) = c₀ + c₂|k|²`. That diagonal path stays
byte-identical. `BandedLinearPart` plus `tail_inverse_bound_from_banded` wires
nearest-neighbour (constant-coefficient) couplings into the same radii-polynomial
consumer; see [banded_tail.md](banded_tail.md). The full nonlocal IPM
streamfunction-Poisson operator remains out of scope (`full_ipm_proved=False`).

::: omnibias.core.verified.radii_spectral
    options:
      show_root_heading: false
      heading_level: 3

## Discovery engine

The finite proposer loop, catalog, condition language, and shared
observation class loop live in [`omnibias.core.proof`](discovery.md).
Cookbook: [Finite discovery engine](../cookbook/discovery-loop.md).

# omnibias-core

The pure-Python mathematical core: polynomial coefficient generators
and the backend-agnostic ActivationSpec.

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
    a *numerical* critical-strip enclosure, and the special values above are exact,
    but the Riemann Hypothesis remains a recorded *external* proof obligation —
    never inferred. A small enclosed magnitude near a putative zero is **not** a
    claim that `ζ` vanishes there. Nothing here makes a statement about the location
    of zeros of `ζ` / `L`, primality, factoring, or any cryptographic hardness
    assumption (see [scope & guarantees](../scope-and-guarantees.md) §6).

::: omnibias.core.verified.dirichlet
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
multipliers — Riesz `R_j = ik_j/|k|` and Leray `P_{ab} = δ_{ab} − k_a k_b/|k|²` —
act coefficient-wise (tail factor `1`), so the SQG velocity `u = (−R₂, R₁)θ` and
the gSQG family become rigorous diagonal operators. Requires `ν ≥ 1` (the
analytic regime where the weight is sub-multiplicative).

::: omnibias.core.verified.complex_interval
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.fourier
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
coercive diagonal linear part `ℓ(k) = c₀ + c₂|k|²`. The linear part must be diagonal
(a Fourier multiplier) and the nonlinearity `Q` bounded (`‖Q(u,v)‖ ≤ C_Q‖u‖‖v‖`);
the non-diagonal self-similar scaling operator `α + β x·∇` of a finite-time
singularity ansatz is the remaining ingredient and is documented as future work.

::: omnibias.core.verified.radii_spectral
    options:
      show_root_heading: false
      heading_level: 3

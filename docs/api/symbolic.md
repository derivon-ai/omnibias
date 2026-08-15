# omnibias-symbolic

Neural-jet equation discovery and interpretable surrogate modeling. Because every
omnibias activation exposes an exact n-th derivative fastpath, the discoverer can
build the full derivative jet `y, dy, d2y, ...` of a fitted field in closed form
and search for compact *implicit* relations among those generic coordinates - a
library-free SINDy variant that recovers `dy = y` (exp), `d2y = -y` (sin/cos),
and `dy = 1 - y^2` (tanh) without any named basis functions.

!!! tip "Looking for theory + worked examples?"
    This page is the **API reference** (precise signatures, auto-generated from
    docstrings). For the book-length, example-driven tour of every function -
    written to be AI-/vibe-coding-friendly - read the
    [**Discovery & Calculus Handbook**](../handbook/index.md). Quick jumps:
    [1-D neural jets](../handbook/01-neural-jet-1d.md) ·
    [vector calculus & PDE discovery](../handbook/02-vector-calculus-pde.md) ·
    [differential geometry](../handbook/03-differential-geometry.md) ·
    [exterior calculus](../handbook/04-exterior-calculus.md) ·
    [information theory](../handbook/05-information-theory.md) ·
    [optimal transport](../handbook/06-optimal-transport.md) ·
    [information geometry](../handbook/07-information-geometry.md).

See the example notebooks (`notebooks/13_*`-`notebooks/18_*`) and the applied
experiments under `examples/symbolic_discovery/`.

## Discovery engine

### Fields, jets & sparse equation search

::: omnibias.symbolic.discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - NeuralField1D
        - fit_neural_field_1d
        - exact_activation_field_1d
        - extract_neural_jets
        - JetBundle
        - SplitData
        - split_x_grid
        - jet_name
        - NeuralJetDiscoverer
        - JetDiscoveryResult
        - SparseEquation
        - build_jet_relation_library
        - fit_sparse_equation
        - discover_activation_identity
        - discover_from_noisy_observations

### Discrete recurrence discovery (integer-sequence twin)

The discrete analogue of jet discovery: instead of a fitted field's derivative
jet, `discover_recurrence` reads the **Newton forward-difference jet** of an
integer sequence and recovers the exact linear recurrence via an integer/rational
null space (`Fraction` arithmetic, no floating-point drift). It recovers both
C-finite recurrences (Fibonacci) and P-recursive ones with polynomial coefficients
(Catalan, Bell, partitions) — the latter beyond a plain constant-coefficient
least-squares baseline (`discover_recurrence_least_squares`), which is provided for
comparison. Builds on `omnibias-difference`'s umbral engine.

::: omnibias.symbolic.discrete
    options:
      show_root_heading: false
      heading_level: 4

### Feature libraries (Taylor / Fourier / CDF / information / fractional)

Five fractional column families ship, and they are **not** interchangeable — each
carries a different honesty label, which is what decides when to reach for it:

| builder | operator | label |
|---|---|---|
| `build_jet_fractional_features` | Grunwald-Letnikov grid convolution | **numerical** |
| `build_jet_fractional_features_closed_form` | analytic gamma-ratio jet sum, one terminal | **closed form** |
| `build_jet_piecewise_fractional_features` | the same, one Taylor jet per terminal | **closed form** |
| `build_jet_activation_fractional_features` | analytic identity for a named activation | **closed form** |
| `build_jet_spectral_fractional_features` | DST-I / DCT-II (or windowed FFT) symbol | **numerical-spectral** |

The closed-form columns have no discretisation error in `alpha`, only the truncation
of the Taylor tower they are built from, which is why they can separate neighbouring
candidate orders that the Grunwald-Letnikov columns cannot: GL discretises a
different lower-terminal convention, and refining the grid shrinks its column error
(measured 3.1% to 0.35% from 24 to 400 points) without fixing the order it selects.
The spectral columns sit in between — exact on the transform's basis modes, so the
error is the spectral truncation of the signal.

Two constraints worth knowing before use. The **piecewise** builder snaps each
terminal to its nearest grid sample, because the Taylor tower is read at a sample
and the operator must expand about that same point; distinct terminals must snap to
distinct samples and the lowest must snap to `x[0]`. The **spectral** builder's basis
is tied to a grid layout — `bc="dirichlet"` needs the interior grid
`x_j = (j+1) L/(N+1)` and `bc="neumann"` the midpoint grid `x_j = (j+1/2) L/N` — and
it rejects a mismatched grid rather than returning the plausible wrong column that
would otherwise result. Its windowed variant returns a complex array, of which the
column is the real part; the imaginary part is a windowing artefact, not a second
feature.

All five are wired into `NeuralJetDiscoverer` through their own `__init__` fields.
The grid-based families carry the usual guard that skips them when their
`source_order` equals the candidate left-hand-side order; the activation family
needs no such guard, since those columns are a function of `x` alone. When a bundle
carries appended target columns, set `piecewise_fractional_tower_width` so they are
not read as high derivatives.

#### Integral (non-local) columns

Every family above is *local*: a row is determined by the signal near that point.
`build_jet_integral_features` is not, and that is the point — it is what lets a
search express an **integro-differential** law such as `y'(x) = -∫₀ˣ y`, or a
Fredholm equation of the second kind `y = f + λ ∫ₐᵇ K(x,t) y(t) dt`, neither of
which any local library can represent at all. Three families: the running integral,
global **Fredholm** columns for user-supplied kernels, and causal **Volterra**
columns. On the field/PDE side, `measure_integral_columns` packages the Fredholm
family as an `extra_columns_fn` for `FieldLawDiscoverer`, so a nonlocal term sits
next to the local differential atoms with no new plumbing.

Honesty label: **numerical**. Every column is a quadrature. The Fredholm family
takes its weights from an `omnibias.measure.Measure` and so inherits that rule's
accuracy — 24 Gauss-Legendre nodes reach round-off where 401 trapezoid points reach
`1e-6` — while the causal families are cumulative trapezoid, because a global rule's
weights are coefficients for the whole interval, not the measure of a neighbourhood
of each node, and so cannot be restricted to a prefix.

Three things behave differently here than in a local library, and all three will
otherwise cost you a wrong answer rather than an error:

- **The measure's nodes must be the bundle's grid.** The signal is known at the
  samples and nowhere else, so any other node set could only be honoured by
  interpolating. Mismatched nodes raise. Splits normally differ in resolution, so
  pass a factory `x -> Measure` rather than a fixed measure.
- **The lower terminal is per-bundle.** An indefinite integral is defined only up
  to a constant, fixed by where the grid starts — so two splits beginning at
  different `x` carry genuinely different columns and a law fitted on one will not
  transfer. Pass a shared `origin`, or give every split the same starting point.
- **Identifiability is not automatic.** A rank-1 separable kernel `K(x,t) = x·t`
  produces a column exactly proportional to `x`, which no fit can tell apart from
  the `x` column. This is why `discover` now reports `design_conditioning_report`:
  `standardized_condition_number` is the one that matters (the raw number is
  dominated by scaling, which `fit_sparse_equation` already divides out), and it
  reads ~10¹⁵ for the degenerate kernel against ~1 for an identifiable one.

The left-hand-side guard also runs the other way round from the derivative
families. Integrating the jet one order *above* the LHS reproduces it (`∫y' = y`),
so the causal families are dropped when `integral_source_order == lhs_order + 1`. A
global Fredholm column carries no such identity and is never dropped — putting the
unknown under the integral as well as on the left is precisely what a Fredholm
equation is.

::: omnibias.symbolic.discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - LibrarySpec
        - FeatureLibraryPlan
        - build_taylor_library
        - build_fourier_library
        - build_hybrid_library
        - build_cdf_band_library
        - build_information_library
        - build_pde_operator_library
        - build_jet_activation_fractional_features
        - build_jet_cdf_features
        - build_jet_fractional_features
        - build_jet_fractional_features_closed_form
        - build_jet_info_features
        - build_jet_integral_features
        - build_jet_piecewise_fractional_features
        - build_jet_spectral_fractional_features
        - fit_cdf_band_library_plan
        - fit_information_library_plan
        - fit_jet_cdf_plan
        - fit_jet_info_plan
        - fit_screened_feature_library_plan
        - gl_fractional_derivative

### Surrogates, PDE operators, datasets, metrics & evaluation

::: omnibias.symbolic.discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - discover_interpretable_surrogate
        - default_surrogate_specs
        - symbolic_hidden_law
        - discover_pde_operator_law
        - discover_fractional_order_law
        - FractionalOrderDiscoveryResult
        - make_heat_equation_operator_data
        - make_symbolic_regression_dataset
        - make_high_dim_sparse_dataset
        - evaluate_high_dim_sparse_validation
        - evaluate_real_world_tabular_validation
        - evaluate_poc
        - mae
        - rmse
        - write_artifacts

## Discovery rigor: coefficient uncertainty & model selection

Confidence intervals, certified (verified-interval) enclosures, and principled
model-order selection for discovered sparse equations. See the handbook section
[Coefficient uncertainty & model selection](../handbook/01-neural-jet-1d.md#coefficient-uncertainty-model-selection).

### Coefficient uncertainty

::: omnibias.symbolic.uncertainty
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - bootstrap_coefficients
        - ridge_coefficient_covariance
        - certified_coefficient_intervals
        - attach_uncertainty

### Model selection (information criteria, k-fold, stability)

::: omnibias.symbolic.selection
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - aic
        - aicc
        - bic
        - mdl
        - information_criterion
        - equation_information_criterion
        - gaussian_log_likelihood
        - kfold_select
        - KFoldSelection
        - stability_selection

## Multivariate fields & PDE discovery

Closed-form mixed-partial jets of an `N`-dimensional field, the full
vector-calculus operator surface, and the PDE law discoverer. See
[Handbook chapter 2](../handbook/02-vector-calculus-pde.md).

### Fields & field jets

::: omnibias.symbolic.field_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - NeuralFieldND
        - fit_neural_field_nd
        - FieldJet
        - extract_field_jet
        - analytic_field_jet
        - field_derivative_jet
        - field_partial_name

### Vector-calculus operators

::: omnibias.symbolic.field_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - field_value
        - field_gradient
        - field_hessian
        - field_laplacian
        - field_grad_norm_sq
        - field_divergence
        - field_curl
        - field_ito_generator
        - field_anisotropic_laplacian
        - field_wirtinger
        - field_operator_columns

### PDE law discovery & canonical datasets

::: omnibias.symbolic.field_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - FieldLawDiscoverer
        - FieldLawResult
        - build_field_relation_library
        - discover_field_pde_law
        - evaluate_field_pde_discovery
        - make_heat_field_split
        - make_heat2d_field_split
        - make_wave_field_split
        - make_burgers_field_split
        - make_laplace_field_split
        - measure_integral_columns

## Piecewise / switched-law discovery

`omnibias.symbolic.piecewise` (a bridge on the [`omnibias-partition`](partition.md)
keystone) discovers **regime-dependent** laws that a single global SINDy fit
would average away. The driver fits one global neural field, partitions the
input space with a soft partition-of-unity, then per region extracts a jet and
runs `fit_sparse_equation` — emitting a per-region `SparseEquation` plus the
hardened `if w·x > t` switch conditions, packaged as a `HybridAutomaton`.
`fit_piecewise_ode_law` is the end-to-end neural-ODE twin (recovering both laws
and the switch surface of a switched dynamical system).

**Honesty label.** STLSQ polish stays numpy / non-differentiable; **gates and
per-cell coefficients** are differentiable (Adam on the soft-weighted residual
``F = sum_l w_l(x) (xi_l · phi(u))``, then harden + STLSQ). The `beta -> inf`
hardened switch is the feasibility / temperature sense of "collapse", never the
founding `delta -> 0` bias collapse. Per-region laws are validated on held-out
data. Vector systems share gates and print `k` formulas in `report()`;
`fit_learned_piecewise_ode` learns those gates (no oracle partition). A
depth-1 SoftTree or `H=1` Arrangement trained on the trajectory's
finite-difference `du` can be hardened via `tree_params` /
`arrangement_params` from the **fitted** split and fed to
`fit_piecewise_law` on a field jet -- that is a tab head, not the
learned-gate ansatz. The Arrangement constructor is **unplanted** (random
`W`, no `e_0` axis init); that path does not call `_refine_split_threshold`.
STLSQ still uses the field jet. Recipe:
[`cookbook/piecewise-hybrid-automaton.md`](../cookbook/piecewise-hybrid-automaton.md).

::: omnibias.symbolic.piecewise
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - PiecewiseLaw
        - HybridAutomaton
        - fit_piecewise_law
        - fit_piecewise_ode_law
        - fit_learned_piecewise_ode
        - global_sparse_law
        - polynomial_value_library
        - polynomial_vector_library

## Gauge-covariant singlet discovery

Optional extra `omnibias-symbolic[gauge]`. Searches a generated
`GaugeInvariantDictionary` of Weyl singlets -- never a coordinate `FieldJet`
of `A_mu^a` and never the 480-component SU(3) 2-jet. Recovers classical local
identities (BPST: `tr(F^2) ~ 8 pi^2 tr(F*Ftilde)`). Complexity is
representation-theoretic (mass dimension + traces). `weak_ym_columns` is an
identity check, not a predict-zero STLSQ headline; 1-D integral features are
not the 4-D Yang-Mills weak form. Wilson / Polyakov traces are a second
language (`LoopLawDiscoverer` on links), not an expanded jet. Not a
Yang-Mills mass-gap claim.

::: omnibias.symbolic.gauge_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - GaugeLawDiscoverer
        - GaugeLawResult
        - discover_yang_mills_singlet_law
        - discover_yang_mills_invariant_law
        - make_yang_mills_bpst_split
        - make_yang_mills_polynomial_split
        - weak_ym_columns

::: omnibias.symbolic.loop_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - LoopLawDiscoverer
        - LoopLawResult
        - discover_wilson_plaquette_law
        - discover_planted_area_law
        - evaluate_loop_gauge_invariance
        - planted_area_law_table

## Differential geometry

Riemannian metric fields, the Levi-Civita connection, curvature tensors, and
learned pullback charts. See
[Handbook chapter 3](../handbook/03-differential-geometry.md).

### Metric fields & algebra

::: omnibias.symbolic.geometry_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - MetricField
        - analytic_metric_field
        - flat_metric_field
        - warped_product_metric_field
        - pullback_metric_field
        - metric_inverse
        - metric_determinant

### Connection, operators & curvature

::: omnibias.symbolic.geometry_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - christoffel_symbols
        - covariant_hessian
        - laplace_beltrami
        - metric_grad_norm_sq
        - riemann_tensor
        - ricci_tensor
        - scalar_curvature
        - gaussian_curvature_2d

### Geometric law discovery

::: omnibias.symbolic.geometry_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - discover_geometric_heat_law
        - make_geometric_heat_split
        - evaluate_geometric_discovery

## Exterior calculus (de Rham-Hodge)

Differential forms, the exterior derivative, the Hodge star and codifferential,
the Hodge Laplacian, and identity certification. See
[Handbook chapter 4](../handbook/04-exterior-calculus.md).

::: omnibias.symbolic.exterior_discovery
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - DifferentialForm
        - scalar_form
        - one_form
        - differential_form
        - exterior_derivative
        - hodge_star
        - codifferential
        - hodge_laplacian
        - wedge
        - closedness_residual
        - coclosedness_residual
        - gradient_form
        - curl_form
        - electromagnetic_field_2form
        - evaluate_exterior_calculus

## Information-theoretic diagnostics

NumPy point-estimate divergences, differential entropy, and residual-quality
reports for model selection. See
[Handbook chapter 5](../handbook/05-information-theory.md).

::: omnibias.symbolic.diagnostics
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - entropy
        - kl_divergence
        - js_divergence
        - mutual_information
        - total_variation_distance
        - hellinger_distance
        - chi_squared_divergence
        - renyi_divergence
        - differential_entropy
        - gaussian_entropy
        - histogram_pmf
        - kl_to_gaussian
        - js_to_gaussian
        - total_variation_to_gaussian
        - hellinger_to_gaussian
        - chi_squared_to_gaussian
        - renyi_to_gaussian
        - wasserstein_to_gaussian
        - wasserstein2_to_gaussian
        - feature_residual_mutual_information
        - residual_distribution_report
        - residual_dependence_report
        - surrogate_residual_diagnostics
        - divergence_objective_term

## Causal parent-ranking (MI + NOTEARS-lite)

A light, numpy-only causal-discovery layer that ranks candidate library terms as
*parents* of a target. It pairs the model-free Miller-Madow mutual-information
screen with a NOTEARS-lite continuous-acyclicity learner of a linear-SEM weighted
adjacency. **Honest scope:** the output is a directed *ranking*, not a certified
DAG -- linear-Gaussian direction identifiability needs assumptions (equal noise,
raw scale) and MI alone is undirected. See the worked demo under
`examples/symbolic_discovery/causal_term_discovery/`.

::: omnibias.symbolic.causal
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - notears_acyclicity
        - notears_lite
        - mutual_information_matrix
        - term_parent_ranking
        - causal_discovery_report

## Dimensional analysis (Buckingham-Pi)

Exact integer null-space of the unit-dimension matrix: the dimensionless
:math:`\Pi`-groups of a set of physical variables, plus a filter that restricts a
candidate monomial library to its dimensionless members (the dimensional-analysis
prior for discovery). Pure `fractions` arithmetic -- no floating point. See the
demo `examples/symbolic_discovery/dimensional_groups/`.

::: omnibias.symbolic.dimensional
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - DimensionalSystem
        - PiGroup
        - dimension_matrix
        - integer_null_space
        - n_dimensionless_groups
        - buckingham_pi_groups
        - dimensionless_residual
        - is_dimensionless
        - filter_dimensionless_monomials

## Latent-state ODE discovery (partial observability)

Recover a hidden dynamical law from a **single observed coordinate**: a Takens
delay-embedding lifts the scalar series back to a state space, an autoencoder
(linear PCA, exact for linear latent dynamics, or a small nonlinear MLP)
compresses it to the latent dimension, and
[`FieldLawDiscoverer`](#omnibias.symbolic.field_discovery.FieldLawDiscoverer)
recovers `dz_i/dt = f_i(z)`. The latent frame is fixed only up to a
diffeomorphism, so the **eigenvalues** (frequency / growth rate) are the honest,
coordinate-invariant claim. Demo: `examples/symbolic_discovery/latent_ode_discovery/`.

::: omnibias.symbolic.latent
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - takens_embedding
        - finite_difference_derivative
        - LinearAutoencoder
        - MLPAutoencoder
        - discover_latent_ode
        - LatentODEResult

## Explicit expressions

::: omnibias.symbolic.expressions
    options:
      show_root_heading: false
      heading_level: 3

## Blasius boundary layer

::: omnibias.symbolic.blasius
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - solve_blasius
        - discover_blasius_identity
        - discover_blasius_from_neural_surrogate
        - discover_blasius_explicit_expression
        - discover_blasius_taylor_pade_expression
        - evaluate_blasius

## CCF CAP validation

::: omnibias.symbolic.ccf
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - verify_cap_bundle
        - assess_ccf_candidate
        - recover_ccf_scaling_law
        - verify_ccf_residual
        - ccf_self_similar_residual
        - ccf_self_similar_line_residual
        - line_even_profile_jet
        - verify_ccf_selfsimilar_blowup_attempt
        - verify_ccf_linearized_operator_bound
        - periodic_hilbert

## 2-D Euler vortex validation

Numpy-only second source for the
[2-D Euler steady-vortex certificate](../cookbook/euler2d-vortex.md). It
reimplements the Biot--Savart velocity, the vorticity gradient and the
second-order Riesz building blocks from scratch, re-confirms the exact steady
state / divergence / Calderon--Zygmund identities on a grid, and densely samples
the radial norm magnitudes to check the reported sups genuinely dominate them
(anti-faking). It asserts no Euler blow-up and keeps `unproven_claim=False`.

::: omnibias.symbolic.euler2d
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - verify_euler2d_steady_vortex
        - euler2d_radial_fields
        - euler2d_grid_residuals

## Periodic Navier–Stokes residual validation

Numpy-only second source for the
[proof-carrying fluid-dynamics certificate](../cookbook/proof-carrying-fluid-dynamics.md).
It regenerates the periodic flow from the certificate's `fixture` descriptor with
its own closed-form Taylor–Green / Kolmogorov expressions, recomputes the
momentum / continuity / pressure-Poisson residual sups with an independent
spectral implementation, and confirms the recorded sups match (catching both
under- and over-statement) and that `exact_solution_claim` is consistent with the
measured residual. It asserts no Navier–Stokes regularity result and keeps
`unproven_claim=False`.

::: omnibias.symbolic.fluid
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - verify_periodic_flow_residual
        - regenerate_periodic_flow
        - periodic_flow_residual_sups
        - finite_difference_residual_sups
        - streamfunction_residual_sups
        - verify_streamfunction_residual
        - verify_rollout_diagnostics

## 2-D SQG vortex validation

Numpy-only second source for the
[2-D SQG steady-vortex certificate](../cookbook/sqg-vortex.md). It reimplements
the SQG velocity `u = R^⊥θ`, the temperature gradient and the single Riesz
transform from scratch, re-confirms the exact steady state / divergence /
`u = R^⊥θ` identities on a grid, and densely samples the radial norm magnitudes
to check the reported sups genuinely dominate them (anti-faking). It asserts no
SQG singularity and keeps `unproven_claim=False`.

`verify_sqg_selfsimilar_blowup_attempt` (with `sqg_selfsimilar_l2_quantities`) is
the second source for the **self-similar obstruction** certificate: it recomputes
\(\lVert\Theta\rVert_2^2\), \(\langle F,\Theta\rangle\) and \(\lVert F\rVert_2\) by
independent radial quadrature, confirms the no-go identity
\(\langle F,\Theta\rangle=-\lVert\Theta\rVert_2^2\) and the obstruction lower bound
\(\lVert F\rVert_2\ge\lVert\Theta\rVert_2>0\), and catches a forged understated
norm or a profile-existence overclaim.

`verify_sqg_linearized_coercivity_attempt` (with `sqg_grad_theta_sup_sample`) is the
second source for the **linearized \(L^2\) coercivity** diagnostic: it densely
re-samples \(\lVert\nabla\bar\Theta\rVert_\infty\) (the certified whole-plane sup
must dominate it — anti-faking), re-derives the Weyl gap
\(1-\lVert\nabla\bar\Theta\rVert_\infty\) and the
`certified_block_operator_gap` engine value, and checks the honesty flags
(`blowup`/`unproven`/`three_d`/`stability` all `False`).

::: omnibias.symbolic.sqg
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - verify_sqg_steady_vortex
        - verify_sqg_selfsimilar_blowup_attempt
        - verify_sqg_linearized_coercivity_attempt
        - sqg_selfsimilar_l2_quantities
        - sqg_grad_theta_sup_sample
        - sqg_radial_fields
        - sqg_grid_residuals

## Navier-Stokes certificate validation

The Navier-Stokes symbolic replay path recomputes CAP residuals and refined
axisymmetric candidates without importing the PINN certificate builders. For
interval reports it checks coefficient containment, certified tail flags,
continuum residual intervals, finite-energy intervals, and axis-certification
metadata while preserving `unproven_claim=False`. Bridge artifacts are replayed by
replaying their axisymmetric, regularity-trace, and blow-up-trace children.
Blow-up closure reports replay the nested interval report and recompute the
radii-polynomial arithmetic/obligation booleans; regularity inequality reports
recompute trace residuals, coefficient containment, and deterministic
counterexample sweeps. The theorem-grade closure replay checks the new
operator/radii/norm/regularity proof-attempt booleans and open obligations
against serialized evidence. These checks validate certificate consistency, not a
global-regularity theorem. The proof-program replay checks exact equation contracts,
function-space definitions, candidate-family status, lemma packages, Lean import
metadata, and external-review gate consistency.
Proof-obligation replay recomputes each obligation hash and verifier-bundle
replay checks theorem names, obligation hashes, source hashes, verifier identity,
freshness metadata, and accepted/rejected obligation sets.

::: omnibias.symbolic.navier_stokes
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - verify_ns_cap_bundle
        - assess_navier_stokes_candidate
        - regularity_feature_vector
        - run_regularity_search
        - build_regularity_candidate_artifact
        - fit_regularity_growth_bound
        - fit_self_similar_blowup_rate
        - assess_blowup_candidate
        - build_blowup_candidate_artifact
        - build_axisymmetric_candidate_bridge_artifacts
        - replay_candidate_artifact
        - verify_axisymmetric_candidate_bridge_artifacts
        - verify_axisymmetric_swirl_candidate_artifact
        - verify_refined_axisymmetric_swirl_candidate_artifact
        - verify_axisymmetric_interval_report
        - verify_blowup_closure_report
        - verify_regularity_inequality_report
        - verify_theorem_grade_closure_attempt
        - verify_ns_proof_program_report
        - verify_proof_obligation_bundle
        - verify_theorem_verifier_bundle

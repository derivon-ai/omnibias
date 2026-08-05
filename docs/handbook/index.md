# The omnibias Discovery & Calculus Handbook

> A book-length, example-driven reference for the closed-form **calculus,
> probability, information-theory and geometry** surface of omnibias — the
> machinery behind `omnibias-symbolic` (neural-jet equation discovery) and the
> math primitives it stands on.

Every operator in this handbook is **closed form**: it reads exact derivatives
off an omnibias activation tower (one `sigma` evaluation per order), so there is
no finite-differencing and no autodiff graph to babysit. The same math is
bit-identical across the PyTorch and JAX backends.

!!! tip "Reading this as an AI agent (vibe coding)"
    Start with the [AI quickstart](ai-quickstart.md): it is a dense, copy-paste
    cheat-sheet (import map, capability table, the five canonical recipes, and the
    gotchas that actually bite). Every function below follows the **same entry
    template** — *What · When · Theory · Example · Returns* — so you can grep for a
    name and paste a working snippet. All examples are self-contained and run on
    CPU in float64.

## How the book is organised

The handbook is a layered tower. Each chapter builds on the one before, from a
single scalar field to curved-manifold PDEs and the de Rham complex:

```
            scalar field  u(x)                      Chapter 1  (1-D neural jet)
                  │  add input variables
                  ▼
        vector field  u(x_1,...,x_d)                Chapter 2  (vector calculus, PDEs)
                  │  add a metric g_ij
                  ▼
   Riemannian manifold  (M, g)                      Chapter 3  (differential geometry)
                  │  antisymmetrise
                  ▼
   differential forms  ω ∈ Λ^k                      Chapter 4  (exterior / de Rham–Hodge)

        residuals & distributions                   Chapter 5  (information theory)
        coupling two distributions                  Chapter 6  (optimal transport)
        the geometry of a model family              Chapter 7  (information geometry)
```

## Table of contents

### [Chapter 1 — The neural jet in 1-D](01-neural-jet-1d.md)

Fit a smooth field, read off its exact derivative tower `y, y', y'', …`, and
discover the hidden ODE / identity (`y' = y`, `y'' = -y`, `y' = 1 - y²`, …).

- Fields: [`fit_neural_field_1d`](01-neural-jet-1d.md#fit_neural_field_1d) ·
  [`NeuralField1D`](01-neural-jet-1d.md#neuralfield1d) ·
  [`exact_activation_field_1d`](01-neural-jet-1d.md#exact_activation_field_1d) ·
  [`extract_neural_jets`](01-neural-jet-1d.md#extract_neural_jets) ·
  [`JetBundle`](01-neural-jet-1d.md#jetbundle) ·
  [`jet_name`](01-neural-jet-1d.md#jet_name)
- Discovery: [`NeuralJetDiscoverer`](01-neural-jet-1d.md#neuraljetdiscoverer) ·
  [`JetDiscoveryResult`](01-neural-jet-1d.md#jetdiscoveryresult) ·
  [`SparseEquation`](01-neural-jet-1d.md#sparseequation) ·
  [`discover_activation_identity`](01-neural-jet-1d.md#discover_activation_identity) ·
  [`discover_from_noisy_observations`](01-neural-jet-1d.md#discover_from_noisy_observations)
- Sparse regression: [`fit_sparse_equation`](01-neural-jet-1d.md#fit_sparse_equation) ·
  [`build_jet_relation_library`](01-neural-jet-1d.md#build_jet_relation_library) ·
  [`gl_fractional_derivative`](01-neural-jet-1d.md#gl_fractional_derivative) ·
  [`build_jet_fractional_features`](01-neural-jet-1d.md#build_jet_fractional_features)
- Surrogates & libraries: [`discover_interpretable_surrogate`](01-neural-jet-1d.md#discover_interpretable_surrogate) ·
  [`build_taylor_library`](01-neural-jet-1d.md#library-builders) ·
  [`build_fourier_library`](01-neural-jet-1d.md#library-builders) ·
  [`build_hybrid_library`](01-neural-jet-1d.md#library-builders) ·
  [`build_information_library`](01-neural-jet-1d.md#cdf-information-surrogate-libraries) ·
  [`build_cdf_band_library`](01-neural-jet-1d.md#cdf-information-surrogate-libraries)
- PDE operators: [`discover_pde_operator_law`](01-neural-jet-1d.md#datasets-metrics-and-evaluators) ·
  [`build_pde_operator_library`](01-neural-jet-1d.md#datasets-metrics-and-evaluators) ·
  [`make_heat_equation_operator_data`](01-neural-jet-1d.md#datasets-metrics-and-evaluators)
- Datasets, metrics, plans & evaluators:
  [the full list](01-neural-jet-1d.md#datasets-metrics-and-evaluators)

### [Chapter 2 — Vector calculus & PDE discovery](02-vector-calculus-pde.md)

Closed-form mixed partials of a `d`-input field, the whole vector-calculus
operator surface, and multivariate PDE-law recovery (Laplace, heat, wave,
Burgers).

- Fields & jets: [`fit_neural_field_nd`](02-vector-calculus-pde.md#fit_neural_field_nd) ·
  [`NeuralFieldND`](02-vector-calculus-pde.md#neuralfieldnd) ·
  [`extract_field_jet`](02-vector-calculus-pde.md#extract_field_jet) ·
  [`FieldJet`](02-vector-calculus-pde.md#fieldjet) ·
  [`analytic_field_jet`](02-vector-calculus-pde.md#analytic_field_jet) ·
  [`field_derivative_jet`](02-vector-calculus-pde.md#field_derivative_jet)
- Operators: [`field_gradient`](02-vector-calculus-pde.md#field_gradient) ·
  [`field_hessian`](02-vector-calculus-pde.md#field_hessian) ·
  [`field_laplacian`](02-vector-calculus-pde.md#field_laplacian) ·
  [`field_divergence`](02-vector-calculus-pde.md#field_divergence) ·
  [`field_curl`](02-vector-calculus-pde.md#field_curl) ·
  [`field_grad_norm_sq`](02-vector-calculus-pde.md#field_grad_norm_sq) ·
  [`field_ito_generator`](02-vector-calculus-pde.md#field_ito_generator) ·
  [`field_anisotropic_laplacian`](02-vector-calculus-pde.md#field_anisotropic_laplacian) ·
  [`field_wirtinger`](02-vector-calculus-pde.md#field_wirtinger) ·
  [`field_value`](02-vector-calculus-pde.md#field_value)
- Discovery: [`FieldLawDiscoverer`](02-vector-calculus-pde.md#fieldlawdiscoverer) ·
  [`build_field_relation_library`](02-vector-calculus-pde.md#build_field_relation_library) ·
  [`field_operator_columns`](02-vector-calculus-pde.md#field_operator_columns) ·
  [`field_partial_name`](02-vector-calculus-pde.md#field_partial_name) ·
  [`discover_field_pde_law`](02-vector-calculus-pde.md#discover_field_pde_law)
- Canonical PDE datasets: [`make_laplace_field_split`](02-vector-calculus-pde.md#canonical-pde-datasets)
  and the heat / wave / Burgers / 2-D heat twins.

### [Chapter 3 — Differential geometry](03-differential-geometry.md)

A metric `MetricField`, the Laplace–Beltrami operator with its Christoffel
drift, Riemann/Ricci/scalar curvature, the pullback metric of a learned chart,
and heat flow on the round sphere.

- Metric: [`MetricField`](03-differential-geometry.md#metricfield) ·
  [`analytic_metric_field`](03-differential-geometry.md#analytic_metric_field) ·
  [`flat_metric_field`](03-differential-geometry.md#flat_metric_field) ·
  [`warped_product_metric_field`](03-differential-geometry.md#warped_product_metric_field) ·
  [`metric_inverse`](03-differential-geometry.md#metric_inverse-and-metric_determinant) ·
  [`metric_determinant`](03-differential-geometry.md#metric_inverse-and-metric_determinant)
- Connection & operators: [`christoffel_symbols`](03-differential-geometry.md#christoffel_symbols) ·
  [`covariant_hessian`](03-differential-geometry.md#covariant_hessian) ·
  [`laplace_beltrami`](03-differential-geometry.md#laplace_beltrami) ·
  [`metric_grad_norm_sq`](03-differential-geometry.md#metric_grad_norm_sq)
- Curvature: [`riemann_tensor`](03-differential-geometry.md#riemann_tensor) ·
  [`ricci_tensor`](03-differential-geometry.md#ricci_tensor) ·
  [`scalar_curvature`](03-differential-geometry.md#scalar_curvature) ·
  [`gaussian_curvature_2d`](03-differential-geometry.md#gaussian_curvature_2d)
- Learned charts & discovery: [`pullback_metric_field`](03-differential-geometry.md#pullback_metric_field) ·
  [`make_geometric_heat_split`](03-differential-geometry.md#make_geometric_heat_split) ·
  [`discover_geometric_heat_law`](03-differential-geometry.md#discover_geometric_heat_law) ·
  [`evaluate_geometric_discovery`](03-differential-geometry.md#evaluate_geometric_discovery)

### [Chapter 4 — Exterior calculus & the de Rham–Hodge complex](04-exterior-calculus.md)

Differential forms whose components are neural jets, the exterior derivative
`d`, the Hodge star, the codifferential, the Hodge Laplacian — and the
`d∘d = 0` identity that unifies `curl grad` and `div curl`.

- Forms: [`DifferentialForm`](04-exterior-calculus.md#differentialform) ·
  [`scalar_form`](04-exterior-calculus.md#scalar_form) ·
  [`one_form`](04-exterior-calculus.md#one_form) ·
  [`differential_form`](04-exterior-calculus.md#differential_form)
- Operators: [`exterior_derivative`](04-exterior-calculus.md#exterior_derivative) ·
  [`hodge_star`](04-exterior-calculus.md#hodge_star) ·
  [`codifferential`](04-exterior-calculus.md#codifferential) ·
  [`hodge_laplacian`](04-exterior-calculus.md#hodge_laplacian) ·
  [`wedge`](04-exterior-calculus.md#wedge)
- Correspondence & physics: [`gradient_form`](04-exterior-calculus.md#gradient_form-and-curl_form) ·
  [`curl_form`](04-exterior-calculus.md#gradient_form-and-curl_form) ·
  [`electromagnetic_field_2form`](04-exterior-calculus.md#electromagnetic_field_2form) ·
  [`closedness_residual`](04-exterior-calculus.md#closedness_residual-and-coclosedness_residual) ·
  [`coclosedness_residual`](04-exterior-calculus.md#closedness_residual-and-coclosedness_residual) ·
  [`evaluate_exterior_calculus`](04-exterior-calculus.md#evaluate_exterior_calculus)

### [Chapter 5 — Information theory & divergences](05-information-theory.md)

Entropy, KL/JS/Rényi/χ²/Hellinger/total-variation divergences and mutual
information in three flavours: differentiable (JAX/Torch), NumPy diagnostics,
and **certified interval enclosures**.

- Differentiable: [`entropy`](05-information-theory.md#entropy) ·
  [`kl_divergence`](05-information-theory.md#kl_divergence) ·
  [`js_divergence`](05-information-theory.md#js_divergence) ·
  [`renyi_divergence`](05-information-theory.md#generalised-divergences-entropies) ·
  [`f_divergence`](05-information-theory.md#generalised-divergences-entropies) ·
  [`mutual_information`](05-information-theory.md#mutual_information) and twins.
- Diagnostics (NumPy): [`differential_entropy`](05-information-theory.md#histogram_pmf-gaussian_entropy-differential_entropy) ·
  [`feature_residual_mutual_information`](05-information-theory.md#feature_residual_mutual_information) ·
  [`surrogate_residual_diagnostics`](05-information-theory.md#residual-reports) ·
  [`divergence_objective_term`](05-information-theory.md#divergence_objective_term-and-divergence_objectives)
- Certified enclosures: [`entropy_enclosure`](05-information-theory.md#the-family) ·
  [`kl_divergence_enclosure`](05-information-theory.md#the-family) and the family.

### [Chapter 6 — Optimal transport](06-optimal-transport.md)

1-D Wasserstein, the sliced and Sinkhorn approximations, the Gaussian
closed form, and certified Wasserstein enclosures.

- Differentiable: [`wasserstein1`](06-optimal-transport.md#wasserstein1) ·
  [`wassersteinp`](06-optimal-transport.md#wassersteinp) ·
  [`wasserstein2_gaussian`](06-optimal-transport.md#wasserstein2_gaussian) ·
  [`sliced_wasserstein`](06-optimal-transport.md#sliced_wasserstein) ·
  [`sinkhorn_distance`](06-optimal-transport.md#sinkhorn_distance)
- Certified: [`certified_wasserstein1_samples`](06-optimal-transport.md#certified_wasserstein1_samples-and-certified_wasserstein2_samples) ·
  [`certified_wasserstein2_gaussian`](06-optimal-transport.md#certified_wasserstein2_gaussian) and twins.

### [Chapter 7 — Information geometry & natural gradient](07-information-geometry.md)

The Fisher metric, exponential-family cumulants, GLM mean/variance, and the
damped natural-gradient step that ties `omnibias-curvature` to a probabilistic
model family.

- Fisher & families: [`fisher_information`](07-information-geometry.md#glm_mean-glm_variance-fisher_information) ·
  [`exponential_family_cumulants`](07-information-geometry.md#exponential_family_cumulants) ·
  [`fit_natural_parameter`](07-information-geometry.md#moment_match-and-fit_natural_parameter) ·
  [`glm_mean`](07-information-geometry.md#glm_mean-glm_variance-fisher_information) · [`glm_variance`](07-information-geometry.md#glm_mean-glm_variance-fisher_information) ·
  [`moment_match`](07-information-geometry.md#moment_match-and-fit_natural_parameter)
- Natural gradient: [`damped_solve`](07-information-geometry.md#damped_solve) ·
  [`natural_gradient_step`](07-information-geometry.md#natural_gradient_step) ·
  [`glm_loss_gradient`](07-information-geometry.md#glm_loss_gradient) ·
  [`glm_natural_gradient_step`](07-information-geometry.md#glm_natural_gradient_step)

## The contract, in one paragraph

Pick an activation with a derivative fastpath (`tanh`, `sigmoid`, `gelu`, …). A
random-feature field `u(x) = c·σ(Wx̃ + β) + b` has an **affine** inner map, so the
multivariate Faà di Bruno chain rule collapses to a single surviving term and
*every* mixed partial is the exact closed form
\[
  \partial^\alpha u(x) = \sum_h c_h\,\sigma^{(|\alpha|)}(z_h)\prod_i (W_{hi}/s_i)^{\alpha_i}.
\]
One activation-tower evaluation per total order yields **all** partials of that
order. Everything else in this book — gradients, Laplacians, curvature, the de
Rham `d` — is a finite algebraic contraction of those exact numbers.

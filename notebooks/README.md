# omnibias notebook gallery

Runnable, visual walkthroughs of omnibias. Every notebook runs on **CPU in
under ~2 minutes** (most in seconds), uses fixed seeds, and ends with a clear
takeaway figure. They are the fastest way to understand what the package does
and why it matters.

| # | Notebook | What it shows | Packages |
|---|---|---|---|
| 01 | [`01_closed_form_derivatives.ipynb`](01_closed_form_derivatives.ipynb) | The core idea: closed-form `σ⁽ⁿ⁾` from the Riccati identity; accuracy + cost vs finite-difference/autograd | `core`, `torch` |
| 02 | [`02_operator_blocks.ipynb`](02_operator_blocks.ipynb) | Identity-nesting (Lemma 1) and the six typed operator blocks (identity / grad / laplacian / derivative / band / integral) | `torch` |
| 03 | [`03_pinn_closed_form_laplacian.ipynb`](03_pinn_closed_form_laplacian.ipynb) | 1D heat-equation PINN with a closed-form Laplacian (no autograd through layers) | `torch` |
| 04 | [`04_qpinn_schrodinger.ipynb`](04_qpinn_schrodinger.ipynb) | Quantum harmonic-oscillator ground state (TISE), energy vs the analytic `½` | `qpinn` |
| 05 | [`05_local_kinetic_energy.ipynb`](05_local_kinetic_energy.ipynb) | VMC/FermiNet local kinetic energy: closed-form Laplacian vs `jax.hessian`, scaling in dimension | `jax` |
| 06 | [`06_second_order_optimization.ipynb`](06_second_order_optimization.ipynb) | Closed-form parameter Hessian / Gauss–Newton Fisher / KFAC; a Newton step | `curvature`, `jax` |
| 07 | [`07_cmbnet_mnist.ipynb`](07_cmbnet_mnist.ipynb) | Operator-typed CNN (`CmbNet`) forward + a tiny training loop on synthetic data | `torch` |
| 08 | [`08_keras_unified_backend.ipynb`](08_keras_unified_backend.ipynb) | The *same* code on TensorFlow / JAX / PyTorch via Keras 3, with a parity check | `keras` |
| 09 | [`09_fields_quadrature_norms.ipynb`](09_fields_quadrature_norms.ipynb) | Definite integration, inner products and L²/Sobolev norms vs closed forms; quadrature vs Monte-Carlo | `fields` |
| 10 | [`10_geometry_sphere_laplace_beltrami.ipynb`](10_geometry_sphere_laplace_beltrami.ipynb) | Sphere metric → scalar curvature `2/R²`; `cos θ` as a Laplace–Beltrami eigenfunction | `geometry` |
| 11 | [`11_fractional_diffusion.ipynb`](11_fractional_diffusion.ipynb) | Grünwald–Letnikov `Dᵅxᵖ` convergence vs the Γ-ratio; spectral semigroup `Dᵅ Dᵅ = D²ᵅ` | `fractional` |
| 12 | [`12_score_ou_generator.ipynb`](12_score_ou_generator.ipynb) | Score, Itô generator and Fokker–Planck adjoint; the OU stationary density gives `ℒ*p∞ ≈ 0` | `score` |
| 13 | [`13_symbolic_neural_jet_discovery.ipynb`](13_symbolic_neural_jet_discovery.ipynb) | Library-free equation discovery: recover `dy=y`/`d2y=-y`/`dy=1-y²`, an AutoML surrogate, the heat-equation coefficient, and the Blasius identity | `symbolic` |
| 14 | [`14_synthetic_feature_discovery.ipynb`](14_synthetic_feature_discovery.ipynb) | Sparse closed-form feature discovery in many dimensions; interpretable model vs raw/dictionary baselines | `symbolic` (example) |
| 15 | [`15_battery_law_discovery.ipynb`](15_battery_law_discovery.ipynb) | Recover an interpretable capacity-fade law `dq/dn ∝ q` and roll it out into a monotone fade | `symbolic` (example) |
| 16 | [`16_cmapss_feature_discovery.ipynb`](16_cmapss_feature_discovery.ipynb) | Causal, leak-free operator features (derivatives, rolling) for turbofan RUL | `symbolic` (example) |
| 17 | [`17_financial_signal_discovery.ipynb`](17_financial_signal_discovery.ipynb) | Closed-form local Taylor jets (velocity/curvature) as limit-order-book calculus channels | `symbolic` (example) |
| 18 | [`18_joint_operator_regressor.ipynb`](18_joint_operator_regressor.ipynb) | One-loop differentiable operator gating + readout; recovers the true operators vs ridge/dictionary | `torch` |
| 19 | [`19_pullback_learned_manifolds.ipynb`](19_pullback_learned_manifolds.ipynb) | The pullback metric `g = JᵀhJ`: recover the round sphere from its embedding, then exact curvature of a *learned* (neural) chart | `geometry` |
| 20 | [`20_manifold_learning_omnibias.ipynb`](20_manifold_learning_omnibias.ipynb) | Laplacian eigenmaps approximate the exact `Δ_g`; a curvature-regularized autoencoder using the pullback metric | `geometry`, `fields` |
| 21 | [`21_spectral_shape_analysis.ipynb`](21_spectral_shape_analysis.ipynb) | Shape-DNA (LBO spectrum) + heat-kernel signature: intrinsic shape classification and spectral segmentation | `geometry` (concept) |
| 22 | [`22_equivariant_operator_features.ipynb`](22_equivariant_operator_features.ipynb) | Exact operators as symmetry features: E(3)-invariant metric/curvature, equivariant Jacobian; what's deferred (SE(3) irreps) | `geometry` |
| 23 | [`23_faa_di_bruno_jets.ipynb`](23_faa_di_bruno_jets.ipynb) | Exact multi-layer directional Taylor jets via Faà di Bruno: deep MLP towers vs autodiff/finite-difference, partition-count avoidance, deep Hessian by polarization | `core`, `jax` |

## Running

The packages are normal Python packages; install the ones a notebook needs
(see the table) and launch Jupyter from the repo root:

```bash
pip install -e packages/omnibias-core packages/omnibias-torch \
            packages/omnibias-jax packages/omnibias-curvature \
            "packages/omnibias-pinn[torch]" "packages/omnibias-qpinn[torch]" \
            packages/omnibias-keras \
            "packages/omnibias-fields[torch]" "packages/omnibias-geometry[torch]" \
            "packages/omnibias-fractional[torch]" "packages/omnibias-score[torch]" \
            matplotlib jupyter
jupyter lab notebooks/
```

Notebooks 09–12 (the field/geometry/fractional/score layer) build a tiny
analytic field via the shared `notebooks/_fields.py` helper, so they need only
the relevant package plus `matplotlib`.

Notebooks 13–18 (symbolic / equation discovery) use `omnibias-symbolic` (install
`"packages/omnibias-symbolic[all]"`); notebooks 14–18 also import the recovered
experiment drivers under `examples/symbolic_discovery/`, so they insert the repo
root on `sys.path` and should be run with `notebooks/` as the working directory
from a checkout. Notebooks 16–17 demonstrate the method on synthetic inputs; the
real-data runs (Severson, NASA C-MAPSS, FI-2010) are described in
`examples/symbolic_discovery/README.md`.

Notebooks 19–22 (learned manifolds / manifold learning / shape analysis /
equivariance) build on the `omnibias-geometry` pullback metric; install
`"packages/omnibias-geometry[torch]"` plus `scipy` and `matplotlib`. Notebook 20
also uses the `notebooks/_fields.py` helper. Notebook 21 is self-contained
(numpy/scipy) and illustrates the discrete face of the exact `laplace_beltrami`.

Notebook 23 (Faà di Bruno multi-layer jets) needs only `omnibias-core`,
`omnibias-jax` and `matplotlib`; it runs on CPU in float64.

For notebook 08, select the Keras backend before launching, e.g.
`KERAS_BACKEND=jax jupyter lab notebooks/`.

All notebooks import the shared `_style.py` in this folder for consistent
visuals, so run them with `notebooks/` as the working directory.

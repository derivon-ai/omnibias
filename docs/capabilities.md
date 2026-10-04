# What omnibias unlocks

**Differentiate deeper. Make decisions differentiable. Math that trains.**

omnibias removes the nested high-order spatial-autodiff bottleneck for supported
models. Activation identities, Taylor composition and direct operator
contractions replace repeated differentiation of a growing backward graph.
The resulting spatial derivatives retain parameter gradients, so a derivative
can participate in a trainable residual, regularizer or conditional model.

## From bottleneck to primitive

| Need | Mechanism and API | Evidence and scope |
| --- | --- | --- |
| High-order activation derivatives | Shared polynomial recurrences; `get_activation(...).fastpath` in the [backend APIs](api/torch.md) | [Activation benchmark](performance.md#direct-activation-derivatives): order eight avoids the nested graph; arithmetic still grows with polynomial degree |
| Derivatives through a deep network | `mlp_jet` for directions; `mlp_jet_mv` for mixed partials | [Executable examples](derivatives.md); [deep-MLP comparison](performance.md#general-deep-network-jets), including JAX Taylor-mode wins |
| A Laplacian without a full Hessian | `deep_field_laplacian`; one-layer `neural_field_polylaplacian` | [Operator measurements](performance.md); [5,000-dimensional training](derivatives.md#a-deep-laplacian-in-5000-dimensions). Deep repeated derivatives report exact support or an estimator |
| Learn conditional structure | Sigmoid gates, partition weights and regional-model composition | [Partition API](api/partition.md), trainable gates and experts in the README; finite-temperature smooth decisions |
| Train using parameter curvature | Residual Jacobians, Hessian-vector products and structured approximations | [Optimizer choices below](#curvature-aware-training); optimization behavior remains problem-dependent |
| Bound a supported quantity | Outward-rounded intervals, Taylor models and checked certificates | [Guarantees](guarantees.md); stated domain, model and assumptions remain part of the result |

The improvement is constructive: compute only the derivative or contraction
the model needs. Nested high-order graphs can become infeasible in time or
memory; these alternative representations make demonstrated workloads
practical. They do not establish that autodiff always has worse numerical
accuracy. The independent [precision sweep](performance.md#precision-is-measured-separately-from-speed)
also records input-dependent cancellation in closed-form polynomials.

## Two limits, two useful operations

For a sufficiently smooth activation, normalized weighted bias collapse gives

$$
\lim_{\delta\to0}\delta^{-n}
\sum_{j=0}^{n}(-1)^{n-j}\binom{n}{j}\sigma(z+j\delta)
=\sigma^{(n)}(z).
$$

The weights and normalization are essential: an unweighted sum of coalescing
activations is not an arbitrary derivative. The backend kernels evaluate the
analytic limit instead of subtracting nearly equal samples. Riccati identities
such as `tanh′ = 1 − tanh²` generate the derivative polynomials. Taylor jets
compose them through supported networks with coefficients `f⁽ᵏ⁾/k!` or
`Dᵅf/α!`; conversion helpers recover ordinary derivatives.

Temperature hardening is a different limit. With
`g(x) = sigmoid(β(w·x − t))`, the mixture
`(1 − g)f₀(x) + g f₁(x)` trains thresholds, directions and experts together.
Increasing `β` approaches hard routing away from ties. The external
**omnibias-tab** consumer supports both jointly trained soft-tree/neural
models and stagewise GBM-style Newton boosting. Those are separate procedures;
this does not differentiate an existing hard tree's discontinuous choices.

## Curvature-aware training

Spatial derivatives from jets remain tensor expressions. Parameter autodiff
can therefore form a residual Jacobian or Hessian-vector product without
reintroducing nested spatial differentiation. Choose curvature according to
the objective and available memory:

| Family | Current API | What it uses |
| --- | --- | --- |
| Least-squares curvature | `omnibias.torch.optim.GaussNewton`, `CubicGaussNewton` | Residual Jacobian and Gauss–Newton curvature; not the full loss Hessian |
| Newton and trust regions | `omnibias.torch.optim.CubicNewton`, `TrustRegionNewtonCG` | Hessian-vector products, cubic regularization or a trust-region solve |
| Structured approximations | `omnibias.torch.optim.NaturalGradient`, `DiagonalCurvature`, `FrugalCurvature`, `KFAC` | Fisher, diagonal, streamed or Kronecker-factor structure; distinct from exact full curvature |
| Closed-form parameter curvature | `omnibias.curvature.one_layer` | Algebraic parameter gradients/Hessians and Fisher factors for supported one-layer fields |
| Curvature regularization | `omnibias.curvature.torch` | Deep-model Hessian operators and sharpness functionals; exact and stochastic routes are separately named |

The functional `GaussNewton` driver takes a flat parameter vector and residual
function. Optimizer subclasses such as `CubicGaussNewton` have their own closure
contracts; inspect the [Torch API](api/torch.md) and implementation before
substituting one into a training loop. The [curvature package](api/curvature.md)
documents its backend and dependency boundary.

### Historical optimizer evidence

The pre-extraction 1-D Poisson benchmark used a one-layer tanh field, 16 hidden
units, 62 interior points, float64 CPU and five seeds. It recorded these
medians under method-specific step budgets:

| Method | Relative L2 | Wall time |
| --- | ---: | ---: |
| Adam, 800 steps | 8.52 × 10⁻⁴ | 0.385 s |
| L-BFGS, 60 steps | 2.19 × 10⁻⁵ | 0.399 s |
| Gauss–Newton, 40 steps | 4.05 × 10⁻⁵ | 0.080 s |
| Cubic Gauss–Newton, 40 steps | 1.25 × 10⁻⁴ | 0.706 s |
| Trust-region Newton-CG, 40 steps | 8.24 × 10⁻⁴ | 0.195 s |

This is historical evidence, not a rerun of the current split ecosystem or
an equal-wall-time comparison. L-BFGS attained the lowest median error here;
Gauss–Newton finished faster. The immutable [artifact](https://github.com/derivon-ai/omnibias/blob/53c00f742089b64017ecf199ccedf13908ea0a7a/docs/benchmarks/optimizer_pinn.json)
and [driver](https://github.com/derivon-ai/omnibias/blob/53c00f742089b64017ecf199ccedf13908ea0a7a/benchmarks/optimizer_pinn.py)
preserve the exact experiment. Curvature methods are an additional training
tool, not a universal replacement for tuned first-order optimizers.

## Four PINN integration routes

The external **omnibias-pinn** repository owns these application integrations.
Module paths below are relative to that consumer, not imports supplied by
the primitive workspace. The historical experiments and artifacts now live
in the separate **omnibias-research** checkout.

| Obstacle | Consumer route | Historical five-seed evidence and boundary |
| --- | --- | --- |
| Causality | `train.torch.march_solve` and JAX twin: gated time windows with handoff | Reaction-family marching median relative L2 ≈ 0.084 versus whole-interval ≈ 0.99; whole-interval won the heat case. [Artifact](https://github.com/derivon-ai/omnibias/blob/53c00f742089b64017ecf199ccedf13908ea0a7a/docs/benchmarks/causal_marching.json) |
| Curved boundaries | `domain.torch.DistanceConstrainedField` and JAX twin: boundary ansatz with a distance factor | Disk boundary error near 10⁻¹⁶; boundary satisfaction and interior accuracy are separate. Nonconvex interior accuracy remained open. [Artifact](https://github.com/derivon-ai/omnibias/blob/53c00f742089b64017ecf199ccedf13908ea0a7a/docs/benchmarks/geometry_sdf.json) |
| Parametric operators | `operator.torch` and JAX counterpart: conditioned DeepONet/FNO models | Conditioned median relative L2 0.0148 versus 0.0186 unconditioned and 0.0185 residual PINN on the tested family. [Artifact](https://github.com/derivon-ai/omnibias/blob/53c00f742089b64017ecf199ccedf13908ea0a7a/docs/benchmarks/operator_zero_shot.json) |
| Spectral bias | `torch.fields`: Fourier/multiscale features and multilevel FBPINN; separate one-shot readout least-squares experiment | Least-squares median relative L2 ≈ 5.2 × 10⁻⁹ through frequency 16; it used more features than the matched arm and ≈ 20× Adam's wall time. [Artifact](https://github.com/derivon-ai/omnibias/blob/53c00f742089b64017ecf199ccedf13908ea0a7a/docs/benchmarks/spectral_bias_fbpinn.json) |

These are achieved, workload-specific acceptance results, not universal claims
about all PDEs. The immutable [acceptance matrix](https://github.com/derivon-ai/omnibias/blob/53c00f742089b64017ecf199ccedf13908ea0a7a/docs/benchmarks/pinn_four_gap_matrix.md)
records both passed gates and remaining problems. The [PINN guide](pinn.md)
starts with an executable residual and explains when to reach for these routes.

# omnibias-curvature

Parameter curvature primitives.

`omnibias.curvature` supplies closed-form parameter derivatives for supported
model structures and curvature-based optimization utilities.

- `one_layer_param_grad`, `one_layer_param_hessian`: one-layer parameter derivatives.
- `mse_loss_hessian`, `mse_gauss_newton_fisher`: loss curvature constructions.
- `kfac_kron_factors`: factorized curvature information.
- `damped_solve`, `regularized_solve`: stabilized linear algebra.

Distinguish an exact Hessian from a Gauss–Newton or factorized approximation.
The one-layer formulas do not automatically cover arbitrary deep framework
modules. Validate conditioning and gradient behavior before using a curvature
step in a PINN training loop.

Install this distribution with `pip install omnibias-curvature`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a1** · Python **>=3.10** · **3 - Alpha** · AGPL-3.0-or-later **or commercial**

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-curvature/src/omnibias/curvature). Modules below are relative to `omnibias.curvature`; underscored modules are internal.

`glm_fisher`, `information`, `natural_gradient`, `one_layer`, `operators`, `regularize`, `sharpness`, `torch`, `torch.optim`, `torch.regularize`, `torch.sharpness`.

Exports from `omnibias.curvature`:

`CollapseResult`, `damped_solve`, `fisher_information_metric`, `glm_fisher`, `glm_loss_gradient`, `glm_natural_gradient_step`, `hessian_frobenius_sq`, `hessian_top_eigenvalue`, `hessian_trace`, `kfac_kron_factors`, `min_norm_solve`, `mse_curvature_sharpness`, `mse_gauss_newton_fisher`, `mse_loss`, `mse_loss_hessian`, `mse_newton_step`, `natural_gradient_step`, `numerical_rank`, `one_layer_param_grad`, `one_layer_param_hessian`, `rank_collapse`, `regularization_path`, `regularized_solve`, `sam_objective`, `sam_sharpness_gap`, `sharpness_aware_loss`.

</details>

<!-- END GENERATED API INVENTORY -->

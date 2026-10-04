# omnibias-jax

JAX derivatives and trainable operators.

- `omnibias.jax.activations.get_activation`: activation specifications.
- `omnibias.jax.jet`: directional network jets and Taylor conversions.
- `omnibias.jax.jet_mv`: full mixed jets and derivative extraction.

The [derivative guide](../derivatives.md) demonstrates `jax.grad` over a
fourth-order residual. Keep array operations traceable and derivative orders
static when compiling. Set `JAX_ENABLE_X64=1` before array creation when
comparing to float64 reference values. Shared coefficient definitions do not
remove compiler or device rounding differences.

Install this distribution with `pip install omnibias-jax`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.5.0rc1** · Python **>=3.10** · **4 - Beta** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-jax/src/omnibias/jax). Modules below are relative to `omnibias.jax`; underscored modules are internal.

`activations`, `activations_complex`, `architectures`, `architectures.attention`, `architectures.ftc_net`, `architectures.hardbc`, `architectures.integral_kernel`, `architectures.jetkan`, `architectures.multiscale`, `architectures.pinn`, `architectures.piratenet`, `bo_derivatives`, `confluence`, `confluent_bank`, `implicit`, `jet`, `jet_mv`, `laplacian`, `line_search`, `multipack`, `optim`, `optim_block_search`, `precision`, `realization`, `refine`, `scan`, `scan_equivariant`, `weight_loss_jet`.

Exports from `omnibias.jax`:

`AdaptivePackBank`, `BankSpec`, `BirkhoffOMBU`, `BlockSpec`, `DEQConfig`, `DEQNotContractive`, `DEQResult`, `DEQSolverUnknown`, `GradientSecant`, `JaxActivationSpec`, `JetLineSearchConfig`, `LineSearchResult`, `WeightLossJetSpec`, `X64_HINT`, `affine_jet`, `affine_jet_mv`, `antiderivative_jet`, `arrangement_w_block`, `bank_forward`, `bias_scan`, `block_direction`, `block_exact_search`, `block_exact_sweep`, `compose_jet`, `compose_jet_mv`, `compose_jet_riccati`, `coulomb_potential`, `deq_du_dW`, `deq_solve`, `deq_vjp`, `derivative_jet`, `equivariant_scan_apply`, `get_activation`, `identity_jet`, `init_bias_scan`, `init_multipack`, `init_pack_bank`, `is_registered`, `jet_attention`, `jet_exp`, `jet_gradient`, `jet_hessian`, `jet_line_search`, `jet_line_search_on_ray`, `jet_multiply`, `jet_partials`, `jet_reciprocal`, `jet_softmax`, `jet_to_tower`, `last_linear_block`, `layer_jet`, `layer_jet_mv`, `lhopital_ratio`, `limit_of_ratio`, `list_activations`, `make_bo_force`, `make_bo_hessian`, `make_local_energy`, `mlp_jet`, `mlp_jet_mv`, `multipack_apply`, `multipack_response`, `neural_field_hessian`, `neural_field_laplacian`, `neural_field_value`, `neural_field_value_and_laplacian`, `neural_field_value_grad_hessian`, `neural_field_value_grad_laplacian`, `ombu_bias_block`, `one_layer_loss`, `one_layer_loss_grad`, `one_layer_loss_hessian`, `one_layer_loss_jet`, `one_layer_newton_direction`, `refine`, `register_activation`, `removable_value`, `require_x64`, `scan_response`, `soft_argmax_offset`, `steerable_basis`, `tower_to_jet`, `vibrational_frequencies`, `x64_enabled`.

</details>

<!-- END GENERATED API INVENTORY -->

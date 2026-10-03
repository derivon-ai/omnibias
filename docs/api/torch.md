# omnibias-torch

PyTorch derivatives and trainable operators.

- `omnibias.torch.activations.get_activation`: obtain an activation specification.
- `omnibias.torch.jet`: `mlp_jet`, `layer_jet`, `compose_jet`, `jet_to_tower`.
- `omnibias.torch.jet_mv`: `mlp_jet_mv`, `jet_partials`, `jet_gradient`, `jet_hessian`.
- `omnibias.torch.blocks.OperatorBlock`: activation/operator layer.

The [PINN guide](../pinn.md) runs fourth-order spatial derivatives through a
training loss. [Mixed derivatives](../derivatives.md) use Taylor coefficients
with an explicit multi-index convention. Keep parameters live to preserve
gradients; new tensors follow the current framework dtype unless specified.

Install this distribution with `pip install omnibias-torch`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.4.0** · Python **>=3.10** · **4 - Beta** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-torch/src/omnibias/torch). Modules below are relative to `omnibias.torch`; underscored modules are internal.

`activations`, `activations.classical`, `activations.nqs`, `activations.piecewise`, `activations.proximal`, `activations.registry`, `activations.smooth`, `activations.tempered`, `activations.trigonometric`, `architectures`, `architectures.attention`, `architectures.ftc_net`, `architectures.hardbc`, `architectures.integral_kernel`, `architectures.jetkan`, `architectures.joint_operator`, `architectures.multiscale`, `architectures.pinn`, `architectures.piratenet`, `blocks`, `blocks.conv`, `blocks.linear`, `blocks.operator`, `confluence`, `confluent_bank`, `fastpath`, `fastpath.dispatch`, `fastpath.eulerian`, `fastpath.hermite`, `fastpath.legendre`, `growable`, `identity_init`, `implicit`, `jet`, `jet_mv`, `laplacian`, `line_search`, `multipack`, `optim`, `optim_block_search`, `realization`, `refine`, `scan`, `scan_equivariant`, `stencil`, `tempered_blocks`, `training`, `training.k_scheduler`, `unit`, `weight_loss_jet`.

Exports from `omnibias.torch`:

`ActivationSpec`, `AdaptivePackBank`, `AnalyticGaussianConv1d`, `AnalyticGaussianConv2d`, `BankSpec`, `BiasScan`, `BirkhoffOMBU`, `BlockSpec`, `DEQConfig`, `DEQNotContractive`, `DEQResult`, `DEQSolverUnknown`, `EquivariantScan`, `GradientSecant`, `GrowStrategy`, `GrowableOMBU`, `GrowableOperatorMultiBiasUnit`, `JetLineSearchConfig`, `LearnablePReLU`, `LineSearchResult`, `MultiPackUnit`, `OMBU`, `OperatorBlock`, `OperatorMultiBiasUnit`, `TemperedActivation`, `WeightLossJetSpec`, `affine_jet`, `affine_jet_mv`, `analytic_gaussian_taps`, `antiderivative_jet`, `arrangement_w_block`, `block_direction`, `block_exact_search`, `block_exact_sweep`, `cmbConv1d`, `cmbConv2d`, `cmbLinear`, `compose_jet`, `compose_jet_mv`, `compose_jet_riccati`, `deq_du_dW`, `deq_solve`, `deq_vjp`, `derivative_jet`, `get_activation`, `grow_ombu`, `identity_jet`, `is_registered`, `jet_attention`, `jet_exp`, `jet_gradient`, `jet_hessian`, `jet_line_search`, `jet_line_search_on_ray`, `jet_multiply`, `jet_partials`, `jet_reciprocal`, `jet_softmax`, `jet_to_tower`, `last_linear_block`, `layer_jet`, `layer_jet_mv`, `lhopital_ratio`, `limit_of_ratio`, `list_activations`, `mlp_jet`, `mlp_jet_mv`, `multipack_response`, `ombu_bias_block`, `one_layer_loss`, `one_layer_loss_grad`, `one_layer_loss_hessian`, `one_layer_loss_jet`, `one_layer_newton_direction`, `refine`, `register_activation`, `removable_value`, `scan_response`, `soft_argmax_offset`, `steerable_basis`, `tower_to_jet`.

</details>

<!-- END GENERATED API INVENTORY -->

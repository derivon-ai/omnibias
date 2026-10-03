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

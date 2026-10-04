---
name: omnibias-torch
description: Maintain PyTorch tensor kernels, trainable modules and optimizer integration in omnibias-torch.
---

# Maintaining omnibias-torch

Owned implementation: [source](../../../packages/omnibias-torch/src/omnibias/torch);
public surface: [API](../../../docs/api/torch.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`fastpath/` and `activations/` adapt activation formulas; `jet.py`, `jet_mv.py`
and `laplacian.py` own spatial operators. Read the corresponding JAX implementation
when changing a paired algorithm, but keep PyTorch module and optimizer behavior
here. The shared numerical reference owns cross-backend invariants.

For `nn.Module` changes, inspect registration of parameters and buffers, device
moves, state dictionaries and framework-default dtype behavior. For optimizer
changes, distinguish functional parameter-vector drivers from optimizer subclasses
whose closures recompute a loss or residual. `JetLBFGSOptimizer` and `JetLBFGS`
are different interfaces. Reusing an old autograd graph across closure calls can
break otherwise correct formulas.

Spatial derivatives must remain connected to trainable weights. Test the final
residual loss backward pass, not just a derivative tensor's numeric value.
`test_deep_laplacian.py` covers wide coordinate dimensions; `test_jet_mv.py` covers
mixed coefficients. Module and closure changes also need `test_nn_optim.py` or
`test_unit.py` as appropriate. Benchmark in the execution mode the documented
user actually runs, with thread policy recorded.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-torch/tests/test_jet.py packages/omnibias-torch/tests/test_deep_laplacian.py packages/omnibias-torch/tests/test_nn_optim.py -q
```

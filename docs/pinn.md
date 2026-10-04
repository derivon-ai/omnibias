# Build a PINN residual

This guide uses PyTorch and a scalar network on `0 <= x <= 1`. Its fourth-order
residual is `d^4u/dx^4 - 24`, with `u(0) = u'(0) = u(1) = u'(1) = 0`.
The manufactured solution is `x**2 * (1 - x)**2`. A jet computes the spatial
derivatives; ordinary parameter autodiff differentiates the training loss.
This removes the need to construct a nested fourth-order spatial-autodiff
graph while keeping the residual trainable.

Install `omnibias-torch`, then run this complete optimizer step:

```python
import torch
from omnibias.torch.jet import jet_to_tower, mlp_jet

generator = torch.Generator().manual_seed(7)
weights = torch.nn.Parameter(torch.randn(8, 1, generator=generator) * 0.5)
bias = torch.nn.Parameter(torch.randn(8, generator=generator) * 0.1)
readout = torch.nn.Parameter(torch.randn(1, 8, generator=generator) * 0.2)
offset = torch.nn.Parameter(torch.zeros(1))
parameters = [weights, bias, readout, offset]
optimizer = torch.optim.Adam(parameters, lr=1e-3)
layers = [(weights, bias, "tanh"), (readout, offset, None)]


def derivatives(x, order):
    point = x.reshape(1)
    jet = mlp_jet(point, torch.ones_like(point), layers, order=order)
    return jet_to_tower(jet)[:, 0]


points = torch.linspace(0.0, 1.0, 10)[1:-1]
optimizer.zero_grad()
fourth_derivatives = torch.stack([derivatives(x, 4)[4] for x in points])
residual_loss = (fourth_derivatives - 24.0).square().mean()
left = derivatives(torch.tensor(0.0), 1)
right = derivatives(torch.tensor(1.0), 1)
boundary_loss = left.square().sum() + right.square().sum()
loss = residual_loss + 10.0 * boundary_loss
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
optimizer.step()
assert torch.isfinite(loss)
```

`order=4` returns value and four derivatives. The spatial input does not need
`requires_grad=True`; the trainable parameters do. `spec=None` makes the last
layer affine. Keep parameter tensors live in `layers` instead of copying or
detaching them.

This executes one training step, not a convergence demonstration. A training
loop should monitor the residual on independent points and compare against
the manufactured solution. Boundary weighting, network capacity and sampling
remain application decisions. The explicit point loop makes the derivative
contract visible; profile batching choices for production workloads.

For multiple coordinates and mixed partials, see the
[derivative guide](derivatives.md). For field state, cached activation towers
and named operators, see [fields](api/fields.md). The higher-level solver is
a separate consumer repository at `../omnibias_projects/omnibias-pinn/`.

## Choose the next training ingredient

Start with the derivative the residual actually needs. A directional jet
avoids unnecessary mixed partials; a direct Laplacian avoids constructing a
full Hessian. The [derivative guide](derivatives.md) includes a deep Laplacian
with parameter gradients in 5,000 dimensions.

For small and medium smooth objectives, inspect
`omnibias.torch.optim` for Gauss–Newton, cubic regularization, trust-region
Newton-CG and structured curvature methods. Distinguish residual-based
Gauss–Newton from a full loss Hessian, and follow the chosen optimizer's
closure contract. Compare held-out solution error and wall time, not step
count alone. The [capability guide](capabilities.md#curvature-aware-training)
maps the families and preserves historical optimizer results.

For application-level difficulties, the external **omnibias-pinn** consumer
provides the following module families; these are not part of this workspace:

| Need | Integration route |
| --- | --- |
| Respect causal ordering in a time-dependent problem | `train`: gated marching windows with explicit handoff |
| Enforce supported curved Dirichlet boundaries | `domain`: distance-based constrained fields |
| Learn a family of parameterized operators | `operator`: conditioned DeepONet and FNO models |
| Represent and learn higher-frequency structure | `torch.fields` / `jax.fields`: Fourier, multiscale and FBPINN fields; research experiments also compare one-shot least-squares readouts |

Soft regional models are another option: `omnibias.partition` supplies
trainable gates and a partition of unity, while the consumer supplies the
PDE-specific coupling. Smooth routing preserves gradients into experts and
gates; it does not automatically enforce interface conditions.

The [four-route evidence](capabilities.md#four-pinn-integration-routes) states
what the historical acceptance tests demonstrated and what remains
problem-dependent. Revalidate the extracted consumers on your workload;
derivatives, optimization, boundary satisfaction and solution accuracy are
separate things to measure.

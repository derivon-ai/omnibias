# Pullback metric: geometry on learned manifolds

The **pullback metric** turns any immersion (chart) \(\varphi:\mathbb{R}^d\to
\mathbb{R}^n\) into a Riemannian metric on its domain,

\[
    g = J^\top h\, J, \qquad J = \frac{\partial\varphi}{\partial x},
\]

with \(h\) the ambient metric (Euclidean by default). The Jacobian \(J\) is taken
by forward-mode autodiff, so \(g\) is exact for analytic **and** neural-network
charts. Because every connection / curvature / field operator only reads
`manifold.metric.g_point`, wrapping a chart into a `MetricSpec` makes the entire
stack (Christoffel, Riemann / Ricci / scalar curvature, Laplace-Beltrami,
geodesics) work on learned manifolds with no further changes.

See the runnable notebook
[`19_pullback_learned_manifolds.ipynb`](https://github.com/derivon-ai/omnibias/blob/main/notebooks/19_pullback_learned_manifolds.ipynb).

## Recover the round sphere from its embedding

```python
import torch
from omnibias.geometry import ChartSpec, ManifoldSpec
from omnibias.geometry.torch import ops as geo

def sphere_phi(x):                       # x = (theta, phi)
    th, ph = x[0], x[1]
    return torch.stack([torch.sin(th) * torch.cos(ph),
                        torch.sin(th) * torch.sin(ph),
                        torch.cos(th)])

chart = ChartSpec(phi=sphere_phi, domain_dim=2, ambient_dim=3, name="S2")
manifold = ManifoldSpec("S2", 2, geo.metric_spec_from_chart(chart))

coords = torch.tensor([[0.7, 0.3], [1.1, 1.5]], dtype=torch.float64)
geo.pullback_metric(coords, chart)        # -> diag(1, sin^2 theta)
geo.scalar_curvature(coords, manifold)    # -> 2  (the round sphere)
```

## A learned (neural) chart

`phi` can be any backend-differentiable callable, including a neural network:

```python
# `.double()` because the curvature stack runs in float64; the chart and the
# coordinates must agree on dtype.
net = torch.nn.Sequential(
    torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3),
).double()
chart = ChartSpec(phi=net, domain_dim=2, ambient_dim=3, name="mlp")
manifold = ManifoldSpec("learned", 2, geo.metric_spec_from_chart(chart))

z = torch.randn(64, 2, dtype=torch.float64)
geo.pullback_metric(z, chart)             # SPD metric at every latent point
geo.scalar_curvature(z, manifold)         # exact curvature of the learned surface
```

## Non-Euclidean ambient metric

Pass `ambient_metric=h` (a callable `y -> (n, n)`) to pull back a non-Euclidean
ambient metric; the default `None` means the Euclidean identity, i.e.
\(g = J^\top J\).

## JAX

The JAX backend (`omnibias.geometry.jax.ops`) is the bit-identical twin
(`metric_spec_from_chart`, `pullback_metric`); cross-backend tests assert
agreement to `rtol=1e-9` in float64.

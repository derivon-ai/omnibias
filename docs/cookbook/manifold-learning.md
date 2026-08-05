# Manifold learning, powered by omnibias

<!-- docs-test: file-skip reason="excerpts from notebook 20, which supplies sphere_phi / make_field / the autoencoder" -->

omnibias does not re-implement scikit-learn; it supplies the *exact differential
geometry* that classical manifold-learning methods approximate. This recipe shows
two connections; the full walkthrough is in
[`20_manifold_learning_omnibias.ipynb`](https://github.com/derivon-ai/omnibias/blob/main/notebooks/20_manifold_learning_omnibias.ipynb).

## Laplacian eigenmaps approximate the Laplace-Beltrami operator

Diffusion maps and Laplacian eigenmaps build a discrete graph Laplacian whose
spectrum converges to the continuous \(\Delta_g\). omnibias computes that operator
exactly. On the unit sphere, \(\cos\theta\) is a degree-1 eigenfunction:

\[
    \Delta_g \cos\theta = -2\cos\theta.
\]

```python
import torch
from omnibias.geometry import ChartSpec, ManifoldSpec
from omnibias.geometry.torch import ops as geo
from omnibias.fields.torch import _ops_dispatch as dispatch
# (see notebooks/_fields.py for make_field / Cos / Const)

sphere = ManifoldSpec("S2", 2, geo.metric_spec_from_chart(
    ChartSpec(phi=sphere_phi, domain_dim=2, ambient_dim=3, name="S2")))
lap = geo.laplace_beltrami(field(coords), "f", sphere)   # ~ -2 cos(theta)
```

## Curvature-regularized autoencoder (uses the pullback metric)

A decoder `D: R^2 -> R^3` is a chart, so `pullback_metric(z, chart)` measures the
geometry of the learned latent manifold. Penalizing the metric distortion
\(\lVert g - I\rVert^2\) drives the chart toward a low-distortion (lower-curvature)
embedding, and `scalar_curvature` reports the exact curvature before/after.

```python
chart = ChartSpec(phi=decoder, domain_dim=2, ambient_dim=3, name="dec")
z = encoder(data)
recon = ((decoder(z) - data) ** 2).mean()
g = geo.pullback_metric(z, chart)
loss = recon + lam * ((g - torch.eye(2)) ** 2).mean()   # differentiable geom. reg.
loss.backward()
```

## Takeaway

Discrete Laplacians approximate the exact \(\Delta_g\); the pullback metric turns
"geometry of the latent space" into a differentiable signal. omnibias is the
foundation these methods sit on, a stronger story than re-implementing them.

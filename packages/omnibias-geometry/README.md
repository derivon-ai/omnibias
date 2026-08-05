# omnibias-geometry

**Status: Beta (v0.2.0).**

Differential geometry on manifolds, built on the `omnibias-fields` substrate with
cross-backend (PyTorch + JAX) parity:

- metric tensor `g_ij`, inverse metric `g^{ij}`, volume element `sqrt|g|`
- Christoffel symbols `Gamma^k_ij`
- covariant derivative `nabla` (scalar / vector / one-form)
- the **Laplace-Beltrami operator** `Delta_g f`
- Riemann / Ricci / scalar curvature
- the geodesic equation RHS
- exterior calculus: `d`, wedge, Hodge star, codifferential (see Phase 4)
- the **pullback metric** `g = JᵀhJ` of a learned chart `phi: R^d -> R^n`
  (analytic or neural), turning curvature / Laplace-Beltrami into tools for
  *learned* manifolds

## Install

```bash
pip install "omnibias-geometry[torch]"   # or [jax], or [all]
```

## Define a manifold

A metric is given by a *per-point* callable `g_point(x): (d,) -> (d, d)` written
with backend ops (so it is `vmap`/autodiff compatible):

```python
import torch
from omnibias.geometry import ManifoldSpec, MetricSpec
from omnibias.geometry.torch import ops as geo

R = 1.0
def sphere_g(x):                      # x = (theta, phi)
    theta = x[0]
    z = 0.0 * theta
    return torch.stack([
        torch.stack([R**2 + z, z]),
        torch.stack([z, R**2 * torch.sin(theta) ** 2]),
    ])

sphere = ManifoldSpec("S2", 2, MetricSpec(sphere_g, dim=2))
coords = torch.tensor([[1.0, 0.5], [2.0, 3.0]], dtype=torch.float64)

geo.scalar_curvature(coords, sphere)  # ~ 2 / R^2
geo.christoffel(coords, sphere)       # (B, k, i, j)
```

`laplace_beltrami` and `covariant_derivative` additionally take a
`FieldState` (any `omnibias-fields` field):

```python
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField

field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("theta", "phi")),
    components=ComponentSpec(("f", "vx", "vy")), hidden=16, base="tanh",
)
state = field(coords)

geo.laplace_beltrami(state, "f", sphere)              # (B,)
geo.covariant_derivative(state, ("vx", "vy"), sphere, kind="vector")
```

## Learned manifolds (pullback metric)

Hand a chart `phi: R^d -> R^n` to a `ChartSpec`; the induced metric `g = JᵀhJ`
(with `J` by autodiff) plugs straight into every operator above:

```python
from omnibias.geometry import ChartSpec
from omnibias.geometry.torch import ops as geo

def phi(x):                       # the standard S^2 embedding, x = (theta, phi)
    th, ph = x[0], x[1]
    return torch.stack([torch.sin(th)*torch.cos(ph),
                        torch.sin(th)*torch.sin(ph), torch.cos(th)])

chart = ChartSpec(phi=phi, domain_dim=2, ambient_dim=3, name="S2")
manifold = ManifoldSpec("S2", 2, geo.metric_spec_from_chart(chart))
geo.scalar_curvature(coords, manifold)   # ~ 2  (recovers the round sphere)
geo.pullback_metric(coords, chart)       # (B, 2, 2) = diag(1, sin^2 theta)
```

`phi` can equally be a neural network: omnibias then supplies exact curvature and
Laplace-Beltrami on the *learned* manifold.

## Two exact mechanisms, one consistent stack

`omnibias-geometry` is **exact** end-to-end. It uses two equally-exact
mechanisms, depending on what is being differentiated:

| Quantity | Mechanism | Why |
|---|---|---|
| **Field-function derivatives** (`grad f`, `hess f`, `Δ_g f` applied to the field) | **Closed-form sigma-tower** — one forward pass at every order, bit-identical across torch / JAX, scales as `O(1)` in input dim. | The field is a Riccati-activation neural field; `omnibias-fields` exposes the closed-form recurrence. |
| **Metric derivatives** inside Christoffel symbols, Riemann / Ricci / scalar curvature | **Forward-mode autodiff of the analytic per-point metric `g(x)`** — exact to machine precision (not a finite-difference approximation). | A user-supplied metric `g_point(x)` is an arbitrary analytic Python function; the right tool is exact forward-mode AD, not a sigma-tower recurrence. |

Both paths produce identical results to a sympy symbolic reference (within
float64 ULPs) on the analytic round sphere — `scalar curvature = 2/R²`,
`Ricci = g/R²`, `Δ_{S²} cos(θ) = -2 cos(θ)` — and to torch ↔ jax parity at
`rtol = 1e-9`. See [`GEOMETRY_DERIVATIONS.md`](GEOMETRY_DERIVATIONS.md) for
the index conventions and the per-op derivation, and
[`docs/scope-and-guarantees.md`](../../docs/scope-and-guarantees.md) for the
project-wide definition of "closed-form" vs "autodiff-exact".

## Validation

Every operator is checked against (1) the analytic round sphere
(`scalar curvature = 2/R^2`, `Ricci = g/R^2`, eigenfunction
`Delta_{S^2} cos(theta) = -2 cos(theta)`), (2) a sympy symbolic computation, and
(3) torch/jax cross-backend parity, all in float64.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.

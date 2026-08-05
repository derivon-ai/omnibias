# Differential geometry on the sphere

This recipe builds the round 2-sphere \(S^2\) as a `ManifoldSpec`, checks its
curvature, and applies the **Laplace-Beltrami operator** to a spherical-harmonic
eigenfunction. Everything runs on CPU in float64.

## The metric

The round sphere of radius \(R\) in polar coordinates \((\theta, \varphi)\) has
the metric \(g = \mathrm{diag}(R^2,\ R^2\sin^2\theta)\). A `MetricSpec` is given
by a *per-point* callable (written with backend ops so it is `vmap`/autodiff
compatible):

```python
import torch
from omnibias.geometry import ManifoldSpec, MetricSpec
from omnibias.geometry.torch import ops as geo

R = 1.0

def sphere_g(x):                      # x = (theta, phi), shape (2,)
    theta = x[0]
    z = 0.0 * theta
    return torch.stack([
        torch.stack([R**2 + z, z]),
        torch.stack([z, R**2 * torch.sin(theta) ** 2]),
    ])

sphere = ManifoldSpec("S2", 2, MetricSpec(sphere_g, dim=2))
coords = torch.tensor([[1.0, 0.5], [2.0, 3.0]], dtype=torch.float64)
```

## Curvature

```python
geo.scalar_curvature(coords, sphere)      # -> 2 / R^2 at every point
geo.ricci_tensor(coords, sphere)          # -> g / R^2
geo.christoffel(coords, sphere)           # (B, k, i, j)
```

The metric derivatives needed for the Christoffel symbols are obtained by exact
forward-mode autodiff of `sphere_g` (exact for an analytic metric -- see the
honesty note in the package docs).

## Laplace-Beltrami of an eigenfunction

For any `omnibias-fields` field whose scalar component is \(f(\theta,\varphi) =
\cos\theta\), the Laplace-Beltrami operator returns the \(l=1\) eigenvalue:

\[
    \Delta_{S^2}\cos\theta = -\frac{l(l+1)}{R^2}\cos\theta
                           = -\frac{2}{R^2}\cos\theta.
\]

<!-- docs-test: skip reason="call sketch on the reader's own FieldState; the runnable eigenvalue check is the analytic_field_jet recipe in the handbook" -->
```python
lb = geo.laplace_beltrami(state, "f", sphere)   # ~ -2/R^2 * cos(theta)
```

`state` here is whatever `FieldState` your field produces at `coords`; for a
self-contained run of the same identity see
[`handbook/03-differential-geometry.md`](../handbook/03-differential-geometry.md),
which builds the field with `analytic_field_jet`.

On the flat Euclidean metric `laplace_beltrami` reduces exactly to the ordinary
closed-form Laplacian, and the de Rham identity `delta d f = -Delta_g f` holds
(see `omnibias.geometry.torch.ops.hodge_laplacian_scalar`).

## General relativity: Schwarzschild curvature invariants

The same curvature machinery powers the general-relativity layer. Build the
Schwarzschild metric in \((t, r, \theta, \varphi)\) coordinates and read off the
Einstein tensor (zero in vacuum) and the Kretschmann invariant (which stays
finite and equals \(48 M^2/r^6\) -- the quantity that distinguishes a real
curvature singularity from a coordinate one):

```python
import torch
from omnibias.geometry import ManifoldSpec, MetricSpec
from omnibias.geometry.torch import ops as geo

M = 1.0

def schwarzschild_g(x):                 # x = (t, r, theta, phi)
    r, th = x[1], x[2]
    f = 1.0 - 2.0 * M / r
    z = 0.0 * r
    diag = [-f, 1.0 / f, r**2, r**2 * torch.sin(th) ** 2]
    return torch.stack([
        torch.stack([diag[i] if i == j else z for j in range(4)])
        for i in range(4)
    ])

schw = ManifoldSpec("schwarzschild", 4,
                    MetricSpec(schwarzschild_g, dim=4, signature=(-1, 1, 1, 1)))
coords = torch.tensor([[0.0, 6.0, 1.1, 0.5]], dtype=torch.float64)

geo.einstein_tensor(coords, schw)                 # ~ 0 (vacuum solution)
geo.einstein_equation_residual(coords, schw)      # ~ 0 (T = 0, Lambda = 0)
geo.kretschmann_scalar(coords, schw)              # -> 48 M^2 / r^6, finite
geo.weyl_tensor(coords, schw)                     # equals lowered Riemann here
```

For a cosmological (de Sitter FRW) metric \(g = \mathrm{diag}(-1, a^2, a^2, a^2)\)
with \(a(t) = e^{Ht}\), the Friedmann component is \(G_{00} = 3H^2\) and
`einstein_equation_residual(..., cosmological_constant=3*H**2)` vanishes. The
`weyl_tensor` is the conformally-invariant, totally trace-free part of curvature
and vanishes on maximally-symmetric spaces (a round `S^3`, de Sitter). All GR ops
use the autodiff-exact metric path -- no numerical-relativity time evolution is
implied.

## JAX

The JAX backend (`omnibias.geometry.jax.ops`) is the bit-identical twin; the
cross-backend tests assert agreement to `rtol=1e-9` in float64.

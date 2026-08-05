# PINN: advection-diffusion

The advection-diffusion equation transports a scalar `c` by a velocity field
while it diffuses:

\[
\partial_t c + (\mathbf{u}\cdot\nabla)c - \nabla\cdot(D\,\nabla c) = s .
\]

`omnibias-fields` ships this residual as a single **closed-form** operator,
`advection_diffusion_residual`, composed from the exact `material_derivative`
and `variable_coefficient_diffusion` primitives. The torch and JAX
implementations are bit-identical twins.

## Build a typed field

```python
import torch
from omnibias.pinn import CoordinateSpec, ComponentSpec
from omnibias.pinn.torch.fields import OneLayerVectorField

field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("x", "y", "t")),     # time axis = "t"
    components=ComponentSpec(("c", "u", "v"), groups={"vel": ("u", "v")}),
    hidden=64,
    base="tanh",
)
coords = torch.rand(256, 3, dtype=torch.float64)
state = field(coords)
```

## The residual

```python
res = state.ops.advection_diffusion_residual(
    state, scalar="c", velocity=("u", "v"), diffusivity=0.01, source=None,
)
loss = (res ** 2).mean()        # PINN PDE loss term
```

The constitutive **flux** `F = -D grad c` and its divergence are available
directly, so you can also impose the conservation form `d_t c + div F = s`:

```python
flux = state.ops.diffusive_flux(state, "c", diffusivity=0.01)   # (B, 2)
res_conservation = state.ops.conservation_residual(
    state, density="c", flux=("u", "v"),       # any named flux components
)
```

A spatially varying diffusivity is supported by naming a scalar component
(`diffusivity="Dfield"`) instead of passing a constant.

## Discoverability

```python
from omnibias.fields import list_operators
[op.name for op in list_operators(domain="conservation")]
```

All operators are closed-form (`op.closed_form is True`) and validated against
analytic references plus torch<->jax bit-parity at `rtol = atol = 1e-12`.

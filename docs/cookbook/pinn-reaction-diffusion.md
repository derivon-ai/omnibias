# PINN: reaction-diffusion & Poisson-Nernst-Planck

Reaction-diffusion couples diffusion to a pointwise reaction term:

\[
\partial_t c - \nabla\cdot(D\,\nabla c) - R(c) - s = 0 .
\]

`omnibias-fields` ships `reaction_diffusion_residual` with the reaction supplied
as a Python callable (e.g. Fisher-KPP `c(1-c)`), a named field component, or a
constant -- all on the **closed-form** path.

## Fisher-KPP

```python
import torch
from omnibias.pinn import CoordinateSpec, ComponentSpec
from omnibias.pinn.torch.fields import OneLayerVectorField

field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("x", "y", "t")),
    components=ComponentSpec(("c",)),
    hidden=64, base="tanh",
)
state = field(torch.rand(256, 3, dtype=torch.float64))

res = state.ops.reaction_diffusion_residual(
    state, scalar="c", diffusivity=0.1, reaction=lambda c: c * (1.0 - c),
)
# or with the attribute DSL:
res = state.c.reaction_diffusion(diffusivity=0.1, reaction=lambda c: c * (1.0 - c))
```

## Poisson-Nernst-Planck (electro-diffusion)

The PNP system couples charged-species transport to electrostatics. omnibias
provides each piece as a first-class operator, so PNP is literally the sum of
two residuals -- no autodiff anywhere:

```python
field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("x", "y", "t")),
    components=ComponentSpec(("c", "phi")),    # concentration + potential
    hidden=64, base="tanh",
)
state = field(torch.rand(256, 3, dtype=torch.float64))

# species continuity: d_t c + div(J_NP) = 0, J_NP = -D grad c - zF mu c grad phi
species = state.ops.nernst_planck_residual(
    state, concentration="c", potential="phi",
    diffusivity=1.0, valence=1.0, mobility=1.0, faraday=1.0,
)
# electrostatics: div(eps grad phi) + rho = 0  (here rho ~ z c)
poisson = state.ops.poisson_residual(state, "phi", source="c", permittivity=1.0)

pnp_loss = (species ** 2).mean() + (poisson ** 2).mean()
```

The bare **Nernst-Planck flux** `J = -D grad c - zF mu c grad phi` is also
available as `state.ops.nernst_planck_flux(state, "c", "phi", ...)`, and Fick's
law as `state.c.fickian_flux(diffusivity=...)`.

See the full [chemistry domain](../operators.md#chemistry-transport) in the
operator catalog.

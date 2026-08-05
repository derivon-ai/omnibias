# PINN: Maxwell's equations

`omnibias-fields` ships the full 3-D Maxwell system (natural units
`c = epsilon_0 = mu_0 = 1`) as **closed-form** residual operators -- pure
compositions of `curl`, `divergence`, and time `derivative` -- with bit-identical
torch and JAX twins.

\[
\partial_t \mathbf{B} + \nabla\times\mathbf{E} = 0,\quad
\partial_t \mathbf{E} - \nabla\times\mathbf{B} + \mathbf{J} = 0,\quad
\nabla\cdot\mathbf{E} = \rho,\quad
\nabla\cdot\mathbf{B} = 0 .
\]

## Field on spacetime

```python
import torch
from omnibias.pinn import CoordinateSpec, ComponentSpec
from omnibias.pinn.torch.fields import OneLayerVectorField

field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("x", "y", "z", "t")),
    components=ComponentSpec(
        ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "rho"),
        groups={"E": ("Ex", "Ey", "Ez"), "B": ("Bx", "By", "Bz")},
    ),
    hidden=64, base="tanh",
)
state = field(torch.rand(256, 4, dtype=torch.float64))
E = ("Ex", "Ey", "Ez")
B = ("Bx", "By", "Bz")
```

## The four laws as residuals

```python
faraday = state.ops.faraday_residual(state, electric=E, magnetic=B)         # (B, 3)
ampere  = state.ops.ampere_residual(state, electric=E, magnetic=B)          # (B, 3)
gauss_e = state.ops.gauss_residual(state, electric=E, charge="rho")         # (B,)
gauss_b = state.ops.gauss_magnetic_residual(state, magnetic=B)              # (B,)

maxwell_loss = sum(
    (r ** 2).mean() for r in (faraday, ampere, gauss_e, gauss_b)
)

# Energy flux:
S = state.ops.poynting_vector(state, electric=E, magnetic=B)                # S = E x B
```

## Potential formulation

Parameterising the fields by potentials enforces `div B = 0` and the Faraday law
**by construction** (`B = curl A`, `E = -grad phi - d_t A`):

```python
field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("x", "y", "z", "t")),
    components=ComponentSpec(
        ("phi", "Ax", "Ay", "Az"), groups={"A": ("Ax", "Ay", "Az")},
    ),
    hidden=64, base="tanh",
)
state = field(torch.rand(256, 4, dtype=torch.float64))

B = state.ops.magnetic_field_from_potential(state, potential=("Ax", "Ay", "Az"))
E = state.ops.electric_field_from_potentials(
    state, scalar_potential="phi", vector_potential=("Ax", "Ay", "Az"),
)
# Lorenz gauge condition, and the wave equation box A = -J:
gauge = state.ops.lorenz_gauge_residual(
    state, scalar_potential="phi", vector_potential=("Ax", "Ay", "Az"),
)
box_A = state.A.dalembertian(c=1.0)        # (B, 3), == -J in Lorenz gauge
```

The package's test-suite verifies the structural identities `div(curl A) = 0`,
`curl(grad phi) = 0`, and that Faraday is automatically satisfied by the
potentials. See the
[electromagnetism domain](../operators.md#electromagnetism-3-d-maxwell-natural-units)
in the operator catalog.

# PINN: linear elasticity

Static isotropic linear elasticity is the Navier-Cauchy equation for a
displacement field `u`:

\[
(\lambda+\mu)\,\nabla(\nabla\cdot \mathbf{u}) + \mu\,\Delta \mathbf{u} + \mathbf{f} = 0 ,
\]

equivalently the divergence of the Hooke stress
`sigma = lambda tr(eps) I + 2 mu eps`. `omnibias-fields` provides both the
**stress builder** and the **closed-form momentum residual**, with matching
torch and JAX twins.

## Build a (steady) displacement field

```python
import torch
from omnibias.pinn import CoordinateSpec, ComponentSpec
from omnibias.pinn.torch.fields import OneLayerVectorField

field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("x", "y", "z"), time_axis=None),   # steady
    components=ComponentSpec(("u", "v", "w"), groups={"disp": ("u", "v", "w")}),
    hidden=64, base="tanh",
)
state = field(torch.rand(256, 3, dtype=torch.float64))
```

## Stress and the equilibrium residual

```python
lam, mu = 1.2, 0.8

# Cauchy stress sigma = lam tr(eps) I + 2 mu eps  -> shape (B, 3, 3)
sigma = state.disp.elastic_stress(lam=lam, mu=mu)

# Navier-Cauchy equilibrium residual (with optional body force f):
res = state.disp.navier_cauchy_residual(lam=lam, mu=mu)            # (B, 3)

# It is exactly the divergence of the Hooke stress:
res_via_stress = state.ops.navier_cauchy_residual(
    state, displacement=("u", "v", "w"), lam=lam, mu=mu,
)
```

The strain rate / spin decomposition and the viscous-dissipation contraction are
also first-class:

```python
eps = state.disp.strain_rate                      # 0.5 (J + J^T)
W   = state.disp.rate_of_rotation                 # 0.5 (J - J^T)
phi = state.disp.viscous_dissipation(viscosity=mu)   # 2 mu eps:eps  (>= 0)
```

For an already-assembled stress tensor (named components `sigma_ij`) the
divergence is `state.ops.stress_divergence(state, ((..,..),(..,..)))`, a thin
alias of `tensor_divergence`.

## Finite strain: hyperelasticity

For large deformations the same substrate carries the finite-strain kinematics
and hyperelastic stress. The deformation gradient is `F = I + ∇u`; the stress is
the exact gradient of a stored energy `W(F)` (a machine-precision autodiff of the
algebraic energy):

```python
from omnibias.fields.torch import ops

F = ops.deformation_gradient_finite(state, ("u", "v", "w"))   # (B, 3, 3)
E = ops.green_lagrange_strain(F)          # 1/2 (F^T F - I); zero for rigid motion
J = ops.jacobian_det(F)                   # det F (= 1 if incompressible)

energy = lambda F: ops.neo_hookean_energy(F, lam=lam, mu=mu)
P     = ops.pk1_stress(F, energy)         # first Piola-Kirchhoff  P = dW/dF
S     = ops.pk2_stress(F, energy)         # second PK  S = F^-1 P  (symmetric)
sigma = ops.cauchy_stress(F, energy)      # true stress  sigma = J^-1 P F^T

# equilibrium residual Div(P) + f, and its small-strain limit is Navier-Cauchy:
res = ops.finite_strain_residual(state, ("u", "v", "w"), energy)          # (B, 3)
# transient elastodynamics rho u_tt - Div(P) - f (needs a time axis):
# res_dyn = ops.elastodynamic_residual(state_t, ("u","v","w"), energy, density=rho)
```

`st_venant_kirchhoff_energy`, `neo_hookean_energy`, and (3-D) `mooney_rivlin_energy`
are provided, along with hand-derived closed-form stresses
(`st_venant_kirchhoff_pk2`, `neo_hookean_pk2`) that the autodiff path is tested
against, and the anisotropic `hooke_stress_general(strain, C_ijkl)`. Elasticity,
hyperelasticity and elastodynamics are exact; history-dependent plasticity /
viscoelasticity return-map solvers are iterative (`numerical`) and out of the
closed-form scope.

See the [mechanics domain](../operators.md#continuum-mechanics) in the operator
catalog.

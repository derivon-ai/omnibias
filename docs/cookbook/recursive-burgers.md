# Finite 1-D recursive / guided-C∞ Burgers

Periodic 1-D viscous Burgers and heat on a torus, not Clay Navier–Stokes.
Cole–Hopf factorial jets linearize the plane-wave identity exactly. Frozen-advection
Picard drops a defect. A C∞ Fourier hunt can seal `Phi @ c == target` over
`Fraction`. That Q-lift is a reconstruction identity, not a PDE proof.

OWNS one-way filters, Adomian polynomials, and 3-D Cauchy–Kovalevskaya stay
unimplemented. `navier_stokes_proof_claim` stays false. The default hunt target
is the constant Cole–Hopf field `u = 2ν`, whose Burgers residual is identically
zero — a toy identity, not a general shock solver.

## Cole–Hopf factorial jets

```python
from omnibias.pinn.solver import cole_hopf_exact_burgers_demo

demo = cole_hopf_exact_burgers_demo(nu=0.05, k=-0.8, order=12)
assert demo["check"]["passed"] is True
assert demo["honesty"]["navier_stokes_proof_claim"] is False
```

## Picard residual drop

```python
import numpy as np
from omnibias.pinn.solver import picard_frozen_advection

n = 48
x = np.linspace(0.0, 1.0, n, endpoint=False)
dx = float(x[1] - x[0])
u0 = 0.3 * np.sin(2.0 * np.pi * x)
report = picard_frozen_advection(u0, viscosity=0.08, dx=dx, iters=25, tol=1e-9)
assert report.residual_history[0] > report.final_residual
assert report.honesty["navier_stokes_proof_claim"] is False
```

## Constant-mode Q reconstruction seal

```python
from omnibias.pinn.solver import guided_cinf_burgers_hunt

hunt = guided_cinf_burgers_hunt(
    n_grid=32,
    n_modes=3,
    k_terms=1,
    viscosity=0.05,
    outer_iters=3,
    use_picard=False,
)
assert hunt.q_reconstruction_seal is True
assert hunt.burgers_residual == 0.0
assert hunt.honesty["navier_stokes_proof_claim"] is False
assert hunt.honesty["q_seal_is_not_pde_identity"] is True
```

## Honesty

- Not a continuum Navier–Stokes regularity claim.
- The Q seal is `Phi @ c == target` over `Fraction`, not a Burgers identity.
- Combinatorics is optional. Without it the hunt takes the top-`k` `|lstsq|`
  coefficients; the span stays C∞.

## See also

- API: [`omnibias.pinn.solver`](../api/pinn-solver.md)
- Linearizing transforms: [`transforms_pde`](../api/transforms_pde.md)

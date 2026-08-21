# Kantorovich-accepted Newton

A unique-zero ball of a **finite** residual map is the accept test for
a Gauss–Newton trial. Empty ball means refuse the step. This is theory
spec 08-04: a sound enclosure, not a continuum PDE theorem and not a
Lean kernel pass.

## Scalar quadratic (spec §5)

```python
from omnibias.core.verified.kantorovich import (
    kantorovich_accept_step,
    polynomial_sqrt2_maps,
    select_accepted_params,
)

func, jac, lip = polynomial_sqrt2_maps()
a_inv = [[1.0 / 3.0]]
good = kantorovich_accept_step(
    func, jac, a_inv, [1.5], lipschitz_df=lip, r_max=0.2
)
far = kantorovich_accept_step(
    func, jac, a_inv, [3.0], lipschitz_df=lip, r_max=0.2
)
assert good.accepted and good.reason == "ball"
assert not far.accepted and far.reason == "empty"
assert good.certificate is not None
assert good.certificate.certificate["payload"]["continuum_pde_claim"] is False
assert "theorem_prover_verified" not in good.certificate.certificate["honesty"]
assert select_accepted_params([0.0], [1.5], good) == [1.5]
assert select_accepted_params([0.0], [3.0], far) == [0.0]
```

`F(x) = x^2 - 2`. The true root `sqrt(2)` lies in `B(1.5, r)` for the
accepted radius; the far point `x = 3` has no admissible `r`.

## Gate a torch Gauss–Newton trial

```python
import torch
from omnibias.torch.optim_kantorovich import (
    kantorovich_gated_gauss_newton_step,
    polynomial_sqrt2_maps,
)

torch.set_default_dtype(torch.float64)
func, jac, lip = polynomial_sqrt2_maps()

def residual(theta):
    x = theta.reshape(-1)[0]
    return (x * x - 2.0).reshape(1)

new, decision = kantorovich_gated_gauss_newton_step(
    residual, func, jac, torch.tensor([1.5]),
    lipschitz_df=lip, r_max=0.2,
)
assert decision.accepted
assert float(new[0]) > 0.0
```

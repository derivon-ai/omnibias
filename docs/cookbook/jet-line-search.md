# Exact jet line search

A directional jet of the loss is a Taylor polynomial along a search
direction. Stationary points are roots of that polynomial. The extra
evaluation (`verify=True`) refuses a step that would raise the true
loss. This is theory spec 03-12: a constant-factor line-search tool, not
a global solver and not a CCF stretch fix.

## Worked quartic (spec §5)

```python
from omnibias.core.line_search import (
    JetLineSearchConfig,
    select_model_step,
    taylor_coeffs_from_derivatives,
)

derivs = (1.0, -2.0, 6.0, -12.0, 48.0)
assert taylor_coeffs_from_derivatives(derivs) == [1.0, -2.0, 3.0, -2.0, 2.0]
result = select_model_step(
    derivs,
    radius=1.0,
    config=JetLineSearchConfig(order=4, verify=False),
)
assert abs(result.step - 0.409461) < 5e-5
assert result.model_value < 1.0
```

## Parameter step on a torch loss

```python
import torch
from omnibias.core.line_search import JetLineSearchConfig
from omnibias.torch.line_search import jet_line_search

torch.set_default_dtype(torch.float64)

def loss(p):
    s = p[0]
    return 1.0 - 2.0 * s + 3.0 * s**2 - 2.0 * s**3 + 2.0 * s**4

out = jet_line_search(
    loss,
    torch.zeros(2, dtype=torch.float64),
    torch.tensor([1.0, 0.0], dtype=torch.float64),
    config=JetLineSearchConfig(order=4, trust_radius=1.0, verify=True),
    next_derivative_bound=0.0,
)
assert out.actual_value is not None and out.actual_value <= 1.0
```

`next_derivative_bound=0.0` is sound here because the restriction is
degree 4 and the jet order is 4. For a non-polynomial `phi`, pass a
true bound on `|phi^(N+1)|` on the trial interval or keep `verify=True`
and treat the radius as uncertified.

Pair CubicNewton / GaussNewton (direction) with this primitive (length).
Spec 08-04 may later accept or reject the step on a unique-zero ball.

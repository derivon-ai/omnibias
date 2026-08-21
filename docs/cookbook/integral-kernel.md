# Integral-kernel operator

The cell matches `OperatorBlock(op="integral")`. On `f = cos` the
named Volterra operator recovers `sin` on `[0, π/2]`. The recorded
mass of a unit source is `0.5`, not `||f||_1`.

```python
from omnibias.core.integral_kernel import (
    antiderivative_skill,
    honesty_payload,
    integral_cell,
    worked_example,
)

ex = worked_example()
assert ex["g1_err"] < 1e-12
assert abs(ex["mass"] - 0.5) < 1e-12
assert abs(integral_cell(0.5, -0.5, 0.5) - ex["g1_cell"]) < 1e-12
skill = antiderivative_skill()
assert skill["below_1e3"] is True
assert skill["skill_positive"] is True
assert honesty_payload()["claimed_bem_net"] is False
```

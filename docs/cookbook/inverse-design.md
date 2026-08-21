# Inverse design

`tanh(2x) = 0.5` has the unique root `artanh(0.5)/2`. Newton from
`x=0` uses exact `sigma' = sech^2`. Saturated `|y|=0.999` raises.

```python
from omnibias.core.inverse_design import honesty_payload, invert_input, invert_skill, worked_example

ex = worked_example()
assert ex["residual"] < 1e-12
assert abs(ex["f_prime"] - 1.5) < 1e-12
skill = invert_skill()
assert skill["g2_earned"] is True
raised = False
try:
    invert_input(None, 0.999, 0.0)
except ValueError:
    raised = True
assert raised
assert honesty_payload()["global_inverse_claimed"] is False
```

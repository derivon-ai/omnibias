# Sharpness regularizer

`L(theta) = 5 theta^2` has Hessian `10`. With `mu=0.1` the
augmented loss at `theta=0` is `1`.

```python
from omnibias.core.sharp_loss import honesty_payload, sharpness_skill, worked_example

ex = worked_example()
assert ex["hvp"] == 10.0
assert ex["aug"] == 1.0
skill = sharpness_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["is_08_06_schedule"] is False
```

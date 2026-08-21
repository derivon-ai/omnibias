# Remainder training

`exp` about `0` has `T_2 = 1 + x + x^2/2`. At `x = 0.2` the
remainder is about `1.4e-3`, and a model that *is* `T_2` has
loss `R_2^2 ≈ 1.97e-6`.

```python
from omnibias.core.remainder_train import honesty_payload, worked_example

ex = worked_example()
assert ex["R2_err"] < 1e-12
assert 1e-6 < ex["loss"] < 3e-6
assert honesty_payload()["is_03_10"] is False
assert honesty_payload()["is_03_13"] is False
assert honesty_payload()["stretch_claim"] is False
```

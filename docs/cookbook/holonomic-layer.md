# Holonomic layer

`exp` is annihilated by `D-1`. Given `u(0)=1`, the jet at `0` is
all ones. `sin` uses `D^2+1` with `(u, u') = (0, 1)` at `0`.

```python
from omnibias.holonomic._core.layer import honesty_payload, sin_skill, worked_example

ex = worked_example()
assert ex["exp_err"] < 1e-12
assert ex["sin_u2"] == 0
assert ex["sin_u3"] == -1
skill = sin_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["nonlinear_pde_claimed_dfinite"] is False
```

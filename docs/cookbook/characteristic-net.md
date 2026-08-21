# Characteristic-Net

Linear advection with frozen `v = 1` shifts a Gaussian by `t`. At
`t = 0.2`, `x = 0` the foot is `-0.2` and
`u = exp(-0.04) ≈ 0.960789`.

```python
from omnibias.pinn.characteristic import honesty_payload, worked_example

ex = worked_example()
assert ex["err"] < 1e-12
assert ex["crossed"] == 0.0
assert honesty_payload()["is_02_13"] is False
assert honesty_payload()["unique_after_shock_claimed"] is False
assert honesty_payload()["ns_claim"] is False
```

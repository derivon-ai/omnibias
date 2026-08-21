# Sheaf-atlas net

Two charts on the line: `Phi_21(x) = 2x` and `Phi_32(y) = y/2`, so
`Phi_31` is the identity. The order-1 jet of the identity is
`(x, 1)`. Composing `(2x, 2)` then `(y/2, 1/2)` recovers `(x, 1)`.

```python
from omnibias.geometry.atlas.cocycle import honesty_payload, worked_example

ex = worked_example()
assert ex["residual"] < 1e-12
assert abs(ex["composed_deriv"] - 1.0) < 1e-12
assert ex["buggy_deriv_gap"] > 1e-3
assert honesty_payload()["temperature_collapse_used"] is False
assert honesty_payload()["sheaf_cohomology_theorem"] is False
```

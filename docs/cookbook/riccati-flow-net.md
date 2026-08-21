# Riccati flow net

Logistic flow `ds/dt = s(1-s)` from `s0 = 0.25` at time `t = 1`
is `s0 / (s0 + (1-s0) e^{-t})`. The jet head `ds/ds0` is
`e^{-t} / (s0 + (1-s0) e^{-t})^2`.

```python
from omnibias.core.riccati_flow import honesty_payload, worked_example

ex = worked_example()
assert ex["err"] < 1e-12
assert ex["ds_ds0_err"] < 1e-12
assert honesty_payload()["claimed_deq"] is False
assert honesty_payload()["claimed_cnf"] is False
```

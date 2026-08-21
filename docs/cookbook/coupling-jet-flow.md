# Coupling jet-flow

`y = tanh(2 x)` at `x = 0.3` has a closed-form log-det. Newton
recovers `x` from `y`.

```python
from omnibias.core.coupling_flow import honesty_payload, worked_example

ex = worked_example()
assert ex["log_det_err"] < 1e-12
assert ex["inv_err"] < 1e-12
assert honesty_payload()["imagenet_claim"] is False
assert honesty_payload()["rewrites_integrate_cnf"] is False
```

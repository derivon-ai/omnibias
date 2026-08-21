# FTC-Net

A symmetric sigmoid window of width `0.2` has collapse head
`0.5` and an FTC derivative equal to the activation band.

```python
from omnibias.core.ftc import honesty_payload, worked_example

ex = worked_example()
assert ex["collapse_err"] < 1e-12
assert ex["ftc_err"] < 1e-12
assert honesty_payload()["claimed_weak_form"] is False
assert honesty_payload()["claimed_vpinn"] is False
```

# Collapse-Net

Centered first difference of `sigmoid` at `0` with `delta = 0.1`
is `(sigmoid(0.1) - sigmoid(-0.1)) / 0.2`. Founding bias collapse
(`delta -> 0`) replaces that stencil with `sigmoid'(0) = 0.25`.
The recorded remainder is the Birkhoff gap, not a continuum PDE.

```python
from omnibias.core.collapse_net import honesty_payload, sin_skill, worked_example

ex = worked_example()
assert ex["rel_err"] < 1e-6
assert abs(ex["collapsed"] - 0.25) < 1e-12
skill = sin_skill()
assert skill["below_1e3"] is True
assert skill["collapsed_no_worse"] is True
assert honesty_payload()["continuum_pde_claimed"] is False
```

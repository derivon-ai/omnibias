# Pack-MoE

Two band windows about the origin have equal mass at `x=0`, so the
gates are `1/2` and `y = 2` when the experts are `1` and `3`.

```python
from omnibias.core.pack_moe import honesty_payload, worked_example

ex = worked_example()
assert ex["g_a_err"] < 1e-12
assert ex["g_b_err"] < 1e-12
assert ex["y_err"] < 1e-12
assert honesty_payload()["router_is_softmax"] is False
assert honesty_payload()["temperature_collapse_used"] is False
```

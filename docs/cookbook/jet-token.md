# Jet-token mix

An affine mix of two order-1 jets, then `tanh`, matches the
closed-form `compose_jet` numbers. Softmax-of-values is a different
mix.

```python
from omnibias.core.jet_token import honesty_payload, recover_tanh_scale, worked_example

ex = worked_example()
assert ex["value_err"] < 1e-12
assert ex["deriv_err"] < 1e-12
assert recover_tanh_scale(0.5)["loss"] < 1e-12
assert honesty_payload()["imagenet_claim"] is False
```

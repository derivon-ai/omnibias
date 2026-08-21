# Frame-UNet

At `x = 0` the band skip of a width-`0.2` integral cell is
`sigmoid(0.1) - sigmoid(-0.1)`, about `0.05`. The order-1 collapse
head at the mean is `sigmoid'(0) = 0.25`. Those numbers must stay
apart.

```python
from omnibias.core.frame_unet import honesty_payload, worked_example

ex = worked_example()
assert ex["band_err"] < 1e-12
assert ex["collapse_err"] < 1e-12
assert ex["skip_gap"] > 1e-3
assert honesty_payload()["sigma_prime_admissible"] is False
assert honesty_payload()["stretch_claim"] is False
```

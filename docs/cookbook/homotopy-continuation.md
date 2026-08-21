# Homotopy continuation

`H(theta, tau) = theta - 1 - tau theta^2`. At `tau=0.1` a Newton
step from `1` is 08-04-accepted and sits on the branch near `1.127`.
At `tau=1` there is no real root; an empty ball is a halt.

```python
from omnibias.core.homotopy import honesty_payload, homotopy_skill, worked_example

ex = worked_example()
assert ex["accepted"] is True
assert ex["abs_h"] < 1e-10
skill = homotopy_skill()
assert skill["honest"] is True
assert skill["g2_earned"] is True
assert honesty_payload()["stretch_claim"] is False
```

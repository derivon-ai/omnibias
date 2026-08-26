# Jet-LQR on a directional restriction

`phi(s) = (s-1)^2` from `s=0` has `phi'=-2`, `phi''=2`. Finite-horizon
LQR with `R=0`, `N=1`, `Qf=1` is Newton and steps to the minimizer.
`R>0` shrinks that step. Bias collapse (`delta -> 0`) supplies the
jet. This is not the activation Riccati.

```python
from omnibias.core.control_lqr import (
    JetLQRConfig,
    honesty_payload,
    lqr_skill,
    lqr_step_from_derivatives,
    worked_example,
)

ex = worked_example()
assert ex["newton_recovered"] is True
assert ex["damped_smaller"] is True
report = lqr_step_from_derivatives(
    (1.0, -2.0, 2.0),
    config=JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1),
)
assert report.step == 1.0
skill = lqr_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["algebraic_riccati_claimed"] is False
assert honesty_payload()["activation_riccati_as_gain"] is False
```

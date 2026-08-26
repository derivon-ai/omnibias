# Jet-PID on a directional restriction

`phi(s) = (s-1)^2` from `s=0` has `phi'=-2`. With `kp=0.5` the P term
alone steps to the minimizer. The I term is the FTC of the Taylor
model, not a discrete gradient sum. Bias collapse (`delta -> 0`)
supplies the jet.

```python
from omnibias.core.control_pid import (
    JetPIDConfig,
    honesty_payload,
    pid_skill,
    pid_step_from_derivatives,
    worked_example,
)

ex = worked_example()
assert ex["abs_step_err"] < 1e-15
assert ex["damped_is_newton"] is True
report = pid_step_from_derivatives(
    (1.0, -2.0, 2.0),
    config=JetPIDConfig(kp=0.5, ki=0.0, kd=0.0, s_max=2.0),
)
assert report.step == 1.0
skill = pid_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["global_min_claimed"] is False
assert honesty_payload()["plant_pid_claimed"] is False
```

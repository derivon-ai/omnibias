# Plant PID with exact I and D

`y = sigmoid(t)` from `t0=-1` to `t=0` with setpoint `1/2` has `e(0)=0`
and `D=-1/4`. The I term is the FTC of `r - sigmoid`, not a running
sum. Bias collapse (`delta -> 0`) supplies `sigma'`.

```python
from omnibias.core.pid_layer import (
    PlantPIDConfig,
    honesty_payload,
    plant_pid,
    plant_pid_skill,
    worked_example,
)

ex = worked_example()
assert ex["ftc_residual"] < 1e-8
assert abs(ex["derivative"] + 0.25) < 1e-15
report = plant_pid(
    0.0,
    config=PlantPIDConfig(kp=0.0, ki=1.0, kd=1.0, setpoint=0.5, t0=-1.0),
)
assert report.measurement == 0.5
skill = plant_pid_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["trainer_claimed"] is False
assert honesty_payload()["discrete_sum_i"] is False
```

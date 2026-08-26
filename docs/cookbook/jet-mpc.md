# Receding jet-MPC on a directional restriction

Unconstrained LQR on `phi(s)=(s-1)^2` recovers Newton. An input box
clips the applied first control; the report stays receding. Bias
collapse (`delta -> 0`) supplies the jet. This is not plant MPC.

```python
from omnibias.core.control_mpc import (
    JetMPCConfig,
    honesty_payload,
    mpc_skill,
    mpc_step_from_derivatives,
    worked_example,
)

ex = worked_example()
assert ex["newton_recovered"] is True
assert ex["boxed"] is True
assert ex["receding"] is True
boxed = mpc_step_from_derivatives(
    (1.0, -2.0, 2.0),
    config=JetMPCConfig(q=0.0, r=0.0, qf=1.0, horizon=1, u_max=0.25),
)
assert boxed.step == 0.25
assert boxed.reason == "boxed"
skill = mpc_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["plant_mpc_claimed"] is False
assert honesty_payload()["general_qp_claimed"] is False
```

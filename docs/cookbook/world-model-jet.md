# World-model-as-jet

The order-2 Taylor of the harmonic oscillator `x''=-x` at `dt=0.1`
is `1 - dt^2/2 = 0.995`. A Lohner box around the linear field
contains `cos(0.1)`.

```python
from omnibias.core.jet_world import honesty_payload, jet_world_skill, worked_example

ex = worked_example()
assert ex["abs_err"] < 1e-12
assert abs(ex["pred"] - 0.995) < 1e-12
skill = jet_world_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["navier_stokes_proof_claim"] is False
```

# Parameter-space jets

The Fourier heat mode `u = e^{-μ t} sin(x)` at `t=1`, `μ=1` has
`∂u/∂μ = -u`.

```python
from omnibias.core.parameter_jets import ParameterJetSpec, honesty_payload, mixed_jet, parameter_jet_skill, worked_example

ex = worked_example()
assert ex["abs_sum"] < 1e-8
skill = parameter_jet_skill()
assert skill["g2_earned"] is True
raised = False
try:
    mixed_jet(None, (1.0, 1.0), 1.0, spec=ParameterJetSpec(mu_in_jet_trunk=False))
except ValueError:
    raised = True
assert raised
assert honesty_payload()["parampinn_package"] is False
```

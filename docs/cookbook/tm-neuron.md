# Taylor-model neuron and PCI

An order-1 sigmoid Taylor model on `[0, 0.2]` encloses the true
range. The remainder is non-vacuous. The PCI forward returns that
box and does not forge Lean flags.

```python
from omnibias.core.verified.tm_neuron import honesty_payload, worked_example
from omnibias.verify._core.pci import worked_sigmoid_box

ex = worked_example()
assert float(ex["remainder_width"]) < 0.1
assert honesty_payload()["imagenet_claim"] is False
result, _tm = worked_sigmoid_box()
assert result.width < 0.1
assert result.theorem_prover_verified is False
assert result.is_08_09_step_filter is False
```

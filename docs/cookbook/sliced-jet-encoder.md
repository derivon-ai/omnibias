# Sliced-jet encoder

The 2×2 image `[[1, 0], [0, 0]]` is two axis scans. Copying those
profiles onto the row and column reconstructs the image exactly.
Global average-pool reconstructs the constant `0.25`.

```python
from omnibias.core.sliced_jet import SlicedJetConfig, honesty_payload, sliced_jet_skill, worked_example

ex = worked_example()
assert ex["encoder_mae"] == 0.0
assert ex["gap_mae"] == 0.375
skill = sliced_jet_skill()
assert skill["g2_earned"] is True
raised = False
try:
    SlicedJetConfig(top_k=1, energy="none")
except ValueError:
    raised = True
assert raised
assert honesty_payload()["vit_claim"] is False
```

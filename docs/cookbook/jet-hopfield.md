# Jet-Hopfield

Two memories `J1=(1,0)` and `J2=(0,1)`. A query `(1, 0.01)` with
`λ=1` and `β=10` is contact-nearest to `J1`. A third memory
`(0.5, 0)` is value-nearest to `(0.6, 1)` but not contact-nearest.

```python
from omnibias.core.jet_hopfield import contact_split, honesty_payload, worked_example

ex = worked_example()
assert ex["value_err"] < 1e-6
skill = contact_split()
assert skill["split"] is True
assert skill["lam"] == 1.0
assert honesty_payload()["temperature_collapse_used"] is False
```

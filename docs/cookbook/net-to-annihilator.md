# Net-to-annihilator

`D-1` round-trips through JSON-ready rationals. Verified flags stay
false. A continuum Lean sentence is refused.

```python
from omnibias.holonomic._core.export import export_annihilator, export_skill, honesty_payload, worked_example
from omnibias.holonomic._core.layer import d_minus_1

ex = worked_example()
assert ex["match"] is True
exported = export_annihilator(d_minus_1())
assert exported.theorem_prover_verified is False
skill = export_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["mathlib_verified"] is False
```

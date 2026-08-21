# Exact score matching

`s(x) = -x` is the standard-normal score. Its divergence is `-1`,
so the Hyvärinen integrand at `x=1` is `-0.5`.

```python
from omnibias.core.score_matching import honesty_payload, score_matching_skill, worked_example

ex = worked_example()
assert abs(ex["div"] + 1.0) < 1e-12
assert abs(ex["hyvarinen"] + 0.5) < 1e-12
skill = score_matching_skill()
assert skill["g2_earned"] is True
assert honesty_payload()["cnf_div_claimed_new"] is False
```

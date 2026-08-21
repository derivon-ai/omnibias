# q-OMBU / timescale hybrid

`D_q z^2` at `q=1.01`, `z=2` is `4.02`. The ordinary derivative is
`4`. The residual shrinks as `q -> 1`.

```python
from omnibias.qcalculus._core.hybrid import (
    honesty_payload,
    q_limit_skill,
    q_ombu_limit,
    worked_example,
)

ex = worked_example()
assert ex["y_err"] < 1e-12
assert ex["limit_err"] < 1e-12
assert q_ombu_limit(2.0) == 4.0
skill = q_limit_skill()
assert skill["monotone"] is True
assert honesty_payload()["continuum_claimed_from_q_limit"] is False
```

# Rank collapse cookbook

Rank collapse reconstructs an exact integer nullspace and checks
`M v = 0` over `Q`. A float singular value is not a proof. An empty
kernel disproves "a nontrivial syzygy exists" for this matrix. It
does not certify a special-function identity.

## A dependent pair has a syzygy; the identity does not

```python
from omnibias.core.collapse import rank_collapse

proved = rank_collapse(((1, 2), (2, 4)))
assert proved.verdict.status == "PROVED"
assert proved.kernel
full = rank_collapse(((1, 0), (0, 1)))
assert full.verdict.status == "DISPROVED"
assert full.kernel == ()
```

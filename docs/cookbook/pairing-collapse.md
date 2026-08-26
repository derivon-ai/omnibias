# Pairing collapse cookbook

Pairing collapse decides the finite weak statement
`<R, φ_g> = 0` for every test in a named pack. Exact `Q` integrals of
polynomial products are the certificate. An odd residual against even
tests can collapse without `R` being zero -- that is a weak statement
on this pack, not a strong solution and not a PDE.

## A constant residual is not weakly zero against mass

```python
from omnibias.core.collapse import pairing_collapse

disproved = pairing_collapse((1,), ((1,),))
assert disproved.status == "DISPROVED"
zero = pairing_collapse((0,), ((1,), (0, 1)))
assert zero.status == "PROVED"
assert zero.outcome.honesty["not_a_strong_solution"] is True
odd = pairing_collapse((0, 1), ((1,),))
assert odd.status == "PROVED"
```

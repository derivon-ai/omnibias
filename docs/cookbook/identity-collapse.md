# Identity collapse cookbook

Identity collapse decides whether two polynomials (or a polynomial and
its degree-`N` jet) agree on a compact. Exact `Q` coefficient
agreement is the singleton `{0}`. A float remainder loss is not a
proof. This is not founding bias collapse (`delta -> 0`) and not
temperature collapse (`beta -> inf`).

## A binomial identity collapses

```python
from omnibias.core.collapse import identity_collapse, remainder_collapse
from omnibias.core.verified.interval import Interval

proved = identity_collapse((1, 2, 1), (1, 2, 1), Interval(-2.0, 2.0))
assert proved.status == "PROVED"
assert proved.outcome.surviving == "germ_identity"
assert remainder_collapse((1, 2, 1), 2, Interval(-1.0, 1.0)).status == "PROVED"
```

## A truncated jet is inconclusive on a fat remainder

```python
from omnibias.core.collapse import remainder_collapse
from omnibias.core.verified.interval import Interval

blocked = remainder_collapse((1, 2, 1), 1, Interval(-1.0, 1.0))
assert blocked.status == "BLOCKED"
disproved = remainder_collapse((1, 1), 0, Interval.point(2.0))
assert disproved.status == "DISPROVED"
```

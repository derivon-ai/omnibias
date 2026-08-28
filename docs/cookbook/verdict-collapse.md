# Verdict collapse cookbook

Verdict collapse turns a *sound residual enclosure* of a finite
obligation into a three-way verdict. A float loss going to zero is
not a proof. Collapse onto `{0}` is `PROVED`. Exclusion of `0` is
`DISPROVED`. A fat interval that still contains `0` is `BLOCKED`
(`Inconclusive`), not false. This is not founding bias collapse
(`delta -> 0`), not temperature collapse (`beta -> inf`), and not
Enclosure Collapse of a value (`width -> 0` of a sound enclosure, a point plus a proof).

## `{0}` proves; a fat zero does not

```python
from omnibias.core.collapse import adjudicate_residual
from omnibias.core.verified.interval import Interval

proved = adjudicate_residual(Interval.point(0.0))
assert proved.status == "PROVED"
blocked = adjudicate_residual(Interval(-1e-6, 1e-6))
assert blocked.status == "BLOCKED"
assert "not false" in blocked.detail
disproved = adjudicate_residual(Interval.point(1.0))
assert disproved.status == "DISPROVED"
```

## A float residual is refused

```python
from omnibias.core.collapse import adjudicate_residual

try:
    adjudicate_residual(0.0)
    raise AssertionError("float residual must be refused")
except TypeError as exc:
    assert "float residual" in str(exc)
```

## Complete existential miss is not a parent claim

```python
from omnibias.core.collapse import search_residuals
from omnibias.core.verified.interval import Interval

miss = search_residuals(
    [Interval.point(1.0), Interval.point(2.0)],
    existential=True,
    complete=True,
)
assert miss.status == "BLOCKED"
assert "not a parent claim" in miss.detail
```

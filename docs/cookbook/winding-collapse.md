# Winding collapse cookbook

Winding collapse encloses `Δarg / 2π` of a polynomial on a circle and
accepts only when that enclosure contains exactly one integer. The
surviving object is a winding number, not a derivative. A contour that
hits a zero, or a box that crosses the branch cut, is `BLOCKED`. This
is not a blow-up proof.

## `z` winds once around the origin

```python
from omnibias.core.collapse import winding_collapse

proved = winding_collapse((0, 1), expected=1)
assert proved.status == "PROVED"
assert proved.outcome.surviving == 1
outside = winding_collapse((-2, 1), expected=0)
assert outside.status == "PROVED"
wrong = winding_collapse((0, 1), expected=0)
assert wrong.status == "DISPROVED"
```

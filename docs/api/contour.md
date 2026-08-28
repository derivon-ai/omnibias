# Contour winding

`winding_enclosure` and `winding_collapse` certify the winding number of a
polynomial image when each segment stays away from zero and its argument can
be enclosed without crossing the principal branch cut. The same winding
collapse supports circular and axis-aligned rectangular contours; a rectangle
is a contour choice, not a new kind of collapse.

```python
from omnibias.core.collapse import winding_collapse

# p(z) = z winds once around the origin on this rectangle.
result = winding_collapse(
    (0, 1),
    center=0j,
    radius=2.0,
    contour="rectangle",
    half_width=2.0,
    half_height=1.0,
    segments=64,
    expected=1,
)
assert result.status == "PROVED"
assert result.outcome.surviving == 1
```

For `contour="rectangle"`, `half_width` and `half_height` default to
`radius`, so a radius-only call uses a square. Rectangle segment counts must
be divisible by four. An image segment containing zero or crossing the
argument branch cut is `BLOCKED`, not a zero-count certificate.

The result applies only to the declared polynomial and contour. It is not a
blow-up proof or a continuum PDE statement.

## API

::: omnibias.core.collapse.winding
    options:
      show_root_heading: false
      heading_level: 3

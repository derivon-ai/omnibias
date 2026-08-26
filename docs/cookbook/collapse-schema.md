# Named-collapse schema cookbook

The founding **bias** (`delta -> 0`), **temperature** (`beta -> inf`),
and **enclosure** (`width -> 0` of a sound enclosure) limits stay
exactly those three. The registry catalogues them and refuses a
rebrand. A float residual is not a certificate.

## Founding three are distinct

```python
from omnibias.core.collapse import are_distinct, list_collapses, reset_collapse_registry

reset_collapse_registry()
names = {spec.name for spec in list_collapses()}
assert {"bias", "temperature", "enclosure"} <= names
founding = {spec.name: spec for spec in list_collapses()}
assert are_distinct(founding["bias"], founding["enclosure"]).distinct
assert founding["bias"].surviving_object == "derivative"
assert founding["temperature"].surviving_object == "indicator"
assert founding["enclosure"].surviving_object == "point_plus_proof"
```

## A rebrand of Enclosure Collapse is refused

```python
from omnibias.core.collapse import CollapseSpec, register_collapse, reset_collapse_registry

reset_collapse_registry()
clone = CollapseSpec(
    name="width_squeeze",
    parameter="width",
    limit="0",
    surviving_object="point_plus_proof",
    failure="Inconclusive",
    home="omnibias.core.collapse.schema",
    register="verified",
)
try:
    register_collapse(clone)
    raise AssertionError("rebrand of enclosure must be refused")
except ValueError as exc:
    assert "rebrand" in str(exc)
reset_collapse_registry()
```

## A float residual is not a proof

```python
from omnibias.core.collapse import require_sound_enclosure
from omnibias.core.verified.interval import Interval

try:
    require_sound_enclosure(0.0)
    raise AssertionError("float residual must be refused")
except TypeError as exc:
    assert "float residual" in str(exc)
assert require_sound_enclosure(Interval.point(0.0)).contains_zero()
```

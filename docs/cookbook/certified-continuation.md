# Finite continuation primitives (not RH)

The Dirichlet *series* register stays `Re(s) > 1`. A separate finite
evaluator encloses the continued *value* of `zeta` on named compact
rectangles via a functional-equation identity (`Re(s) < 0`) and an
approximate functional equation on a locked `T` compact. This is
numerical / compact, not a continuation theorem, and it does not infer
the Riemann Hypothesis.

## Left half-plane evaluator

```python
from omnibias.core.verified.xi import (
    continuation_honesty,
    zeta_via_functional_equation,
    zeta_winding_count,
)

enc = zeta_via_functional_equation(-1.0, num_terms=80)
assert enc.re.contains(-1.0 / 12.0)
honesty = continuation_honesty()
assert honesty["rh_claim"] is False
assert honesty["continuation_theorem"] is False
count, winding = zeta_winding_count(3.0, 0.25, 0.25, segments=8, num_terms=60)
assert count == 0
assert winding is not None
```

## Honesty

The smoke
[`certified_continuation_smoke.json`](../benchmarks/certified_continuation_smoke.json)
keeps `rh_claim`, `continuation_theorem`, and `zeros` false. Far-field
and a whole-line `H_t` cover stay unearned. Do not attach this JSON to
the RH ledger Distance cell.

## See also

- API: [`omnibias.core.verified.xi`](../api/core.md)
- Dirichlet series register: [dirichlet-enclosure](dirichlet-enclosure.md)
- Ledger: [`frontier-ledger.md`](../frontier-ledger.md)

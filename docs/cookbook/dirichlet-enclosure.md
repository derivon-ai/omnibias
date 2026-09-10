# Dirichlet enclosures on `Re(s) > 1`

The verified Dirichlet primitives enclose `zeta`, Dirichlet `L`, and
Jacobi theta **only** on `Re(s) > 1`. A width smoke records coverage
and a named width cap. This is a ledger **non-entry** for the Riemann
Hypothesis: there are no zeros, no continuation, and no
`zeta_euler_maclaurin` in the artifact.

## Zeta at `s = 2`

```python
from omnibias.core.verified.dirichlet import zeta_enclosure

enc = zeta_enclosure(2.0, num_terms=240)
assert enc.re.contains(1.644934)  # π²/6
assert enc.re.lo > 1.0
try:
    zeta_enclosure(1.0, num_terms=8)
except ValueError as exc:
    assert "Re" in str(exc)
```

## Dirichlet beta and theta

```python
from omnibias.core.verified.dirichlet import (
    l_function_enclosure,
    theta_enclosure,
)

chi = (0.0, 1.0, 0.0, -1.0)
beta2 = l_function_enclosure(chi, 2.0, num_terms=240)
assert beta2.re.lo > 0.0
th = theta_enclosure(0.0, 1.0, num_terms=40)
assert th.lo > 1.0
assert th.hi > th.lo
```

## Honesty

The smoke
[`dirichlet_enclosure_smoke.json`](../benchmarks/dirichlet_enclosure_smoke.json)
sets `riemann_hypothesis_subject`, `zeros`, and
`zeta_euler_maclaurin` to false. Padé / Borel must not consume a
Dirichlet series.

## See also

- API: [`omnibias.core.verified.dirichlet`](../api/core.md)
- Finite evaluator (not RH): [certified-continuation](certified-continuation.md)
- Ledger: [`frontier-ledger.md`](../frontier-ledger.md)

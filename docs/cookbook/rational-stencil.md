# Rational stencil Lean obligations

The Birkhoff scheme from spec 01-04 / 01-11 recovers `f'(0)` from
`f(-h)`, `f(0)`, and `f'(h)`. The scale-free weights are exact
rationals. The Lean kernel checks the finite identities `C_j`, not
the collapse.

```python
from fractions import Fraction

from omnibias.core.proof import (
    seal_stencil_certificate,
    stencil_consistency_obligation,
)
from omnibias.core.proof.obligations.rational_stencil import RationalStencil

st = RationalStencil(
    (Fraction(-1), Fraction(0), Fraction(1)),
    ((0,), (0,), (1,)),
    ((Fraction(-2, 3),), (Fraction(2, 3),), (Fraction(1, 3),)),
    1,
    leading_coeff=Fraction(5, 18),
    name="birkhoff_spec",
)
assert st.moment_sum(0) == 0
assert st.moment_sum(1) == 1
assert st.moment_sum(2) == 0
assert st.computed_leading() == Fraction(5, 18)
obl = stencil_consistency_obligation(st)
assert obl.holds
report = seal_stencil_certificate(st, run_lean=False)
assert report.mathlib_verified is False
assert "theorem_prover_verified" not in report.certificate["honesty"]
assert report.certificate["payload"]["type"] == "rational_stencil_consistency"
```

Corrupting a weight makes the Python algebra fail. The emitted Lean
asks `allRatEq [...] = true` of those unequal pairs, so a kernel
build rejects it.

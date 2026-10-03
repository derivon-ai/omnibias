# Exact-Q Groebner engine

omnibias.holonomic._core.groebner is a genuine exact-Q Buchberger
implementation over PolyN: lex / grlex / degrevlex monomial orders,
multivariate division with remainder, both classical Buchberger
pair-skipping criteria (coprime leading monomials; the chain criterion), a
reduced (monic, minimal, interreduced) Groebner basis, and ideal / radical
membership that return an exact cofactor witness a caller replays without
rerunning Buchberger. Buchberger's algorithm is doubly exponential in the
worst case, so every entry point takes a GroebnerBudget and raises
GroebnerBudgetExceeded loudly rather than running unbounded -- a budget
refusal is an engineering limit, not a mathematical statement that no
Groebner basis exists.

This module does not provide a completion procedure beyond Buchberger
(no F4/F5, no signature-based criteria), and it does not itself decide
anything about a center variety or a physical dynamical system -- see
omnibias.dynamics.bautin for the consumer that builds a focal-value ideal
on top of it.

## Basis and membership

```python
from omnibias.holonomic._core.groebner import (
    ideal_member,
    reduced_groebner_basis,
    verify_ideal_membership,
)
from omnibias.holonomic._core.poly_n import PolyN

# The classic unit-circle-union-both-axes example: (x^2+y^2-1, x*y).
x, y = PolyN.var(2, 0), PolyN.var(2, 1)
generators = [x * x + y * y - PolyN.const(2, 1), x * y]
basis = reduced_groebner_basis(generators, "degrevlex")

candidate = x * x * y - x * y
membership = ideal_member(candidate, generators, "degrevlex")
assert membership.is_member
assert verify_ideal_membership(candidate, generators, membership)
assert len(basis) >= 1
```

ideal_member and radical_member (Rabinowitsch's trick, `1 in (generators, 1
- t*f)`) both return an IdealMembership(is_member, remainder, cofactors,
basis) whose `f == sum(cofactors[i] * generators[i]) + remainder` identity
verify_ideal_membership replays by plain polynomial arithmetic -- the
witness, not the search, is what a caller should trust.

## Budget refusal

```python
from omnibias.holonomic._core.groebner import (
    GroebnerBudget,
    GroebnerBudgetExceeded,
    reduced_groebner_basis,
)

refused = False
try:
    reduced_groebner_basis(
        generators, "degrevlex", budget=GroebnerBudget(max_pairs=1, max_polynomials=1)
    )
except GroebnerBudgetExceeded:
    refused = True
assert refused
```

## Reference

::: omnibias.holonomic._core.groebner
    options:
      show_root_heading: false
      heading_level: 3

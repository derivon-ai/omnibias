# Frobenius & Puiseux local series

Two classical "expand around a singularity" constructions, packaged on top of
the `omnibias.core.verified.sequence_space.ValidatedSeries` primitive so every
returned series carries an explicit, honestly-derived tail bound rather than a
bare truncated float sum.

**Frobenius series** (`indicial_roots`, `frobenius_coefficients`,
`solve_frobenius`) solve a 2nd-order linear ODE with a regular singular point
at `x = 0`: `x**2 y'' + x p(x) y' + q(x) y = 0`. The indicial equation
`r(r-1) + p_0 r + q_0 = 0` is solved as a certified quadratic, and the caller
picks one of its two roots to build `y = x**r * sum a_n x**n`. When the two
indicial roots differ by a non-negative integer the classical second
solution needs a logarithmic term; that generalized construction is *out of
scope*, and every entry point raises `ValueError` rather than silently
returning an incomplete series.

**Puiseux branches** (`newton_polygon_leading_term`, `puiseux_coefficients`,
`solve_puiseux`) solve `F(x, y) = 0` near a branch point at `x = 0` with a
*caller-supplied* ramification index `e` (deriving the full Newton polygon
automatically is out of scope). A single Newton-polygon leading-term step
recovers `y ~ c_0 x**(k0/e)`, and substituting the local uniformizer
`x = t**e` turns the rest of the branch into an ordinary power series in `t`.

Every returned object carries an explicit tail bound and the exact hypothesis
under which it is sound (a caller-supplied consecutive-ratio bound on the
recurrence's tail); an unjustifiable hypothesis is either rejected outright
or produces an enclosure that a directly-sampled true value can violate --
soundness is never assumed silently.

## Frobenius series: Bessel's equation

`x**2 y'' + x y' - (x**2 + nu**2) y = 0` is the *modified* Bessel equation:
`p = [1]`, `q = [-nu**2, 0, -1]`. Its indicial roots are `+-nu`, and the
larger root's Frobenius series relates to `I_nu` by
`I_nu(x) = y(x) / (2**nu * Gamma(nu+1))`.

```python
from omnibias.core.verified.frobenius import (
    frobenius_coefficients,
    indicial_roots,
    solve_frobenius,
)

nu = 1.0
p = [1.0]
q = [-nu * nu, 0.0, -1.0]

r_plus, r_minus = indicial_roots(p[0], q[0])
assert r_plus.contains(1.0) and r_minus.contains(-1.0)

# a_0=1, odd a_n vanish, a_2 = 1/8 for nu=1 (the classical Bessel coefficient).
coeffs = frobenius_coefficients(p, q, nu, num_terms=5)
assert coeffs[0].contains(1.0)
assert coeffs[2].contains(0.125)

# Package with a tail bound: |a_{n+1}/a_n| <= 0.1 from the last kept term on,
# weighted at radius nu_weight=1.0 -- a hypothesis that must be justified from
# the recurrence's own asymptotics (see the module docstring).
sol = solve_frobenius(p, q, nu, num_terms=9, ratio=0.1, nu=1.0)
y = sol.evaluate(0.5)
i1_enclosure = y / 2.0  # 2**nu * Gamma(nu+1) = 2 * 1! = 2
assert i1_enclosure.contains(0.257894305390896)  # I_1(0.5), independently known
```

Roots differing by a non-negative integer are refused once the recurrence
actually reaches the resonant order -- here the Euler equation
`x**2 y'' - x y' + y = 0` (`p0=-1`, `q0=0`) has indicial roots `0` and `2`:

<!-- docs-test: raises=ValueError -->

```python
from omnibias.core.verified.frobenius import frobenius_coefficients

p, q = [-1.0], [0.0]
frobenius_coefficients(p, q, 0.0, num_terms=5)  # root=0 is the smaller root
```

The larger root of the same pair (`root=2.0`) never resonates and computes
cleanly for any `num_terms`; only the *smaller* root of a gapped pair hits
the classical log-term obstruction.

## Puiseux branch: `y**2 = x**3`

The curve `y**2 = x**3` has two branches `y = +-x**(3/2)`, ramification
`e=2`. The Newton-polygon step recovers the leading exponent `k0=3` (so
`y ~ c0 * x**(3/2)`) and both leading coefficients `c0 = +-1`; substituting
`x = t**2` then gives an ordinary series in `t` whose higher coefficients are
all exactly zero for this branch.

```python
from omnibias.core.verified.frobenius import (
    newton_polygon_leading_term,
    puiseux_coefficients,
    solve_puiseux,
)

# F(x, y) = y**2 - x**3 = 0
coeffs = {(0, 2): 1.0, (3, 0): -1.0}
k0, roots = newton_polygon_leading_term(coeffs, e=2)
assert k0 == 3
assert len(roots) == 2  # c0 = +1 and c0 = -1

branch = solve_puiseux(coeffs, e=2, k0=k0, c0=roots[0], num_terms=4, ratio=0.1, nu=2.0)
y = branch.evaluate_x(0.64)
assert y.contains(0.64**1.5)
```

## API

::: omnibias.core.verified.frobenius
    options:
      show_root_heading: false
      heading_level: 3

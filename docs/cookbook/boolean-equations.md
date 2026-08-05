# Differentiable Boolean algebra: equations, spectra, and the derivative bridge

`omnibias-boolean` treats a Boolean function two ways at once: as an **exact**
combinatorial object (truth table, ANF, Walsh spectrum, Boolean derivative,
reproductive equation solution) and as a **differentiable relaxation** (its
multilinear extension, whose mixed partial derivatives *are* the spectrum). This
page is the conceptual tour; everything here runs against the package.

!!! warning "What is and isn't claimed"
    The pure-Python `_core` transforms are exact (integer / GF(2)). The torch/jax
    backends are differentiable **heuristics** with no completeness guarantee.
    Numerically exact derivatives are not logically complete reasoning. The system
    solver is **propose-and-verify**: it relaxes and optimizes, then checks the
    exact Boolean system. There is no SAT/SMT engine or theorem prover here.

## Two dual representations

Every \(f:\{0,1\}^n\to\{0,1\}\) has a unique polynomial in each of two bases:

- the **algebraic normal form** (ANF / Reed-Muller) over GF(2), natural for the
  \(\{0,1\}\) codomain and for AND/XOR logic:
  \[ f(x) = \bigoplus_{S\subseteq[n]} a_S \prod_{i\in S} x_i, \qquad a_S\in\{0,1\}; \]
- the **Walsh-Fourier expansion** over the reals in the \(\{\pm1\}\) ("spin")
  basis \(\chi_S(x)=\prod_{i\in S}(-1)^{x_i}\), natural for noise/sensitivity:
  \[ f(x) = \sum_{S\subseteq[n]} \hat f(S)\,\chi_S(x). \]

```python
from omnibias.boolean import (
    truth_table_from_callable, anf_from_truth_table, anf_to_string,
    walsh_spectrum, influences,
)

AND = truth_table_from_callable(lambda a, b: a & b, 2)
XOR = truth_table_from_callable(lambda a, b: a ^ b, 2)

anf_to_string(anf_from_truth_table(AND))   # 'x0*x1'
anf_to_string(anf_from_truth_table(XOR))   # 'x0 + x1'
walsh_spectrum(XOR)[frozenset({0, 1})]     # 1.0  (XOR is a single Walsh character)
influences(AND)                            # [0.5, 0.5]
```

This is the encoding choice behind the two quantizer twins in `omnibias-binary`:
`binarize` lands in \(\{-1,+1\}\) (Walsh-natural, XOR = product), while
`binarize01` lands in \(\{0,1\}\) (ANF-natural, AND = product).

## The discrete <-> continuous derivative bridge

The **multilinear extension** of \(f\) is the unique multilinear polynomial that
agrees with \(f\) on the cube vertices:
\[ F(x) = \sum_{S\subseteq[n]} m_S \prod_{i\in S} x_i, \qquad x\in[0,1]^n. \]

The coefficient \(m_S\) is exactly a **mixed partial derivative** of \(F\):
\[ m_S = \left.\frac{\partial^{|S|} F}{\prod_{i\in S}\partial x_i}\right|_{x=0}, \]
because \(F\) is degree-1 in each variable. Two consequences make the discrete
spectrum a *derivative read-off*:

1. \(\partial F/\partial x_i = F|_{x_i=1} - F|_{x_i=0}\): the continuous partial is
   the arithmetic **Boolean difference**;
2. \(a_S = m_S \bmod 2\): the integer Mobius transform reduces to the GF(2) ANF.

So one multivariate jet of \(F\) yields the whole spectrum. The differentiable
engine builds that jet with the omnibias multi-index machinery
(`identity_jet` + the truncated Cauchy product) and reads it off with
`jet_partials` -- and it matches `_core` to machine precision:

```python
import torch
from omnibias.boolean import multilinear_coeffs            # exact (pure python)
from omnibias.boolean.torch.ops import mobius_coeffs        # differentiable jet read-off

OR = truth_table_from_callable(lambda a, b: a | b, 2)
multilinear_coeffs(OR)                                      # (0, 1, 1, -1) -> x0 + x1 - x0*x1
mobius_coeffs(torch.tensor(OR, dtype=torch.float64))        # tensor([ 0., 1., 1., -1.])
```

The Boolean derivative and its integral close the calculus. Integration recovers
\(f\) only up to a free function independent of the integration variable -- the
discrete "constant of integration":

```python
from omnibias.boolean import boolean_derivative, boolean_derivative_reduced, boolean_integral

boolean_derivative_reduced(AND, 0)        # (0, 1)  ==  d(x0 & x1)/dx0 = x1
g = boolean_derivative(XOR, 0)            # full n-var table, constant in x0
anti = boolean_integral(g, 0)             # antiderivative; anti.general(c) adds the free constant
```

## Worked example: solving `x1 & x2 = y` for `x2`

Not every Boolean equation has a unique solution. Writing the equation as
\(\varphi = 0\) with \(\varphi = (x_1\wedge x_2)\oplus y\) and the cofactors
\(E_0=\varphi|_{x_2=0}\), \(E_1=\varphi|_{x_2=1}\):

- the **eliminant** \(E_0\wedge E_1\) is the condition under which *no* value of
  \(x_2\) works; here it is \(y\wedge\neg x_1\), so the consistency condition is
  \(y\Rightarrow x_1\);
- the **reproductive general solution** is
  \[ x_2 = \neg E_1 \wedge (E_0 \vee c), \]
  with a free parameter \(c\) -- the Boolean "constant of integration." It returns
  the forced value where \(x_2\) is determined and \(c\) where \(x_2\) is free, and
  setting \(c\) to any actual solution reproduces it.

```python
from omnibias.boolean import equation_from_callables, eliminant, solve_for

# variables (x0, x1, x2) play the roles (x1, x2, y).
eq = equation_from_callables(lambda x1, x2, y: x1 & x2, lambda x1, x2, y: y, 3)

elim = eliminant(eq, 1)          # condition: no x2 works  (== y AND NOT x1)
sol = solve_for(eq, 1)           # reproductive solution for x2
assert sol.satisfies(eq)         # plugging it back solves the equation everywhere consistent
```

Contrast XOR: `x1 ^ x2 = y` is uniquely invertible (`x2 = y ^ x1`, no free
parameter), because XOR is a bijection in each argument. AND destroys
information, so its inverse needs the parameter \(c\). For whole systems,
`solve_system` performs Boole's successive elimination and exposes the full
solution space; for the XOR-linear case the solver's `gf2_solve` fast-path is
Gaussian elimination over GF(2).

## Where this is used

- **Automatic test-pattern generation (ATPG):** the Boolean difference
  \(\partial f/\partial x_i\) is exactly the fault-sensitization condition for a
  stuck-at fault on line \(x_i\).
- **Differential cryptanalysis:** per-bit Boolean derivatives and the Walsh
  spectrum quantify how input differences propagate (linearity / nonlinearity of
  S-boxes).
- **Reed-Muller codes & logic synthesis:** the ANF is the Reed-Muller code word;
  algebraic degree controls circuit depth and code distance.
- **Spectral learning:** influences and the degree profile are the differentiable
  design targets in [`omnibias.boolean.torch.ops.design`](../api/boolean.md).

See the [RSA-limitation cookbook](rsa-limitation.md) for an explicit *negative*
result: why this differentiable front-end does **not** shrink the factoring
search space.

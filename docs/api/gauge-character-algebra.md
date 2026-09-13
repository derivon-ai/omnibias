# Central SU(2) character enclosures

The pure-Python module `omnibias.geometry.gauge.transfer.character_algebra`
propagates exact rational character polynomials and explicit analytic
errors. It supports pointwise products, Haar convolution, derivative norms,
logarithms, and the potential induced by a positive vacuum.
These operations are conditional on the supplied enclosure; the module
does not certify that an arbitrary input is a quantum vacuum.

## Exact polynomial operations

A key \(\ell\) is a **doubled spin**: its character is
\(\chi_{\ell/2}\), with dimension \(d_\ell=\ell+1\).
Coefficients must be integers or `Fraction` values; floats and booleans
are refused.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_character_product, su2_character_convolution,
    su2_character_norm, su2_character_laplacian,
    su2_character_log_bound, su2_character_potential_bound,
    su2_character_even_tail, su2_character_round,
)

assert su2_character_product({1: 1}, {1: 1}) == {0: Q(1), 2: Q(1)}
assert su2_character_convolution({1: 1}, {1: 1}) == {1: Q(1, 2)}
assert su2_character_laplacian({1: 1}) == {1: Q(-3, 4)}
assert su2_character_norm({1: Q(1, 100)}) == Q(2, 25)
```

Products retain every term in the SU(2) fusion rule
\[
\chi_{a/2}\chi_{b/2}
=\sum_{\ell=|a-b|,\,|a-b|+2,\ldots,a+b}\chi_{\ell/2}.
\]
Central convolution uses normalized Haar measure:
\[
(f*h)_\ell=\frac{f_\ell h_\ell}{\ell+1}.
\]
These are different operations.

## A norm that controls derivatives

For integer \(s\ge0\), define
\[
\|f\|_{\mathcal A_s}
=\sum_{\ell\ge0}|f_\ell|(\ell+1)^{s+1}.
\]
This is a unital Banach algebra. Indeed
\(\ell+1\le(a+1)(b+1)\) on the fusion support and
\(\sum_\ell(\ell+1)=(a+1)(b+1)\), so
\(\|fh\|_{\mathcal A_s}\le\|f\|_{\mathcal A_s}\|h\|_{\mathcal A_s}\).

With fundamental Casimir \(3/4\), the same norm gives
\[
\|f\|_\infty\le\|f\|_{\mathcal A_0},\qquad
\|\nabla f\|_\infty\le\tfrac12\|f\|_{\mathcal A_1},\qquad
\|\Delta f\|_\infty\le\tfrac14\|f\|_{\mathcal A_2}.
\]
The gradient bound follows by contracting against each unit Lie-algebra
direction; it does not introduce a factor for the Lie-algebra dimension.
The Laplacian bound uses its exact character eigenvalues.

## Complete Taylor tails

`su2_character_even_tail` requires a precise support premise:
an even analytic series \(F(g)=\sum_mg^{2m}F_{2m}\) has character
support \(\ell\le m\) at degree \(2m\), and its complex-circle \(L^1\)
norm is at most \(M\) at radius \(\rho\).
Then
\[
\left\|\sum_{m\ge K}g^{2m}F_{2m}\right\|_{\mathcal A_s}
\le M\sum_{m\ge K}P_s(m)x^m,\qquad
P_s(m)=\sum_{d=1}^{m+1}d^{s+2},\quad x=(|g|/\rho)^2.
\]
Here `first_omitted=K` indexes the even series, not the power of \(g\).
The implemented rational generating functions for \(s=0,1,2\) are
\[
\sum_{m\ge0}P_s(m)x^m
=\frac{Q_s(x)}{(1-x)^{s+4}},\qquad
Q_0=1+x,\quad Q_1=1+4x+x^2,\quad Q_2=1+11x+11x^2+x^3.
\]
Removing the initial finite sum is exact rational arithmetic.

```python
# The theta-cell analytic vacuum supplies circle norm M=4 at rho=3/32.
tail = su2_character_even_tail(
    Q(4, 361) / Q(3, 32), first_omitted=9, circle_norm=4,
)
assert Q(2098, 10**15) < tail < Q(2099, 10**15)
```

The support and circle premises must be proved for the input model.
They are available for the actual two-plaquette vacuum from its complete
magnetic recurrence and Riesz-contour estimate. An arbitrary \(L^2\)
remainder alone does not justify this pointwise upgrade.

## Logarithm and induced Hamiltonian

Suppose the actual density \(w\) satisfies
\(\|w-q\|_{\mathcal A_s}\le\varepsilon\), and put
\(b=\|q-1\|_{\mathcal A_s}\).
When \(b+\varepsilon<1\), the logarithm polynomial
\[
L_J(q)=\sum_{j=1}^J\frac{(-1)^{j+1}}j(q-1)^j
\]
has the rigorous error
\[
\|\log w-L_J(q)\|_{\mathcal A_s}
\le\frac{\varepsilon}{1-b-\varepsilon}
+\frac{b^{J+1}}{(J+1)(1-b)}.
\]
`su2_character_log_bound` returns this polynomial and bound. An invalid
unit-ball criterion raises `ValueError`.

```python
density = {0: Q(1), 1: Q(1, 100)}
log_density, log_error = su2_character_log_bound(
    density, error=Q(1, 10**8), order=5,
)
log_vacuum = {spin: c / 2 for spin, c in log_density.items()}
potential, potential_error = su2_character_potential_bound(
    log_vacuum, error=log_error / 2, kinetic=6,
)
assert log_error > 0 and potential_error > 0
assert 1 in potential
```

For the exact \(S=\tfrac12\log w\), the potential
\[
V=k\bigl(\Delta S+|\nabla S|^2\bigr)
\]
makes \(\exp S\) a zero-energy vacuum of \(-k\Delta+V\).
The exact polynomial gradient square is calculated from
\[
\Gamma(S,S)=\tfrac12\Delta(S^2)-S\Delta S.
\]
If \(\|S-S_*\|_{\mathcal A_2}\le e\) and
\(h=\|S_*-(S_*)_0\|_{\mathcal A_2}\), the returned potential
has **supremum norm** error at most
\[
\frac{k e}{4}(1+2h+e).
\]
This is not an \(\mathcal A_2\) potential-error bound: the Laplacian
uses two derivative weights. The scalar potential coefficient is retained
to fix the zero-vacuum-energy convention.

## Error-controlled compression and composition

`su2_character_round` rounds each coefficient down to a dyadic grid and
returns the exact error in the requested norm. The caller must add this
error to its existing enclosure. Exact integral coefficients, including
a unit Haar mean, are preserved.

```python
rounded, compression_error = su2_character_round(
    {0: 1, 1: Q(1, 3)}, bits=8,
)
assert rounded == {0: Q(1), 1: Q(85, 256)}
assert compression_error == Q(1, 96)
```

For mean-zero inputs, central convolution has the stronger estimate
\[
\|f*h\|_{\mathcal A_s}
\le2^{-s-2}\|f\|_{\mathcal A_s}\|h\|_{\mathcal A_s}.
\]
If \(w,q\) both have Haar mean one and
\(\|w-q\|_{\mathcal A_2}\le\varepsilon\), then
\[
\|w*w-q*q\|_{\mathcal A_2}
\le\frac{\varepsilon(2\|q-1\|_{\mathcal A_2}+\varepsilon)}{16}.
\]
This can propagate actual independent-cell marginal enclosures.
For cells with shared interactions, independence is an additional
premise requiring proof; single-cell marginals do not determine the
joint holonomy law.

Character expansions and exact blocking have an established literature.
For example, [Liu et al., *Exact blocking formulas for spin and gauge
models*](https://arxiv.org/abs/1307.6543) derive blocking formulas including
three-dimensional SU(2) gauge theory. The functions here implement the
specified algebra and error estimates; their use in a new physical
refinement still requires the model-specific analytic premises.

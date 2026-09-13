# Joint SU(2) vacuum enclosures

`omnibias.geometry.gauge.transfer.joint_vacuum` provides exact rational
Fourier norms, complete support tails, and density comparison bounds on
two group coordinates. These are conditional arithmetic helpers. The
caller must supply a proved analytic vacuum source and its errors.

## Basis and norm

For admissible doubled spins \((a,b,c)\), use
\[
B_{a,b,c}(U,V)=
\frac{\operatorname{Tr}[P_c(D_{a/2}(U)\otimes D_{b/2}(V))]}{c+1}.
\]
Its squared Haar norm is \(1/[(a+1)(b+1)(c+1)]\). Orthogonal
total-spin projectors give the exact matrix Fourier norm
\[
\|f\|_{\mathfrak A_s}=\sum_{a,b,c}(1+a+b)^s|f_{a,b,c}|.
\]
This jointly invariant basis includes correlations which separate
central character series do not represent. No numerical SVD is used.

`su2_theta_product_characters` expresses the pointwise product
\(u(U)v(V)\) in this basis, with coefficient \((c+1)u_av_b\).
It is not Haar convolution.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_theta_norm,
    su2_theta_product_characters,
    su2_theta_product_reference_bound,
    su2_theta_vacuum_tail,
)

product = su2_theta_product_characters({1: 1}, {1: 1})
assert product == {(1, 1, 0): Q(1), (1, 1, 2): Q(3)}
assert su2_theta_norm(product) == 36
```

## Complete joint tail

Suppose the actual mean-one analytic vacuum has
\(\|\psi_n\|_2\le M\rho^{-n}\) and complete support \(a+b\le n\).
Then its omitted coefficients obey
\[
\left\|\sum_{n\ge K}g^n\psi_n\right\|_{\mathfrak A_s}
\le M\sum_{n\ge K}t^n(n+1)^s\binom{n+3}{3},
\qquad t=|g|/\rho<1.
\]
`su2_theta_vacuum_tail` evaluates this infinite sum rationally for
`norm_order` 0, 1 or 2. `first_omitted` is the ordinary coupling
order; it does not index an even series.

```python
tail = su2_theta_vacuum_tail(
    Q(4, 361) / Q(1, 6), first_omitted=17, circle_norm=9,
)
assert 0 < tail < Q(35227, 10**18)
```

The physical seven-edge electric metric satisfies
\(\|C_{\rm phys}f\|_\infty\le\|f\|_{\mathfrak A_2}\).
The factor \(1/4\) for the standard product-group Casimir cannot be
copied to that physical metric.

## Product of the actual density's own marginals

`su2_theta_product_reference_bound` takes a real mean-one polynomial
\(p\), a proved \(\mathfrak A_s\) vacuum error, and its separate
\(L^2\) error. It constructs \(q=(\int_Vp)(\int_Up)\) and the normalized
product comparison density \(\sigma=q^2/\int q^2\). Neither comparison
factor is identified with an actual marginal.

Exact Gram norms and character fusion give
\(\|\rho-\sigma\|_{\mathfrak A_s}\le\delta\).
Writing \(b=\max(\|\sigma_U-1\|_{\mathfrak A_s},
\|\sigma_V-1\|_{\mathfrak A_s})\), disjoint Fourier sectors imply
\[
\|\rho-\rho_U\rho_V\|_{\mathfrak A_s}
\le\max\{\delta,b\delta+\delta^2/4\}.
\]
The corresponding keys are `actual_joint_to_product_reference_error`
and `actual_joint_to_own_marginals_error`. All returned values are
exact fractions; the helper sets no physical or formal verification flag.

The ensemble-laws consumer checks the complete theta vacuum source
before using these bounds. At \(\kappa=19\), order 16 and circle \(1/6\),
it gives an actual connected-density \(\mathfrak A_2\) bound below
\(148602/10^9\), while separate moment intervals prove nonfactorization.
These are fixed-graph statements.

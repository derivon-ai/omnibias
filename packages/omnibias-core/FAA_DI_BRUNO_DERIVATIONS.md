# Faà di Bruno jet derivations

Math, conventions, and numerical notes for the exact multi-layer *directional
jet* primitive. The combinatorics live in
[`omnibias.core.bell`](src/omnibias/core/bell.py) (pure Python); the backend
kernels are [`omnibias.jax.jet`](../omnibias-jax/src/omnibias/jax/jet.py) and its
bit-identical torch twin [`omnibias.torch.jet`](../omnibias-torch/src/omnibias/torch/jet.py).

## Conventions

A *jet* is the truncated Taylor expansion of a scalar-parametrised path. We store
Taylor coefficients

\[
  a_k = \frac{f^{(k)}(0)}{k!}, \qquad k = 0, \dots, N,
\]

along the leading axis of an array of shape `(N+1, ...)`, broadcasting over the
trailing axes (hidden units, output components, batch). The conversions
`tower_to_jet` / `jet_to_tower` apply the `k!` scaling to/from the *derivative
tower* `(f, f', f'', ...)`. The derivative-tower convention matches the
scalar-curve series returned by `jax.experimental.jet`, which is used as a
validation oracle.

## Affine layer

For `u = W z + b`, the map is linear in the path parameter `t`, so jets transform
per order:

\[
  u^{[k]} = W\,z^{[k]} \quad (k = 0, \dots, N), \qquad u^{[0]} \mathrel{+}= b.
\]

This is the same observation that makes the *single-layer* derivative trivial:
along `x(t) = x_0 + t v`, the pre-activation is `z(t) = W x_0 + b + t\,W v`, whose
only nonzero jet coefficients are orders 0 and 1. Hence a one-layer field
reproduces `sigma^{(k)}(z_0)\,(W v)^k` (the closed form already used by the
omnibias one-layer fields), which the test-suite checks as a reduction.

## Activation layer (the composition)

For `b(t) = sigma(u(t))` two equivalent kernels are implemented and
cross-validated.

### Bell-polynomial form (oracle)

With the partial (incomplete) exponential Bell polynomials `B_{n,k}`,

\[
  b^{(n)} = \sum_{k=1}^{n} \sigma^{(k)}\!\big(u^{(0)}\big)\,
    B_{n,k}\!\big(u^{(1)}, \dots, u^{(n-k+1)}\big),
\]

\[
  B_{n,k}(x_1,\dots) = \sum \frac{n!}{j_1!\,j_2!\cdots}\,
    \prod_{i\ge 1}\Big(\frac{x_i}{i!}\Big)^{j_i},
  \quad \sum_i j_i = k,\ \sum_i i\,j_i = n,
\]

where `x_i = u^{(i)}` are the *derivatives* of `u`. Each `(j_i)` is a partition
of `n` into `k` parts; the integer coefficient is
`n! / (\prod_i (i!)^{j_i}\, j_i!)`. `omnibias.core.bell.faa_di_bruno_terms(n)`
returns the list of `(k, exps, coeff)` triples; `bell_partial` / `bell_complete`
expose the polynomials, and `bell_number(n) = B_n(1,\dots,1)`.

This form is exact (integer arithmetic) but enumerates partitions, so its term
count grows like `p(n)`; it is the test oracle, not the production path.

### Shifted-power form (production)

Let `w(t) = u(t) - u^{[0]}`, so `w^{[0]} = 0`. Then

\[
  \sigma(u(t)) = \sum_{k=0}^{N}
    \frac{\sigma^{(k)}\!\big(u^{[0]}\big)}{k!}\, w(t)^k,
\]

truncated at order `N`. Because `w` has no constant term, `w^k` starts at order
`k`, so only `k \le N` contribute and the sum is triangular. Series powers are
built by repeated truncated convolution
`(a \star b)[n] = \sum_{i=0}^{n} a[i]\,b[n-i]`. This is `O(N^2)` per element,
numerically stable in float64, and avoids the partition explosion. The exact
`\sigma^{(k)}(u^{[0]})` comes from the omnibias activation fast paths
(Eulerian / Legendre / Hermite recurrences in `omnibias.core.polynomials`), so
the composition is exact, not an autodiff or finite-difference approximation.

## Deep tower and order caps

`mlp_jet(x0, v, layers, order)` seeds the input jet `(x_0, v, 0, \dots)` and
applies (affine, activation) per layer; a `None` activation is a pure affine
readout. Riccati-class activations (`tanh`, `sigmoid`, `softplus`, `gaussian`,
`exp`, `sin`, `cos`, `sinh`, `cosh`) support every order. Bounded-order
activations raise `NotImplementedError` in their fastpath beyond the supported
order; the tower builder wraps that into a `ValueError` naming the activation and
requested order.

## Hessian by polarization

A directional 2-jet gives `v^\top H v` exactly. For a scalar output the full
Hessian follows from
`2\,e_i^\top H e_j = (e_i+e_j)^\top H (e_i+e_j) - e_i^\top H e_i - e_j^\top H e_j`,
each a single order-2 directional jet. This recovers `jax.hessian` to machine
precision and is shown in notebook 23.

## Validation (float64)

| Check | Oracle | Tolerance |
| --- | --- | --- |
| Bell identities, Bell numbers | hand values | exact (int) |
| single-layer reduction | `sigma^{(k)}(z)(W v)^k` | `1e-12` |
| deep MLP tower | `jax.experimental.jet` | `1e-10` |
| deep MLP tower | nested `jax.jacfwd` / `torch.func.jacfwd` | `1e-9` |
| shifted-power vs Bell | the two kernels | `1e-10` |
| cross-backend | jax vs torch | `1e-12` |
| golden regression | `tests/data/faa_di_bruno_mlp_golden.npz` | `1e-12` |

## Multivariate (multi-index) jets

The directional kernel above is the 1-D restriction of the full multivariate
primitive in [`omnibias.jax.jet_mv`](../omnibias-jax/src/omnibias/jax/jet_mv.py)
and its torch twin
[`omnibias.torch.jet_mv`](../omnibias-torch/src/omnibias/torch/jet_mv.py). Instead
of a single series in the path parameter `t`, a *multivariate jet* carries the
whole truncated Taylor expansion of `f: R^D -> R^C` around `x_0`,

\[
  f(x_0 + \delta) = \sum_{|\alpha| \le N} c_\alpha\, \delta^\alpha,
  \qquad c_\alpha = \frac{D^\alpha f(x_0)}{\alpha!},
\]

over multi-indices `\alpha \in \mathbb{N}^D`, with `\delta^\alpha = \prod_i
\delta_i^{\alpha_i}` and `\alpha! = \prod_i \alpha_i!`. One forward pass thus
yields *every* mixed partial up to total order `N` (gradient, full Hessian,
third-order tensors, ...).

### Representation and combinatorics

Coefficients `c_\alpha` are stored densely along the leading axis in the
canonical order of `omnibias.core.multi_index.multi_indices(D, N)` (sorted by
`|\alpha|`, then lexicographically), so torch and jax agree row-for-row. There
are `\binom{D+N}{D}` rows. The pure-Python core supplies:

* `multi_indices(D, N)` / `index_position(D, N)` -- canonical ordering and its
  inverse;
* `multiply_table(D, N)` -- the truncated Cauchy-product table
  `(a*b)_\gamma = \sum_{\alpha+\beta=\gamma} a_\alpha b_\beta` (the multivariate
  replacement for the 1-D convolution);
* `multi_index_factorial(\alpha)` -- `\alpha!`, converting coefficients to raw
  partials `D^\alpha f = \alpha!\,c_\alpha`.

### Kernel

Affine maps act per coefficient (degree-1 in `\delta`), the bias on the constant
row only. The activation composition reuses the **same** shifted-power identity

\[
  \sigma(u) = \sum_{k=0}^{N} \frac{\sigma^{(k)}(u^{[0]})}{k!}\,(u - u^{[0]})^k,
\]

but the powers `(u - u^{[0]})^k` are built with the multivariate truncated
product from `multiply_table` rather than 1-D convolution. The exact tower
`\sigma^{(k)}(u^{[0]})` again comes from the omnibias fast paths, so the whole
expansion is exact. The seed is the identity jet `x(\delta) = x_0 + \delta`
(constant row `x_0`, unit-multi-index rows the standard basis vectors). The
extractors `jet_partials`, `jet_gradient`, `jet_hessian` read off raw
derivatives.

### Directional restriction (cross-check)

The two primitives are tied together exactly: restricting the multivariate jet
to a line recovers the directional tower,

\[
  \frac{d^k}{dt^k}\,f(x_0 + t v)\Big|_{t=0}
    = \sum_{|\alpha| = k} \frac{k!}{\alpha!}\, D^\alpha f(x_0)\, v^\alpha,
\]

which the test-suite checks against the (independently validated) directional
`mlp_jet`.

### Multivariate validation (float64)

| Check | Oracle | Tolerance |
| --- | --- | --- |
| multi-index combinatorics | binomial counts, brute-force product | exact (int) |
| full mixed-partial tensor | nested `jax.jacfwd` / `torch.func.jacfwd` | `1e-9` |
| capstone (depth 4, `N=5`, `C=2`) | nested AD, all `\binom{8}{3}` indices | `1e-8` |
| gradient / Hessian extraction | `grad` / `hessian` | `1e-10` |
| directional restriction | directional `mlp_jet` | `1e-9` |
| cross-backend | jax vs torch | `1e-12` |
| golden regression | `tests/data/faa_di_bruno_mv_golden.npz` | `1e-12` |

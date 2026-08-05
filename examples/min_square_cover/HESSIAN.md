# Closed-form Hessian of the coverage energy

This note derives the **exact** Hessian of the soft-coverage energy with respect to the
square centers (and, as an extension, the existence gates), reducing every quantity to
the Riccati sigmoid tower so it needs no autodiff. It is the mathematical specification
behind `coverage_energy_hessian` in [API.md](API.md), and the curvature the second-order
optimizer arms in [PLAN.md](PLAN.md) exploit.

The single structural fact that makes the result clean: the soft box is built purely from
sigmoids, so all its center-derivatives are closed-form polynomials in `sigmoid`; and the
soft-OR union is **multilinear** in the per-shape memberships, so the union's own second
derivative in any single membership vanishes.

## 1. Notation

- Grid pixels indexed by `p = (i, j)`; the 1-pixel set is `Omega_1 = { p : I(p) = 1 }`.
- Square `k` has center `c_k = (x_k, y_k)`, side `A`, sharpness `beta`, gate
  `alpha_k = sigmoid(a_k) in [0, 1]`.
- `sigma` is the logistic sigmoid; its derivatives are the Riccati polynomials
  `sigma' = P_1(sigma) = sigma (1 - sigma)` and
  `sigma'' = P_2(sigma) = sigma (1 - sigma)(1 - 2 sigma)`, evaluated by
  [riccati_sigmoid_derivative](../../packages/omnibias-binary/src/omnibias/binary/torch/ops/quantize.py)
  with coefficients from
  [sigmoid_polynomial_coeffs](../../packages/omnibias-core/src/omnibias/core/polynomials.py).

## 2. The 1-D soft box and its center derivatives

For a single axis with coordinate `t` and center `c`, define

\[
b(t;c)=\sigma(u_{lo})-\sigma(u_{hi}),\qquad
u_{lo}=\beta\!\left(t-c+\tfrac{A}{2}\right),\quad
u_{hi}=\beta\!\left(t-c-\tfrac{A}{2}\right).
\]

Since `du_lo/dc = du_hi/dc = -beta`,

\[
b'(t;c)=\frac{\partial b}{\partial c}=-\beta\big[\sigma'(u_{lo})-\sigma'(u_{hi})\big],
\qquad
b''(t;c)=\frac{\partial^2 b}{\partial c^2}=\beta^{2}\big[\sigma''(u_{lo})-\sigma''(u_{hi})\big].
\]

Both `b'` and `b''` are evaluated with a single pass of the order-1 and order-2 Riccati
derivatives at `sigma(u_lo)` and `sigma(u_hi)`. As `beta -> inf`, `b -> 1` on the interval
`(c - A/2, c + A/2)` and `b', b''` concentrate at the two edges (a fixed-integral bump
approaching a Dirac derivative), the same soft-to-hard limit that
[BetaAnnealScheduler](../../packages/omnibias-binary/src/omnibias/binary/schedule.py)
anneals along.

## 3. The 2-D occupancy and its per-shape Hessian

Occupancy separates across axes, `m_k(i,j) = b_x(i) \, b_y(j)` with
`b_x(i) = b(i; x_k)` and `b_y(j) = b(j; y_k)`. Hence

\[
\frac{\partial m_k}{\partial x_k}=b_x'\,b_y,\quad
\frac{\partial m_k}{\partial y_k}=b_x\,b_y',
\]
\[
\frac{\partial^2 m_k}{\partial x_k^2}=b_x''\,b_y,\quad
\frac{\partial^2 m_k}{\partial y_k^2}=b_x\,b_y'',\quad
\frac{\partial^2 m_k}{\partial x_k\,\partial y_k}=b_x'\,b_y'.
\]

So the per-shape `2 x 2` occupancy Hessian `H^{(m)}_k` is assembled from the three
factored products `{b_x, b_x', b_x''}` and `{b_y, b_y', b_y''}` (this is
`soft_box_hessian` in [API.md](API.md); the `D`-dimensional generalisation is the obvious
separable one, with off-diagonal blocks being products of two order-1 axis derivatives and
diagonal blocks carrying one order-2 axis derivative).

## 4. The soft-OR union and its first and second derivatives

Let `P = prod_k (1 - alpha_k m_k)` and write `P_{\k}` for the leave-one-out product
(over `l != k`) and `P_{\{k,l}}` for the leave-two-out product (over indices not in
`{k, l}`). Coverage is `C = 1 - P`. Because `P` is **multilinear** in the memberships
`m_k`,

\[
\frac{\partial C}{\partial m_k}=\alpha_k\,P_{\setminus k},\qquad
\frac{\partial^2 C}{\partial m_k^2}=0,\qquad
\frac{\partial^2 C}{\partial m_k\,\partial m_l}=-\,\alpha_k\alpha_l\,P_{\setminus\{k,l\}}\quad(k\neq l).
\]

Numerically `P_{\k} = P / (1 - alpha_k m_k)` and
`P_{\{k,l}} = P_{\k} / (1 - alpha_l m_l)` with a guarded division (or an explicit
prefix/suffix-product scan), so the whole cache is `O(K)` per pixel rather than `O(K^2)`.
These are the fields of `CoverageCache` in [API.md](API.md).

## 5. Energy, gradient, and the center Hessian

The energy is

\[
E(\theta)=\sum_{p\in\Omega_1}\ell\big(C_p\big)+\lambda\sum_k\alpha_k,
\]

where `ell(C) = L(1 - C)` is the per-pixel coverage penalty (`L = softplus` or a squared
hinge), so `ell'` and `ell''` are its derivatives with respect to `C`. The count term is
independent of the centers, so it does not enter the center Hessian (it appears only in the
gate block, section 7).

Gradient with respect to a center coordinate `xi in {x_k, y_k}`:

\[
\frac{\partial E}{\partial \xi}
=\sum_{p\in\Omega_1}\ell'(C_p)\,\frac{\partial C_p}{\partial m_k}\,\frac{\partial m_k}{\partial \xi}
=\sum_{p\in\Omega_1}\ell'(C_p)\,\alpha_k P_{\setminus k}\,\frac{\partial m_k}{\partial \xi}.
\]

Hessian, **same square** `k` (the `2 x 2` diagonal block), for `xi, zeta in {x_k, y_k}`
(the `d^2 C / d m_k^2 = 0` term drops):

\[
\boxed{\;\frac{\partial^2 E}{\partial \xi\,\partial \zeta}
=\sum_{p\in\Omega_1}\Big[\ell''(C_p)\,\big(\alpha_k P_{\setminus k}\big)^2\,
\partial_\xi m_k\,\partial_\zeta m_k
+\ell'(C_p)\,\alpha_k P_{\setminus k}\,\partial^2_{\xi\zeta} m_k\Big]\;}
\]

Hessian, **different squares** `k != l` (an off-diagonal `2 x 2` block), for
`xi in {x_k, y_k}`, `zeta in {x_l, y_l}`:

\[
\boxed{\;\frac{\partial^2 E}{\partial \xi\,\partial \zeta}
=\sum_{p\in\Omega_1}\Big[\ell''(C_p)\,\alpha_k\alpha_l\,P_{\setminus k}P_{\setminus l}
-\ell'(C_p)\,\alpha_k\alpha_l\,P_{\setminus\{k,l\}}\Big]\,
\partial_\xi m_k\,\partial_\zeta m_l\;}
\]

The full center Hessian is the `2K x 2K` symmetric matrix of these blocks. Every factor
(`b, b', b''`, the products `P, P_{\k}, P_{\{k,l}}`, and `ell', ell''`) is closed form, so
no autodiff is used at any point.

## 6. Gauss-Newton connection

Take the squared-hinge / least-squares form: residual `r_p = 1 - C_p` (optionally weighted
`sqrt(w_p)`), objective `E = (1/2) sum_p w_p r_p^2`. As a function of `C_p`,
`ell(C_p) = (1/2) w_p (1 - C_p)^2`, so `ell''(C_p) = w_p` and `ell'(C_p) = -w_p r_p`.

- The `ell''` terms in section 5 are exactly the Gauss-Newton matrix `J^T W J`, where
  `J_{p, xi} = d r_p / d xi = - d C_p / d xi` is the coverage Jacobian. This block is PSD.
- The `ell'` terms (the ones carrying `partial^2 m_k` and the union cross-curvature
  `partial^2 C / partial m_k partial m_l`) are precisely the residual curvature that
  Gauss-Newton **drops**.

This is why the two optimizer families in [PLAN.md](PLAN.md) see different curvature:
`CubicGaussNewton` / `GaussNewton`
([optim.py](../../packages/omnibias-torch/src/omnibias/torch/optim.py) L1839 / L584)
consume `coverage_residual` and implicitly use `J^T W J`; `CubicNewton` /
`TrustRegionNewtonCG` (L1747 / L2534) consume `coverage_energy` and use the full Hessian
above (matrix-free), which `coverage_energy_hessian(..., gauss_newton=False)` reproduces in
closed form and `gauss_newton=True` truncates to the PSD part.

## 7. Gate extension

With `alpha_k = sigmoid(a_k)`, the membership enters the union linearly through
`s_k = alpha_k m_k`, so the union derivatives mirror section 4:

\[
\frac{\partial C}{\partial \alpha_k}=m_k P_{\setminus k},\qquad
\frac{\partial^2 C}{\partial \alpha_k^2}=0,\qquad
\frac{\partial^2 C}{\partial \alpha_k\,\partial \alpha_l}=-\,m_k m_l\,P_{\setminus\{k,l\}}\quad(k\neq l),
\]

and `P_{\k}` is independent of `alpha_k` (it excludes `k`). Chaining through
`alpha_k = sigmoid(a_k)` (so `d alpha_k / d a_k = sigmoid'(a_k)`,
`d^2 alpha_k / d a_k^2 = sigmoid''(a_k)`) and adding the count term `lambda sum_k sigmoid(a_k)`:

- gate gradient:
  `dE/da_k = sigma'(a_k) [ sum_p ell'(C_p) m_{k,p} P_{\k,p} + lambda ]`.
- gate diagonal:
  `d^2E/da_k^2 = sum_p ell''(C_p) (sigma'(a_k) m_k P_{\k})^2 + sigma''(a_k) [ sum_p ell'(C_p) m_k P_{\k} + lambda ]`.
- gate off-diagonal (`k != l`):
  `d^2E/da_k da_l = sigma'(a_k) sigma'(a_l) sum_p [ ell''(C_p) m_k P_{\k} m_l P_{\l} - ell'(C_p) m_k m_l P_{\{k,l}} ]`.
- mixed center-gate, same square, e.g. `xi = x_k`:
  `d^2E/dxi da_k = sum_p [ ell''(C_p) (alpha_k P_{\k} partial_xi m_k)(sigma'(a_k) m_k P_{\k}) + ell'(C_p) sigma'(a_k) P_{\k} partial_xi m_k ]`.
- mixed center-gate, different squares (`xi = x_k`, gate `l != k`), using
  `d P_{\k} / d alpha_l = - m_l P_{\{k,l}}`:
  `d^2E/dxi da_l = - sigma'(a_l) alpha_k sum_p [ ell''(C_p) P_{\k} P_{\l} - ... ] ... ` follows the same
  two-term (`ell''`, `ell'`) pattern; the `ell'` piece uses
  `d^2 C / d x_k d alpha_l = - sigma'(a_l) alpha_k m_l P_{\{k,l}} partial_{x_k} m_k`.

The full parameter Hessian (`wrt="all"` in `coverage_energy_hessian`) is the symmetric
assembly of the center block (sections 5), the gate block, and these mixed blocks.

## 8. Log-sum-exp union variant

If the soft-OR is replaced by the log-sum-exp smooth-max over the memberships
`s_k = alpha_k m_k`, the union's first and second derivatives come directly from
[logsumexp_jacobian / logsumexp_hessian](../../packages/omnibias-hopfield/src/omnibias/hopfield/torch/ops/hopfield.py):
`dC/ds = softmax(beta_u s)` and `d^2C/ds ds = beta_u (diag(p) - p p^T)` with
`p = softmax(beta_u s)`. Chaining `s_k = alpha_k m_k` through sections 2, 3, and 7 gives the
same block structure; unlike soft-OR, `d^2C/ds_k^2 != 0`, so the same-square block picks up
an extra diagonal term. `coverage_energy_hessian` selects the union via the same flag as
`soft_or_coverage` vs `lse_coverage`.

## 9. Validation gate

`coverage_energy_hessian` is a *closed-form* claim, so it must be checked against
independent oracles (a package test, mirroring the repo's "dense vs matrix-free agree"
discipline):

1. Autodiff: `torch.func.hessian(lambda p: coverage_energy(...))` (double backward) at
   random `theta`, random tiny `K`, several `beta`, both loss types, both union types.
2. Finite differences: central differences on `coverage_energy_grad` (itself checked
   against `torch.func.grad`).
3. Assert `max relative error <= tol` (a few ulps scaled by the grid size); assert the
   Gauss-Newton truncation is PSD; assert symmetry.

## 10. Complexity

With the `CoverageCache` (`P`, `P_{\k}`) computed once per pixel in `O(K)`, assembling the
dense `2K x 2K` center Hessian costs `O(|Omega_1| K^2)` (the cross-square blocks dominate).
For the wide regime the same formulas give a matrix-free Hessian-vector product in
`O(|Omega_1| K)` per probe (accumulate `sum_k (row contribution) . v_k` without
materialising the matrix), which is what the drop-in optimizers use; the dense closed form
is reserved for the small-image second-order arms and the autodiff cross-check.

# omnibias-fractional derivations

Definitions, discretisations, and the numerical-error budget for the fractional
operators. The package ships **two honest classes** of operator, and the
project's honesty constraint requires the distinction wherever it is described:

- **Grid / spectral** (Grünwald–Letnikov, Riemann–Liouville, Caputo, spectral):
  **non-local numerical approximations**, **not** closed-form sigma-tower
  derivatives -- accuracy is set by the grid / spectrum (the sections below).
- **Closed-form analytic** (`fractional_derivative` / `mlp_fractional_derivative`):
  a genuine closed form, but on the *analytic-function class* only -- an order-`N`
  truncation of the gamma-ratio Taylor-jet series, exact for polynomials of degree
  `≤ N` (see
  [Closed-form fractional derivative of an analytic function (jet)](#closed-form-fractional-derivative-of-an-analytic-function-jet)).

## Grünwald–Letnikov (GL)

\[
    D^\alpha f(x) = \lim_{h\to 0} h^{-\alpha}
        \sum_{k=0}^{\lfloor (x-a)/h\rfloor} (-1)^k \binom{\alpha}{k} f(x - kh).
\]

The weights `w_k = (-1)^k C(alpha, k)` are built by the stable recurrence
`w_0 = 1`, `w_k = w_{k-1} (1 - (alpha+1)/k)`. The discrete operator is a
lower-triangular Toeplitz matrix; accuracy is first order, `O(h)`.

## Riemann–Liouville and Caputo

For `0 < alpha < 1` the Caputo derivative removes the boundary term:
\[
    {}^C D^\alpha f = {}^{RL} D^\alpha\big(f(x) - f(0)\big).
\]
We discretise both with GL; `caputo` subtracts `f[0]` first. The package exposes
`riemann_liouville` as the GL discretisation directly.

### Analytic check

For a power law (`x > 0`),
\[
    D^\alpha x^p = \frac{\Gamma(p+1)}{\Gamma(p+1-\alpha)}\,x^{p-\alpha}.
\]
The test asserts `< 2%` relative error on an interior window at `n = 4000`
(consistent with the `O(h)` GL rate; the grid edges are least accurate).

## Spectral (FFT) fractional derivative

On a periodic domain of period `L` sampled at `n` points,
\[
    D^\alpha f = \mathcal{F}^{-1}\big[(i k)^\alpha\,\hat f\big],
\]
with the zero mode dropped. For integer `alpha` the real part recovers the
ordinary derivative; the operator is spectrally accurate for band-limited inputs.
The semigroup property `D^{alpha}(D^{alpha} f) = D^{2 alpha} f` holds exactly
(the multipliers multiply), validated to `atol=1e-9`.

## Closed-form fractional derivative of an analytic function (jet)

The operators above are grid-based; omnibias also provides a **genuinely
closed-form** fractional derivative on the analytic-function class, built from
the Taylor jet omnibias already produces. For a jet `a_k = f^(k)(a)/k!` about a
terminal `a`, with `t = x - a`,
\[
    {}_a D_x^{\alpha} f(x)
        = \sum_{k} a_k \,\frac{\Gamma(k+1)}{\Gamma(k+1-\alpha)}\,(x-a)^{k-\alpha}.
\]
This is exactly the power-law rule `D^alpha t^p = Gamma(p+1)/Gamma(p+1-alpha)
t^{p-alpha}` applied term by term to the local Taylor model.

- **Riemann–Liouville** sums over all `k`; **Caputo** drops `k < ceil(alpha)`
  (so it is regular at the terminal -- the Caputo derivative of a constant is
  `0`, whereas the RL derivative of a constant `c` is `c t^{-alpha}/Gamma(1-alpha)`).
- **`alpha = 0`** gives the ratio `1` and powers `t^k`, i.e. `f` itself. **Integer
  `alpha = m`** recovers `f^(m)`: for `k < m` the argument `k+1-m` hits a Gamma
  pole, the ratio is `0`, and the surviving terms are the Taylor series of
  `f^(m)` (valid for `t > 0`).
- **Branch point.** The power `t^{k-alpha}` requires `t >= 0`; for non-integer
  `alpha` the RL form is singular at `t = 0` (use Caputo or `t > 0`).
- **Truncation.** It is an order-`N` local Taylor model: exact for polynomials of
  degree `<= N`, otherwise accurate within the radius of convergence. Validated
  against the Grünwald–Letnikov grid path on an interior window (`< 2%`).

### Real gamma ratio via `lgamma` + sign

`Gamma` is negative on `(-1, 0), (-3, -2), ...`, and torch has no
`torch.special.gamma`, so both backends compute the ratio as
\[
    \frac{\Gamma(k+1)}{\Gamma(k+1-\alpha)}
        = \mathrm{sign}\,\cdot\,\exp\!\big(\ln\Gamma(k+1) - \ln\Gamma(k+1-\alpha)\big),
\]
with `lgamma = log|Gamma|` and `sign = (-1)^{ceil(alpha-k-1)}` for a negative
argument (`+1` otherwise). Using `gammaln` + an explicit real sign (rather than a
complex `gamma`) keeps torch and JAX bit-identical. It is differentiable in
`alpha` (through `lgamma`) and in `a_k`, so the order is learnable and the op
composes with `mlp_jet` and `LearnableOrder`; the only unstable point is the
alpha-gradient *exactly* at an integer order (`0 * inf` at a pole), which the
non-integer regime and `LearnableOrder`'s open band avoid.

At the terminal (`t = 0`) the masked / vanishing terms (`ratio = 0`: Caputo
below `ceil(alpha)`, or a Gamma pole) are forced to contribute exactly `0` rather
than the indeterminate `0 * inf` (`0^{-alpha} = inf`) -- so Caputo evaluates to
its regular terminal value `0` while the genuinely singular RL terminal term is
preserved. All `t > 0` values are unchanged.

### Field fractional partial and fractional-diffusion residual

The same closed form lifts onto the `omnibias-fields` `FieldState`
(`omnibias.fractional.<backend>.field`). For a field `u` and an axis `i` with
lower terminal `a`, `field_fractional_partial` expands the **per-axis** Taylor jet
about the terminal -- the field is re-evaluated with axis `i` pinned to `a` (the
other axes held), giving `a_k = ∂_{x_i}^k u(…, a, …)/k!` -- and sums the
gamma-ratio series at the collocation points `x_i` (offset `t = x_i - a`),
contracted **batch-aligned** (each jet column pairs with its own `t`). It is the
fractional twin of a per-axis field `derivative`: closed form on the analytic
class (order-`N` truncation, exact for degree-`≤ N` polynomial slices),
differentiable in `alpha` and the field parameters, requiring a field with a
closed-form per-axis tower. A plain-number integer order is steered to the exact
derivative tower (avoiding the Gamma pole).

`fractional_diffusion_residual` composes it into the space-fractional diffusion
PINN residual `u_t - Σ_a {}_a D_{x_a}^{alpha_a} u - s` (Caputo by default; one
order per spatial axis), returning a `(residual, diag)` pair. It is validated by
the method of manufactured solutions: for a polynomial field with an analytic
source the residual is `0` to float64 (`atol 1e-10`).

## Differentiability w.r.t. the order

Both the grid and spectral operators are smooth functions of the order `alpha`, so
a tensor-valued `alpha` (e.g. an `nn.Parameter`) trains end-to-end. omnibias keeps
the numpy kernels for a Python-`float` `alpha` (byte-identical to before) and takes
an in-backend path when `alpha` is a tensor:

- **GL weights.** The recurrence `w_k = w_{k-1} (1 - (alpha+1)/k)` is a degree-`k`
  polynomial in `alpha`, so `dw_k/dalpha = (dw_{k-1}/dalpha)(1 - (alpha+1)/k)
  - w_{k-1}/k`. Combined with `d/dalpha h^{-alpha} = -ln(h) h^{-alpha}` this gives
  an exact `d(D^alpha f)/dalpha`. The weights are computed in-backend as a
  `cumprod`, so autograd supplies this gradient directly.
- **Spectral multiplier.** `d/dalpha (ik)^alpha = (ik)^alpha ln(ik)`; the op is
  evaluated as `exp(alpha·ln(ik))` with the zero mode masked before the `ln`, so
  the gradient is exact and `nan`-free.

`alpha = 0` recovers the identity exactly (the GL weights collapse to
`[1, 0, ...]`), and the gradient is defined there too -- the operator interpolates
smoothly across orders. torch and JAX agree on both values and the order gradient
(`rtol=1e-9` / `1e-6` in float64).

## Numerical notes / cross-backend

- All weights/multipliers are built once in numpy float64 and converted to the
  backend tensor type, so torch and jax agree to `rtol=1e-9` (looser than the
  `1e-12` of the closed-form ops: FFT and the GL matmul accumulate rounding).
- The spectral op returns a complex tensor; take `.real` for integer orders.

## References

- Podlubny, *Fractional Differential Equations*.
- Oldham & Spanier, *The Fractional Calculus*.

# omnibias-fractional

Fractional calculus in **two classes**: grid / spectral operators (Grünwald-
Letnikov, Riemann-Liouville, Caputo, FFT) that are *non-local numerical
approximations*, and a *closed-form* analytic fractional derivative on the
analytic-function class (an order-`N` truncation).

!!! warning "The grid / spectral operators are not closed-form"
    Fractional derivatives are **non-local**. The Grünwald–Letnikov, Riemann–
    Liouville, Caputo and spectral operators below are grid-based numerical
    approximations, **not** the exact closed-form sigma-tower derivatives the
    other omnibias packages provide. See
    [`FRACTIONAL_DERIVATIONS.md`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-fractional/FRACTIONAL_DERIVATIONS.md)
    for the error budget. The [analytic operator](#analytic-closed-form-fractional-derivative)
    *is* closed form, on the analytic-function class.

## Kernels

::: omnibias.fractional._core
    options:
      show_root_heading: false
      heading_level: 3

## Ops (torch)

The grid / spectral operators (non-local approximations).

::: omnibias.fractional.torch.ops.fractional
    options:
      show_root_heading: false
      heading_level: 3

## Analytic (closed-form) fractional derivative

Distinct from the grid ops above, this operator is **closed form on the
analytic-function class**: it evaluates the gamma-ratio series

\[
    {}_a D_x^{\alpha} f(x) = \sum_{k} a_k \,\frac{\Gamma(k+1)}{\Gamma(k+1-\alpha)}\,(x-a)^{k-\alpha}
\]

directly from a Taylor jet `a_k = f^(k)(a)/k!` (hand-built, or produced by
`omnibias.torch.jet.mlp_jet`). It is a single vectorised sum -- no grid, no
history -- differentiable in both the order `alpha` (via `lgamma`) and the jet
coefficients, so it composes with autograd and `LearnableOrder`. Riemann–
Liouville sums over all `k`; Caputo drops `k < ceil(alpha)` (regular at the
terminal). `alpha = 0` recovers `f`; an integer order recovers the ordinary
derivative for `t > 0`. It is an order-`N` truncation: exact for degree-`≤ N`
polynomials, otherwise accurate within the Taylor radius. Require `t = x - a ≥ 0`
(the branch point; RL is singular at `t = 0` for non-integer `alpha`).

`piecewise_fractional_derivative` extends the operator beyond a **single** Taylor
radius: given per-patch jets expanded about a sequence of terminals `a_0 < a_1 <
…`, it evaluates each `x` against its owning patch (hard selection by default, or
a soft sigmoid blend near patch boundaries). Riemann–Liouville is singular at a
patch's lower terminal, so its evaluation points are clamped by a small `gap`;
continuity across a patch boundary is only achievable with `kind="caputo"` (which
is regular at the terminal). This turns the single-terminal analytic operator into
a **piecewise-analytic** one for functions represented by a chart of local jets.

::: omnibias.fractional.torch.ops.analytic
    options:
      show_root_heading: false
      heading_level: 3

The JAX twin `omnibias.fractional.jax.ops.analytic` mirrors this surface
(values and the order gradient agree with torch to `rtol=1e-9` / `1e-6`).

## Special functions

Differentiable truncated series for the transcendental functions that express the
closed-form fractional derivatives of specific activations: the two-parameter
**Mittag-Leffler** `E_{alpha,beta}`, the **polylogarithm** `Li_s`, the **Lerch
transcendent** `Phi(z, s, a)`, and the **lower incomplete gamma** `gamma(s, x)`.
Each is a plain vectorised sum -- differentiable in its arguments and cross-backend
bit-identical -- intended for bounded arguments (the truncation is honest, not a
globally-convergent evaluator).

::: omnibias.fractional.torch.ops.special
    options:
      show_root_heading: false
      heading_level: 3

The JAX twin `omnibias.fractional.jax.ops.special` mirrors this surface.

## Activation-specific closed forms

For a handful of activations the fractional derivative has a genuine closed form in
the special functions above, registered on `ACTIVATION_FRACTIONAL` and dispatched by
`activation_fractional_derivative(name, x, alpha=…)`. `exp_fractional` uses the
Mittag-Leffler identity `D^alpha e^{lambda x} = x^{-alpha} E_{1,1-alpha}(lambda x)`
(Riemann–Liouville) / `E_{1,1}` shifted (Caputo); `cosh_fractional` and
`sinh_fractional` are the numerically-stable even/odd combinations of `exp_fractional`
at `lambda = ±1`. These are exact on the analytic-function class (differentiable in
`alpha`), distinct from the grid operators.

::: omnibias.fractional.torch.ops.activation
    options:
      show_root_heading: false
      heading_level: 3

The JAX twin `omnibias.fractional.jax.ops.activation` mirrors this surface.

## Non-periodic spectral operators

The periodic FFT spectral path assumes the signal wraps; `spectral_fractional_laplacian`
applies the exact fractional-Laplacian symbol `|xi|^alpha` on a **bounded, non-periodic**
interval using an orthonormal sine (**DST-I**, Dirichlet BC) or cosine (**DCT-II**, Neumann
BC) transform, so eigenmodes are reproduced without the periodic-wrap artefact.
`windowed_spectral_fractional` instead applies a `tukey_window` taper and the periodic path,
for signals that decay at the boundary. Both are differentiable in `alpha`.

::: omnibias.fractional.torch.ops.spectral
    options:
      show_root_heading: false
      heading_level: 3

The JAX twin `omnibias.fractional.jax.ops.spectral` mirrors this surface.

## Trainable layers

`nn.Module` wrappers so a fractional operator drops into a network as a layer with a
**learnable order** (previously only `LearnableOrder` was a Module): `GrunwaldLetnikovLayer`
(grid GL), `SpectralFractionalLayer` (periodic FFT), and `SpectralFractionalLaplacianLayer`
(non-periodic DST/DCT). Each keeps `alpha` in a stable band via the sigmoid
reparametrisation and is autograd-trainable end-to-end.

::: omnibias.fractional.torch.layers
    options:
      show_root_heading: false
      heading_level: 3

The JAX twins in `omnibias.fractional.jax.layers` mirror these as
`register_pytree_node_class` pytrees (order carried as a differentiable leaf).

## Field fractional partial + fractional-diffusion residual

The closed-form operator lifts onto the `omnibias-fields` `FieldState`:
`field_fractional_partial` is the fractional twin of a per-axis field
`derivative`. It expands the field's Taylor jet **along one axis about the lower
terminal `a`** (re-evaluating the field with that axis pinned to `a`) and sums the
gamma-ratio series at the collocation points -- closed form on the
analytic-function class, differentiable in `alpha` and the field parameters.
`fractional_diffusion_residual` composes it into the space-fractional diffusion
PINN residual `u_t - Σ_a {}_a D_{x_a}^{alpha_a} u - s`. This module needs
`omnibias-fields` (the `[torch]` / `[jax]` extras) and is imported lazily, so the
`ops` surface keeps its `omnibias-fields`-free install.

::: omnibias.fractional.torch.field
    options:
      show_root_heading: false
      heading_level: 3

The JAX twin `omnibias.fractional.jax.field` is bit-identical (parity to
`rtol=1e-9`).

## Learnable order

Every operator accepts `alpha` as either a Python `float` (the fast numpy kernel,
unchanged) or a **backend tensor**. A tensor takes an in-backend, autograd-friendly
path, so the fractional *order itself is differentiable and learnable* -- for
example an `nn.Parameter`, or an order discovered by a PINN / the neural-jet
engine. The Grünwald-Letnikov weights are then built by a `cumprod` recurrence and
the spectral multiplier as `exp(alpha·log(ik))` (zero mode masked), both smooth in
`alpha`. `LearnableOrder` keeps the order inside a stable band via a sigmoid
reparametrisation.

::: omnibias.fractional.torch.order
    options:
      show_root_heading: false
      heading_level: 3

The JAX twin `omnibias.fractional.jax.order` exposes `init_order` /
`constrain_order` (JAX keeps parameters as plain leaves).

## JAX twin

The JAX backend (`omnibias.fractional.jax.ops`) mirrors the torch surface;
cross-backend agreement (values and the order gradient) is asserted to `rtol=1e-9`
/ `1e-6` in float64.

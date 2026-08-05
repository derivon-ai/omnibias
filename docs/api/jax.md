# omnibias-jax

JAX backend for omnibias.

## Top-level API

::: omnibias.jax
    options:
      show_root_heading: false
      heading_level: 3

## Activation registry

::: omnibias.jax.activations
    options:
      show_root_heading: false
      heading_level: 3

## Closed-form integral transforms

Bit-identical twin of
[`omnibias.torch.transforms`](torch.md#closed-form-integral-transforms):
closed-form Laplace / Fourier / Mellin transforms of the activations whose
transform is itself elementary, resolved through the shared identity table in
[`omnibias.core.transforms`](core.md#integral-transform-identities).

Every kernel is a pure array expression and is safe under `jit`, `grad` and
`vmap`. Two JAX-specific points: `fermi_dirac_mellin` *is* differentiable here
(`jax.scipy.special.zeta` defines a gradient rule where `torch.special.zeta`
does not), and its `Re(s) > 1` scope check reads a concrete value, so it is
skipped under tracing -- the same trade-off `omnibias.measure.jax.integraleq`
makes for its solvability check. Validate in eager mode; the arithmetic itself
traces either way.

`TransformBlock` is the functional counterpart of the torch module: `init()`
hands back a parameter pytree and `apply(params, x)` evaluates
`T[sigma](scale * x + shift)`, so it composes with any JAX optimizer without a
module system.

::: omnibias.jax.transforms
    options:
      show_root_heading: false
      heading_level: 3

## Closed-form Laplacian primitives

::: omnibias.jax.laplacian
    options:
      show_root_heading: false
      heading_level: 3

## Faà di Bruno jets

::: omnibias.jax.jet
    options:
      show_root_heading: false
      heading_level: 3

## Multivariate (multi-index) jets

::: omnibias.jax.jet_mv
    options:
      show_root_heading: false
      heading_level: 3

## Born-Oppenheimer derivatives

::: omnibias.jax.bo_derivatives
    options:
      show_root_heading: false
      heading_level: 3

## Architectures

::: omnibias.jax.architectures
    options:
      show_root_heading: false
      heading_level: 3

## Optimisers (Gauss-Newton / energy natural gradient, natural / Riemannian gradient)

Alongside the Gauss-Newton / Levenberg-Marquardt core, `natural_gradient_direction` solves
the metric-preconditioned system `(M + damping I) delta = g` on any dense `(P, P)` metric and
`natural_gradient_step` applies the update `theta - lr (M + damping I)^{-1} g` -- the
bit-identical functional twin of the torch `NaturalGradient`. Two closed-form metrics plug in:
the Gauss-Newton **Fisher** `(1/N) J^T J` (`gauss_newton_fisher`; Newton on a residual linear
in `theta`) and the **geometry pullback** `g = J^T h J`
(`omnibias.geometry.jax.ops.pullback_metric`).

::: omnibias.jax.optim
    options:
      show_root_heading: false
      heading_level: 3

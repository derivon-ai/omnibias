# omnibias-binary

Closed-form quantization gradients for binary / ternary / `k`-bit neural-network
training. The forward pass is a hard quantizer (`sign`, a ternary deadzone, or a
uniform `k`-bit staircase); the backward pass is the **exact** derivative of the
smooth `tanh(beta z)` surrogate via the omnibias Riccati identity
`tanh'(z) = 1 - tanh(z)^2` (coefficients from
`omnibias.core.tanh_polynomial_coeffs`, shared by every backend -- no per-backend
fork). As `beta -> infinity` the surrogate converges to the hard quantizer while
the gradient stays well defined.

`binarize01` (alias `heaviside`) is the `{0, 1}` codomain twin of `binarize`,
backed by the Eulerian `sigmoid_polynomial_coeffs` Riccati tower
(`sigmoid'(z) = s(1-s)`); it is affinely conjugate to `binarize`
(`binarize01(z, beta) == (binarize(z, beta / 2) + 1) / 2`). Use the `{0,1}` twin
for AND/OR/Reed-Muller-style logic (see `omnibias-boolean`) and the `{-1,+1}`
`binarize` for XOR/Walsh-style logic.

## Ops (torch)

::: omnibias.binary.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## Beta-annealing scheduler

Beyond STE: anneal the surrogate sharpness `beta` (soft-to-hard curriculum), use
the curvature-aware `surrogate_jet` / `binarize_curvature` backward, and learn
`beta` end-to-end. See the [better-than-STE cookbook](../cookbook/better-than-ste-binary.md).

::: omnibias.binary.schedule
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.binary.jax.ops`) is the twin built with
`jax.custom_vjp`; the tests assert torch/jax gradient agreement to `rtol=1e-9`
in float64.

Status: Alpha (`0.1.0a1`).

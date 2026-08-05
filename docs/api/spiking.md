# omnibias-spiking

Spiking-neuron LIF / IF primitives whose forward spike is a hard Heaviside
threshold and whose backward uses an **exact closed-form surrogate gradient** --
the derivative of an omnibias dictionary activation (`fast_sigmoid` via
`sigmoid_polynomial_coeffs`, or a Gaussian bump via `hermite_coeffs`), not an
ad-hoc surrogate. The surrogate is selectable by name or by an
`omnibias.core.ActivationSpec`.

## Ops (torch)

::: omnibias.spiking.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.spiking.jax.ops`) is the twin; forward spikes match
exactly and the surrogate gradients agree with the torch `autograd.Function`
backward to `rtol=1e-9` in float64.

Status: Alpha (`0.1.0a1`).

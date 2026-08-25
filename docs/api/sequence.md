# Causal transverse filter (05-02 G5)

A designed causal FIR whose taps are one closed-form `sigma^(n)` pack.
Default `order=0` is the logistic survival / integral-role tail
`c * sigma(tau - alpha k)` — the leaky-integrator class. Pack order is a
band selector (01-07): `n=1` (`sigma'`) is a mid-lag bump and is the
wrong kernel for an AR(1) / S4D comparison.

Inference is `O(T K)` convolution, not a recurrence. Gate G5 asks for an
`R^2` match within `0.02` of a named S4D (Gu et al. 2022, `N=1`) at four
parameters, five seeds, worst-seed. Width equals the horizon so FIR
truncation is not an extra handicap. Founding tower taps (`delta -> 0`),
not temperature collapse. Not `omnibias.struct`.

```python
import torch
from omnibias.torch.sequence import CausalTransverseFilter

filt = CausalTransverseFilter.from_leaky_integrator(0.95, width=8)
x = torch.zeros(1, 8)
x[0, 0] = 1.0
y = filt(x)
assert y.shape == (1, 8)
```

## Core taps

::: omnibias.core.sequence
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch module

::: omnibias.torch.sequence
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.sequence
    options:
      show_root_heading: false
      heading_level: 3

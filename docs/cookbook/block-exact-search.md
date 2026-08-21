# Block exact search

A last-linear, OMBU-bias, or arrangement-normal block is a sparse
direction. Spec 03-12 finds the step. Last-layer least squares is
exactly quadratic. This is theory spec 08-07: a coordinate sweep, not
a global solver and not CCF stretch.

## Last-layer least squares (spec §5)

```python
import torch
from omnibias.torch.optim_block_search import block_exact_search

torch.set_default_dtype(torch.float64)

def loss(params):
    p = params.reshape(-1)
    return (p[0] * 1.0 + p[1] * 0.5 - 1.0) ** 2

new, result = block_exact_search(
    loss,
    torch.tensor([0.0, 0.0]),
    mask=(True, False),
    exact_quadratic=True,
)
assert result.fell_back is False
assert abs(result.step - 1.0) <= 1e-12
assert abs(float(new[0]) - 1.0) <= 1e-12
assert result.actual_value is not None and abs(result.actual_value) <= 1e-12
```

`verify=True` is the default. A step that would raise the true loss
shrinks toward zero. Named blocks (`last_linear_block`,
`ombu_bias_block`, `arrangement_w_block`) build the mask from a flat
layout; they do not import `omnibias.tab`.

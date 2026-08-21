# Depth-causal local jet

A two-layer readout can take a local Gauss–Newton step on a named
residual before the rest of the net is jointly trained. Spec 08-03
is a warm start, not a replacement for backprop and not CCF stretch.
`compose_jet` is the chain rule. Bias collapse (`delta -> 0`)
supplies the tower.

## Spec §5: one hidden unit, local readout

```python
import torch
from omnibias.core.local_jet import LocalJetConfig
from omnibias.torch.train_local import local_jet_step

torch.set_default_dtype(torch.float64)

layers = [
    (torch.tensor([[0.5]]), None, "tanh"),
    (torch.tensor([[0.2]]), None, None),
]
cfg = LocalJetConfig(n_directions=1, damping=0.0, variant="readout")
new_layers, report = local_jet_step(
    layers,
    torch.tensor([1.0]),
    config=cfg,
    target=torch.tensor([1.0]),
)
v = new_layers[1][0]
w = new_layers[0][0]
yhat = float(v * torch.tanh(w))
assert abs(yhat - 1.0) <= 1e-12
assert report.greedy_only_claimed_optimal is False
assert report.n_params == 2
```

`n_directions >= n_params` raises `LocalJetForbidden` unless
`allow_full=True`. A 2-parameter toy therefore uses
`n_directions=1`. Invert-and-match is only for strictly monotone
`sigma` (tanh / sigmoid); GELU raises.

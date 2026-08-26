# Closed-form one-layer weight-space loss jet

Evaluate `σ^(n)` once from the shared tower, then assemble
`φ^(k)(0) = d^k L(θ + s d) / ds^k` at `s = 0`. The identities below
are the chain rule, not a skip of it. One-layer only.

```python
from omnibias.core.weight_loss_jet import (
    honesty_payload,
    one_layer_loss_jet,
    pack_one_layer_params,
    worked_example,
)

ex = worked_example()
assert ex["jet0_matches_loss"] is True
assert ex["jet1_matches_gdotd"] is True
assert ex["jet2_matches_dHd"] is True
payload = honesty_payload()
assert payload["closed_form"] is True
assert payload["skip_chain_rule"] is False
assert payload["full_parameter_jacobian"] is False
assert payload["global_min_claim"] is False

theta = pack_one_layer_params(0.0, (1.0,), (0.0,), ((0.5,),))
direction = pack_one_layer_params(0.0, (0.0,), (0.0,), ((1.0,),))
jet = one_layer_loss_jet(
    ((0.5,), (-0.25,)),
    (0.1, -0.2),
    theta,
    direction,
    order=2,
    hidden=1,
    dim=1,
    activation="tanh",
)
assert len(jet) == 3
assert jet[0] > 0.0
```

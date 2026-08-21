# Certified step

An affine map `f(x) = w x` on `[0, 1]` has Lipschitz `|w|`. Spec 08-09
accepts a trial only when the sealed bound stays inside `P_max`.
Empty enclosures reject; that is not robustness. Bias collapse
(`delta -> 0`) supplies `sigma'` on the Lipschitz path.

## Spec §5: accept 1.8, reject 2.5

```python
from omnibias.core.verified.interval import Interval
from omnibias.verify import Network, affine_layer
from omnibias.verify.train_step import CertifiedStepConfig, certified_accept

net = Network([affine_layer([[1.5]], [0.0])])
cfg = CertifiedStepConfig(
    property="lipschitz",
    p_max=2.0,
    cell=[Interval(0.0, 1.0)],
)
ok = certified_accept(net, (1.8, 0.0), config=cfg)
bad = certified_accept(net, (2.5, 0.0), config=cfg)
assert ok.accepted and ok.reason == "ok"
assert ok.bound_hi is not None and ok.bound_hi <= 2.0
assert not bad.accepted and bad.reason == "violates"
assert bad.bound_hi is not None and bad.bound_hi > 2.0
assert ok.robust_without_enclosure is False
```

A missing cell or an overflowing ReLU stack returns
`reason="vacuous"`, never `accepted=True`.

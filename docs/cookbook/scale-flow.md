# Scale flow and coarse-graining

A pack at `alpha = 1`, order `2` evaluated at `0.4` rescales exactly
to the same pack at `alpha = 2` evaluated at `0.2`.

```python
from omnibias.core.composed_curvature import eval_tanh_derivative
from omnibias.core.scale import ScaledPack, overlap, rescale_pack

pack = ScaledPack(order=2, mean=0.0, alpha=1.0, base="tanh")
got = rescale_pack(pack, 2.0).value(0.2)
assert abs(got - 4.0 * eval_tanh_derivative(0.4, 2)) < 1e-12
back = rescale_pack(rescale_pack(pack, 8.0), 0.125)
assert back.alpha == pack.alpha
```

Gaussian overlaps are a finite exact Gauss–Hermite rule. The
derived curriculum comes from order-as-frequency, not a hand-tuned
window list.

```python
from omnibias.core.scale import ScaledPack, overlap
from omnibias.fields.scale import scale_schedule

slow = ScaledPack(order=1, mean=0.0, alpha=1.0)
fast = ScaledPack(order=1, mean=0.0, alpha=8.0)
corr = overlap(slow, fast, derivative_order=2)
assert abs(corr) < 1.0
alphas = scale_schedule(
    target_band=lambda t: 2.0 + 14.0 * t,
    steps=4,
    base="gaussian",
    order=1,
)
assert len(alphas) == 4
assert alphas[0] < alphas[-1]
```

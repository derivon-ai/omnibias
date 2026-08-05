# Better-than-STE binary training

The straight-through estimator (STE) trains a hard quantizer by *pretending* its
backward is the identity. omnibias does something principled instead: the hard
forward is paired with the **exact gradient of a smooth `tanh(beta z)` surrogate**,
evaluated in closed form via the Riccati identity `tanh'(z) = 1 - tanh(z)^2`
(one `tanh` call, any order). This page shows the three levers that take it past
vanilla STE: beta-annealing, the curvature-aware (jet-STE) backward, and a
learnable `beta`.

!!! abstract "The theory (and certified bounds)"

    The formal story behind this page -- STE as the uniform-kernel corner of a
    closed-form mollifier family, the jet bias-order theorem, the dead-unit
    dichotomy, annealing as graduated non-convexity, and the surrogate ↔ robust-IRLS
    bridge -- is written up in [Differentiable binarization](../theory-binary.md).
    The finite instances of those theorems are *machine-checked*: see
    `omnibias.verify._core.surrogate_bounds` (interval enclosures + a Lean certificate for
    the agreement margin and the no-dead-unit property).

## The surrogate, and why STE is its crudest special case

For `binarize(z) = sign(z)`, the backward multiplies the incoming gradient by
`s'(z) = beta (1 - tanh^2(beta z))`. As `beta -> inf` this concentrates at the
decision boundary (the `2 * delta` Dirac limit) -- exactly the saturation the
Phase 1 `lim` operator records on the activation spec. STE replaces `s'(z)` with
the box `1_{|z|<1}`; the omnibias surrogate is smooth, exact, and `beta`-tunable.

```python
import torch
from omnibias.binary.torch import ops as q

z = torch.linspace(-2.0, 2.0, 9)   # pre-activations

q.binarize(z, beta=10.0)        # hard forward {-1,+1}, smooth-surrogate backward
q.surrogate_tower(z, 10.0, 3)   # [s, s', s'', s'''] of tanh(beta z), one tanh eval
```

!!! warning "Scale `beta` to your activations — the one mistake that sinks BNN training"

    The surrogate gradient `beta (1 - tanh^2(beta z))` is a bump of height `beta`
    and **width `~1/beta`**. A unit only learns while its pre-activation sits inside
    that window. After `BatchNorm` the pre-activations are roughly unit-scale, so the
    default `beta=10` makes the window *ten times too narrow*: the vast majority of
    units land in the flat tails, receive a **zero** gradient, and die. The symptom
    is a binary net that trails even the crude STE.

    Pick `beta` so `1/beta` matches the spread of `z` (often `beta ~ 1` right after a
    `BatchNorm`), anneal *gently*, or use a heavy-tailed kernel (below) that never
    sends an exactly-zero gradient. The `examples/binary_vs_ste` benchmark includes
    an `omnibias_b10` arm precisely to show this failure — and the `beta~1` arms
    overtaking the STE once the window is scaled correctly.

## Lever 1 - beta-annealing (soft-to-hard curriculum)

Raising `beta` over training interpolates from an easy, well-conditioned early
objective to the exact straight-through limit. The scheduler is pure-Python, so
the same object drives torch and jax loops:

<!-- docs-test: skip reason="illustrative training loop; model / criterion / x / y are the reader's own" -->
```python
from omnibias.binary import BetaAnnealScheduler

# Keep beta_end modest: it sets the *final* window width 1/beta_end. Annealing all
# the way to beta=100 reintroduces the dead-unit problem at the end of training.
sched = BetaAnnealScheduler(beta_start=0.5, beta_end=3.0, num_steps=10_000, schedule="exp")
for step in range(10_000):
    beta = sched.step()
    loss = criterion(model(x, beta=beta), y)   # forward stays exactly hard
    ...
```

`schedule` is `"linear"`, `"exp"` (geometric, natural for a multiplicative
sharpness), or `"cosine"`.

## Lever 2 - the jet-STE / curvature-aware backward

The surrogate's **Taylor jet** carries not just the slope but the local
curvature, built with the same `compose_jet` machinery as the Phase 1 limits:

```python
jet = q.surrogate_jet(z, beta=10.0, order=2)   # jet[1]=s'(z), jet[2]=s''(z)/2
```

A 2nd-order-corrected quantizer uses the *windowed-average* slope
`s'(z) + (h^2/6) s'''(z)` (with `h = 1/beta`, the surrogate's natural smoothing
scale) instead of the point slope -- a better effective gradient through the hard
step because it accounts for the finite transition width:

```python
q.binarize_curvature(z, beta=10.0)        # hard forward, curvature-corrected backward
q.curvature_corrected_slope(z, 10.0)      # the corrected slope itself
```

## Lever 3 - a learnable `beta`

`beta` is itself differentiable: the surrogate cotangent
`d/dbeta tanh(beta z) = z (1 - tanh^2(beta z))` lets the network *learn* its own
quantization sharpness. Both backends now support this identically (the JAX
quantizers return the `beta` cotangent, matching the torch `autograd.Function`):

=== "torch"

    ```python
    beta = torch.tensor(4.0, requires_grad=True)
    q.binarize(z, beta).sum().backward()
    beta.grad   # surrogate gradient w.r.t. sharpness
    ```

=== "jax"

    ```python
    import jax, jax.numpy as jnp
    from omnibias.binary.jax import ops as q

    z = jnp.linspace(-2.0, 2.0, 9)
    jax.grad(lambda b: jnp.sum(q.binarize(z, b)))(4.0)
    ```

## Lever 4 - the surrogate kernel is a free choice (the nascent-delta menu)

The hard `sign` is the `beta -> inf` limit of *any* smooth sigmoidal CDF, so the
forward is base-independent. That makes the **backward a free choice of probability
density** `k`, a "nascent delta" whose bandwidth `beta` you set independently. The
library ships the `tanh`/`sigmoid` members (`binarize`/`binarize01`); the
`examples/binary_vs_ste/kernels.py` helper exposes the wider menu, peak-normalised
so `beta` controls only the window *width*:

| kernel | `k(u)`, `u = beta z` | tail | smooth CDF |
| --- | --- | --- | --- |
| `box` | `1_{\|u\|<=1}` | compact | hard-tanh (**STE**) |
| `tanh` | `sech^2 u` | exponential | tanh |
| `logistic` | `4 s(1-s)`, `s=sigmoid(u)` | exponential | logistic |
| `gaussian` | `exp(-u^2/2)` | gaussian | probit |
| `cauchy` | `1/(1+u^2)` | **heavy** `~1/u^2` | arctan |

The tail is what matters for trainability. The STE's compact box gives an *exactly
zero* gradient once `|z| > 1/beta` — those units are stuck forever. The heavy-tailed
`cauchy` kernel never does: a unit far from the boundary still gets a small but
non-zero pull back toward it, so there are no permanently dead units. This is the
same family STE belongs to (it is just the `box` member); omnibias simply lets you
pick a better-conditioned density.

## Backends & honesty

`binarize` / `ternarize` / `kbit_quantize`, the surrogate towers/jets, the
curvature-corrected slope, and the learnable-`beta` gradients are bit-identical
torch/jax twins (float64 parity tests). The forward is always the exact hard
quantizer; only the backward is a (principled, closed-form) surrogate -- this is
an estimator, not a claim that the hard step is differentiable.

## Runnable benchmark: the levers vs vanilla STE

A controlled, apples-to-apples comparison lives at
[`examples/binary_vs_ste`](https://github.com/derivon-ai/omnibias/tree/main/examples/binary_vs_ste).
It trains a binary-weight / binary-activation network on MNIST, Fashion-MNIST and
CIFAR-10 where every arm shares the *exact same hard forward, architecture,
initialisation, optimiser and data order* and differs **only in the backward**:
`ste` (compact-box baseline), `omnibias_b10` / `omnibias_b1` (the shipped
`binarize` at the mis-scaled `beta=10` vs the scaled `beta=1`), the kernel menu
`tanh` / `logistic` / `gaussian` / `cauchy` (all `beta=1`, peak-normalised),
`anneal` (a gentle `0.5 -> 3` curriculum) and `learnable_beta`. It reports
**best-epoch** mean ± std over seeds. The `omnibias_b10` control is expected to
*trail* the STE — that is the dead-unit failure — while the scaled / heavy-tailed
arms pass it. An offline synthetic smoke test runs in CI; the real-data sweep is
one command:

```bash
python -m examples.binary_vs_ste.run_demo \
    --datasets mnist fashion_mnist cifar10 --download --epochs 60 --seeds 0 1 2
```

### Measured results

Best-epoch top-1 accuracy (%, mean over 3 seeds; MLP 30 epochs, CIFAR-10 convnet 40
epochs, Adam, BatchNorm):

| dataset | `ste` | `tanh` (β=1) | `gaussian` | `cauchy` | `anneal` 0.5→3 | `omnibias_b10` (control) |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | 98.08 | 98.07 | 98.02 | 97.93 | 98.07 | 97.60 |
| Fashion-MNIST | 85.64 | 85.56 | 85.12 | 85.06 | **85.65** | 84.48 |
| CIFAR-10 | 82.09 | 81.90 | 82.01 | 81.36 | **82.85** | 75.93 |

(On MNIST the overall best arm is `learnable_beta` at **98.20**, which self-tunes β≈1.)

What this shows — read it honestly:

- **The best omnibias arm ≥ STE on every dataset**, and *wins decisively where it
  matters*: `anneal` beats STE by **+0.76%** on the deep CIFAR-10 net (and has the
  lowest training loss everywhere). β-annealing is graduated non-convexity done
  right (soft→hard), the regime where the closed-form tower pays off.
- **`omnibias_b10` is the dead-unit control** and trails STE everywhere —
  catastrophically on CIFAR-10 (−6%). A too-sharp β (window ≪ data scale) starves
  units; depth amplifies it. This — not the surrogate idea — was the original deficit.
- **At the matched β=1, `tanh`/`omnibias_b1` ties STE** (they are bit-identical: at
  β=1 the exact `β·tanh'(βz)` *is* the peak kernel). Under BatchNorm+Adam, STE is
  already near-optimal in the bulk, so a peak-capped kernel can only tie it there.
- **`cauchy`'s heavy tail helps on shallow MLPs but backfires on the deep convnet**
  (−0.7%): a never-zero gradient keeps saturated units from committing across many
  layers. The dead-unit advantage is regime-dependent.

### Two further levers (`scaled`, `curvature`)

Two arms target the residual gap directly:

- **`scaled` / `scaled_anneal`** standardise the pre-activation by its own std before
  the kernel, so β is *scale-free* like the STE box. One global β otherwise can't fit
  both the unit-scale BatchNorm activations *and* the ~0.03-scale kaiming weights, so
  the weight path silently collapses to plain STE; standardising restores omnibias
  shaping on both.
- **`curvature`** is the jet-STE backward `s'(z) + (h²/6)s'''(z)` — a 4th-order
  windowed-average slope (Theorem 2 in [Differentiable binarization](../theory-binary.md)),
  free because `s'''` is closed-form.

```bash
python -m examples.binary_vs_ste.run_demo --datasets cifar10 \
    --arms ste tanh anneal curvature scaled scaled_anneal --epochs 40 --seeds 0 1 2
```

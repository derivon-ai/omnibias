<!-- SPDX-License-Identifier: Apache-2.0 -->
# Binary neural networks: omnibias surrogate vs the STE baseline

A controlled, apples-to-apples benchmark of *how you back-propagate through a hard
quantizer*. Every arm shares the **exact same hard `sign` forward**, the same
architecture, initialisation, optimiser, learning-rate schedule and data order
(fixed by `seed`); the only thing that changes is the **backward** — the
surrogate-gradient *kernel* and its bandwidth `beta`:

| arm | backward through `sign` (`u = beta z`) | source |
| --- | --- | --- |
| `ste` | compact box `1_{\|z\|<=1}` (hard-tanh STE) | `ste.py` (baseline) |
| `omnibias_b10` | exact `beta * tanh'(beta z)`, `beta=10` | `omnibias.binary.binarize` |
| `omnibias_b1` | exact `beta * tanh'(beta z)`, `beta=1` | `omnibias.binary.binarize` |
| `tanh` | `sech^2 u`, peak-normalised, `beta=1` | `kernels.binarize_kernel` |
| `logistic` | `4 s(1-s)`, `s=sigmoid(u)`, `beta=1` | `kernels.binarize_kernel` |
| `gaussian` | `exp(-u^2/2)`, `beta=1` | `kernels.binarize_kernel` |
| `cauchy` | `1/(1+u^2)` (heavy tails), `beta=1` | `kernels.binarize_kernel` |
| `anneal` | `tanh` kernel, `beta` annealed `0.5 -> 3` | `+ BetaAnnealScheduler` |
| `learnable_beta` | `tanh` kernel, trained bandwidth `beta` | `binarize_kernel(z, beta=Parameter)` |
| `curvature` | jet-STE `s'(z)+(h^2/6)s'''(z)`, `beta=1` | `omnibias.binary.binarize_curvature` |
| `scaled` | `tanh` kernel on standardised `z/std(z)`, `beta=1` | `kernels.binarize_kernel` (scale-aware) |
| `scaled_anneal` | scale-free `tanh`, `beta` annealed `0.5 -> 3` | `+ BetaAnnealScheduler` |

The forward is always the exact hard quantizer, so this measures **which surrogate
gradient trains better** — it is *not* a claim that the hard step is
differentiable. The **temperature-collapse** view (`beta -> inf`, one gate
sharpened into a step -- *not* the founding bias collapse, which coalesces `K`
parallel hyperplanes into one): the hard `sign` is the `beta -> inf` limit of
*any* smooth sigmoidal CDF, so the backward is a free choice of probability
density (a "nascent delta") -- and each such density is the order-1 rung of that
base's bias-collapse derivative tower. The kernel arms walk that menu; `box`
at `beta=1` recovers the STE, and `tanh`+`exact` recovers the shipped `binarize`.

**The headline lesson — scale `beta` to the data.** The surrogate window has width
`~1/beta`. The library default `beta=10` (the `omnibias_b10` arm) makes that window
*ten times narrower than the unit-scale post-BatchNorm pre-activations*, so most
units fall outside it and receive **zero** gradient — they die, and accuracy drops
below the STE. Setting `beta~1` (matching the activation scale), or using a
heavy-tailed `cauchy` kernel that never sends a zero gradient, fixes it.

Weights and activations are both binarized; following standard BNN practice the
first and last layers stay full precision and `BatchNorm` precedes each activation
binarization.

## Layout

```
ste.py          # the STE baseline autograd.Function
kernels.py      # hard-sign forward + selectable surrogate kernels (box/tanh/logistic/gaussian/cauchy)
arms.py         # the arm matrix: name -> per-step quantizer factory
models.py       # BinaryMLP (1-channel) and BinaryConvNet (3-channel)
data.py         # MNIST / Fashion-MNIST / CIFAR-10 loaders + offline synthetic data
train.py        # one fair training loop; only the backward differs per arm
experiment.py   # sweep arms x datasets x seeds -> JSON/CSV + summary table
run_demo.py     # CLI entry point
tests/          # offline synthetic smoke test (no download, CPU, runs in CI)
```

## Quick start

Offline smoke test (synthetic data, no download, CPU) — this is what CI runs:

```bash
python -m pytest examples/binary_vs_ste/tests -q
python -m examples.binary_vs_ste.run_demo --synthetic --epochs 2 --seeds 0
```

Real MNIST + Fashion-MNIST (downloads once into `./data`):

```bash
python -m examples.binary_vs_ste.run_demo \
    --datasets mnist fashion_mnist --download --epochs 15 --seeds 0 1 2
```

Full kernel-menu sweep on all three datasets on a **GPU node** (CIFAR-10 needs one):

```bash
python -m examples.binary_vs_ste.run_demo \
    --datasets mnist fashion_mnist cifar10 \
    --download --epochs 60 --seeds 0 1 2 --device cuda
```

Restrict the arms with `--arms` (e.g. `--arms ste omnibias_b10 tanh cauchy anneal`)
to keep the more expensive CIFAR-10 sweep cheap. The `scaled` / `scaled_anneal`
arms standardise each pre-activation by its own std before the kernel, so one global
`beta` fits both the unit-scale activations *and* the `~0.03`-scale weights (without
it the weight path collapses to plain STE); `curvature` is the jet-STE backward
`s'(z)+(h^2/6)s'''(z)`, a 4th-order-accurate windowed slope that the closed-form
`s'''` makes free.

Two equal-opportunity ablation levers (they lift every arm, so the surrogate
comparison stays fair): `--xnor-scale` adds the XNOR-Net per-filter weight scale
`alpha=mean|W|`, and `--lr-schedule cosine` decays the LR (which tames the
final-epoch noise of sharp surrogates). Both tables print best-epoch *and*
final-epoch accuracy.

Datasets cache under `--data-root` (default `./data`) and result artifacts
(`results.json`, `results.csv`) under `--out-dir`
(default `examples/binary_vs_ste/results/`); both are git-ignored. Pre-download the
data on a node with network access first if your GPU cluster's compute nodes are
offline — the loaders read the cache and never reach for the network unless
`--download` is passed.

## Reading the results

`run_demo` prints two `dataset x arm` grids of mean ± std top-1 test accuracy: the
**best-epoch** grid (the headline metric, robust to late-training wobble) and the
final-epoch grid. The expected qualitative ranking, once `beta` is scaled to the
data, is roughly

```
omnibias_b10  <  ste  <=  tanh ~ omnibias_b1 ~ logistic ~ gaussian  <=  cauchy ~ anneal
```

with `learnable_beta` competitive, and the gap widening from MNIST to
Fashion-MNIST to CIFAR-10. `omnibias_b10` is the deliberately *mis-scaled* control:
it should **trail the STE**, demonstrating that the failure mode is a `beta` too
large for the activation scale, not the omnibias surrogate itself. If the
correctly scaled arms don't catch and pass the STE, something is mis-wired.

> Note on reproducing GPU runs: cluster-specific submission commands (scheduler,
> queue, host paths) are intentionally **not** checked in. Keep them in the
> separate, private `omnibias_experiments` project, per the repository's
> vendor-neutral policy.

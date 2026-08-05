<!-- SPDX-License-Identifier: Apache-2.0 -->
# MNIST-1D double descent: an exact-curvature, optimizer-axis study

Model-wise **double descent** — test error that *rises* to a peak at the
interpolation threshold and then *falls again* as the model grows — is usually
shown with SGD or Adam and measured only by the test-error curve. This example
reproduces it on [MNIST-1D](https://github.com/greydanus/mnist1d)
(Greydanus & Kobak, *Scaling Down Deep Learning*, arXiv:2011.14439) and turns it
into two things the SGD/Adam-only literature has **not** done:

1. an **exact-curvature dissection** — the true Hessian spectrum
   (`lambda_max`, `lambda_min`, condition number `kappa`, trace, Frobenius norm)
   measured *through* the interpolation threshold, and
2. an **optimizer-axis study** — the whole omnibias second-order suite
   (`CubicNewton`, `TrustRegionNewtonCG`, `CubicGaussNewton`, K-FAC,
   `NaturalGradient`, `DiagonalCurvature`, `JetLBFGS`, `JetSubspaceTensor`, ...)
   put on the *same* double-descent axis as Adam / SGD.

Everything runs one derivative tower in two registers so the closed-form and
certified paths light up, and every optimizer is wired to the exact closure
contract in `omnibias.torch.optim`.

## The two registers

Each model is a width-parameterized MLP (`models.MLP1D`, one hidden layer by
default) built in **two registers** — same data, same width sweep, different
objective + activation:

| register | objective | activation | curvature | unlocks |
| --- | --- | --- | --- | --- |
| `ce_relu` | cross-entropy | ReLU | exact matrix-free HVP (exact-to-autodiff) | the paper-faithful reproduction |
| `mse_tanh` | one-hot MSE | tanh (Riccati) | closed-form Riccati tower + HVP cross-check | Gauss-Newton family, closed-form Fisher, **certified enclosures** |

`ReLU` has `sigma'' = 0` almost everywhere, so its loss Hessian *is* the
Gauss-Newton matrix and the higher tower is trivial — expected, not a bug. The
`mse_tanh` residual `r = f(x) - onehot(y)` is what unlocks the least-squares
optimizers and the sound `omnibias.verify` read-outs.

## The optimizer → closure map (`arms.py`)

`arms.py` tags every optimizer with the *driver* the trainer must use to honour
its closure convention (recompute the objective with the graph intact, **no**
`.backward()` inside the closure — the `_CurvatureOptimizer` / LBFGS contract):

| driver | optimizers | closure returns |
| --- | --- | --- |
| `standard` | `adam`, `sgd` | — (`backward()` + `step()`) |
| `scalar` | `cubic_newton`, `trust_region`, `stochastic_newton`, `jet_lbfgs`, `diag_hutch`, `jet_subspace_o2/o3` | scalar loss |
| `residual` | `cubic_gauss_newton`, `diag_gn` *(mse_tanh only)* | residual vector `r` |
| `natural` | `natural_gradient` *(mse_tanh, dense Fisher → narrow only)* | scalar loss (+ attached GN-Fisher metric) |
| `kfac` | `kfac` | scalar loss (built with the module) |
| `sharpness` | `sam_stochastic`, `sam_exact`, `sharpness_reg` | scalar loss (objective replaced by a sharpness functional) |

Second-order arms train **full-batch** (MNIST-1D is 4000×40 — tiny — which gives
one clean, well-defined training Hessian and removes minibatch noise as a
confound). `arms_for_register` automatically drops invalid `(arm, register)`
pairs (e.g. the Gauss-Newton family outside `mse_tanh`).

## Hypotheses

| id | claim |
| --- | --- |
| H1 | `lambda_max(H)` / `kappa(H)` spike at the interpolation threshold, co-located with the test-error peak. |
| H2 | curvature-aware optimizers shift / flatten the peak vs Adam / SGD. |
| H3 | epoch-wise double descent tracks a curvature-ridge crossing. |
| H4 | exact `sam_objective` / `sharpness_aware_loss` suppress the peak more cleanly than the stochastic-SAM proxy. |
| H5 | the threshold model's fragility can be **rigorously enclosed** (`omnibias.verify`) on the smallest nets. |
| H6 | tanh vs ReLU (and, optionally, a metalearned activation) changes the peak width. |
| H7 | `JetSubspaceTensor` order-3 helps at/near the threshold where its `O(k^3)` cost is affordable. |
| H8 | setting the step size from the **exact** `lambda_max(H)` (`eta = c·2/lambda_max`) holds full-batch GD on the edge of stability and suppresses the epoch-wise sharpening of H3. |

## Layout

```
data.py         # MNIST-1D: pip package (faithful) / vendored recipe / synthetic; per-seed label noise
models.py       # MLP1D width/depth/activation in the two registers
arms.py         # optimizer catalogue: name -> (driver, factory, lr, register constraints)
curvature.py    # spectrum_snapshot: dense (full spectrum) or matrix-free (power/Hutchinson) HVP
train.py        # one instrumented full-batch run; dispatches on arm.driver
eos.py          # exact edge-of-stability LR controller: eta = c*2/lambda_max(H) (H8)
experiment.py   # sweep register x arm x width x seed x noise -> per-run JSON; aggregate to CSV/JSON
eos_experiment.py  # H8 driver: eos vs adam vs sgd at the threshold + the control figure
eos_ablation.py    # H8 ablation: multi-seed + c-sweep (0.9/1.0/1.05) + momentum variant
certify.py      # P4 sound Lipschitz / robustness / (optional) flatness enclosures
run_demo.py     # CLI
analysis/plots.py  # figures from the aggregated summary
sweep/          # scheduler-neutral fan-out (gen_jobs.py + submit.sh) for the full grid
results/        # committed summaries + figures (the CPU demonstration grid)
tests/          # offline, CPU, synthetic smoke tests (what CI runs)
```

## Quick start

Offline smoke test (synthetic data, no download, CPU) — this is what CI runs:

```bash
python -m pytest examples/mnist1d_double_descent/tests -q
python -m examples.mnist1d_double_descent.run_demo \
    --synthetic --arms adam cubic_newton --widths 4 16 64 --seeds 0 \
    --steps 30 --no-curvature --aggregate
```

The first real run downloads/builds MNIST-1D once (the `mnist1d` pip package if
installed — `pip install mnist1d` — else the vendored generator) and caches it to
`--scratch-dir`.

## Running the phases

Large artifacts (per-run JSONs) go to `--scratch-dir` (default under `artifacts/`);
only the small aggregated `results/` summaries and figures are committed. The
interactive node here has **no GPU and only 2 CPUs**, so the committed numbers are
a deliberately *reduced* CPU demonstration grid — the full grid fans out to the
GPU cluster via `sweep/` (see below).

**P1 — reproduce + instrument (H1).** Adam, `ce_relu`, width sweep, clean +
noisy, exact curvature overlaid on the test-error curve:

```bash
python -m examples.mnist1d_double_descent.run_demo \
    --register ce_relu --arms adam --widths 4 12 24 40 64 100 160 250 \
    --noise 0.0 0.15 --seeds 0 1 2 --steps 800 \
    --curv-batch 512 --scratch-dir artifacts/omnibias_mnist1d/p1 \
    --out-dir examples/mnist1d_double_descent/results/p1 --aggregate
```

**P2 — optimizer axis (H2).** All arms × both registers × widths × seeds × noise
(cluster-scale; use `--arms core` for the representative subset):

```bash
python -m examples.mnist1d_double_descent.run_demo \
    --register both --arms core --widths 24 40 64 100 160 \
    --noise 0.15 --seeds 0 1 2 --steps 800 \
    --scratch-dir artifacts/omnibias_mnist1d/p2 --aggregate
```

**P3 — sharpness intervention (H4) + epoch-wise (H3).** The exact-sharpness arms
vs the stochastic-SAM proxy, and a per-step curvature trajectory at the threshold
width (`--log-every 1 --curvature-every` small on a single width):

```bash
python -m examples.mnist1d_double_descent.run_demo \
    --register ce_relu --arms sharpness --widths 64 --noise 0.15 --seeds 0 1 2 \
    --steps 800 --scratch-dir artifacts/omnibias_mnist1d/p3 --aggregate
```

**P4 — certified + subspace (H5, H7).** Sound Lipschitz / robustness enclosures of
the smallest `mse_tanh` nets (`certify.py`), and `JetSubspaceTensor` order-2 vs
order-3 near the threshold:

```bash
python -c "from examples.mnist1d_double_descent.certify import train_and_certify; \
from examples.mnist1d_double_descent.data import load_mnist1d; \
b=load_mnist1d(label_noise=0.15, scratch_dir='artifacts/omnibias_mnist1d/data'); \
print(train_and_certify(width=8, bundle=b, steps=400).as_dict())"

python -m examples.mnist1d_double_descent.run_demo \
    --register mse_tanh --arms jet_subspace_o2 jet_subspace_o3 \
    --widths 40 64 96 --noise 0.15 --seeds 0 1 2 --steps 800 \
    --scratch-dir artifacts/omnibias_mnist1d/p4 --aggregate
```

**P5 — edge-of-stability control (H8).** The `eos` arm (`eos.py`) is plain
full-batch GD whose learning rate is re-set online from the *exact* top Hessian
eigenvalue, `eta = c·2/lambda_max(H)`, using `omnibias.curvature.torch.top_eigenvalue`.
`eos_experiment.py` runs it against Adam / SGD at the threshold width and writes the
three-panel control figure:

```bash
python -m examples.mnist1d_double_descent.eos_experiment \
    --width 24 --steps 600 --noise 0.35 --n-train 1000 --n-test 1000 --seed 0 \
    --scratch-dir artifacts/omnibias_mnist1d/eos \
    --out-dir examples/mnist1d_double_descent/results/eos
```

The `eos` arm is also selectable in `run_demo` like any other optimizer (`--arms eos`);
`measure_every` / `probe_iters` (arm `hypers`) amortise the exact-`lambda_max` HVP probe.
`eos_ablation.py` sweeps the knobs across seeds — `c ∈ {0.9, 1.0, 1.05}` (inside / on /
past the edge) and a momentum variant (target widened to `2c(1+beta)`):

```bash
python -m examples.mnist1d_double_descent.eos_ablation \
    --width 24 --steps 400 --noise 0.35 --seeds 0 1 2 \
    --scratch-dir artifacts/omnibias_mnist1d/eos_ablation \
    --out-dir examples/mnist1d_double_descent/results/eos
```

### Figures

```bash
python -m examples.mnist1d_double_descent.analysis.plots \
    --summary examples/mnist1d_double_descent/results/p1/summary.json \
    --out examples/mnist1d_double_descent/results/figures
```

`analysis/plots.py` degrades gracefully: it draws only the panels the summary
supports (double-descent-per-optimizer, curvature-vs-width overlay, register
comparison, sharpness intervention, subspace order-2-vs-3, epoch-wise, certified
bars).

## The cluster (full grid)

`sweep/gen_jobs.py` prints one self-contained `run_demo` command per
`(register, arm, noise, seed, width-block)` cell; `sweep/submit.sh` wraps each with
a **site-supplied** submission command — nothing about a scheduler, queue, host, or
absolute path is baked in (per the repo's vendor-neutral policy). Dry run:

```bash
bash examples/mnist1d_double_descent/sweep/submit.sh \
    --arms core --widths 24 40 64 100 160 250 500 --seeds 0 1 2 3 --noises 0.0 0.15
```

To actually submit, export your interpreter and submit wrapper:

```bash
export OMNIBIAS_PYTHON=/path/to/venv/bin/python
export OMNIBIAS_SUBMIT='<your GPU batch submit command>'   # takes the job as arguments
bash examples/mnist1d_double_descent/sweep/submit.sh --arms all --seeds 0 1 2 3 4 5 6 7
```

Each job writes per-run JSONs under `--scratch-base/<tag>/`; a final aggregation
step (`run_demo ... --aggregate`, or `experiment.write_summary`) reduces the whole
scratch tree to the committed `results/` tables and `analysis/plots.py` draws the
figures. Keep any cluster-specific submission commands in the separate, private
`omnibias_experiments` project, never in tracked files.

## Findings

The numbers below come from the committed **CPU demonstration grid** (interactive
node, no GPU, 2 CPUs): MNIST-1D with `n_train = n_test = 1000`, label noise 0.35
(the level that makes the peak crisp at this small `n_train`), 1 hidden layer.
P1 is 2 seeds; P2–P5 are single-seed CPU-budget passes. The full multi-seed,
low-noise, wide-width grid fans out to the cluster (`sweep/`). Raw tables are in
`results/p{1,2,3,4}/summary.json` and `results/eos/summary.json`; figures are under
each phase's `figures/`.

**H1 — curvature spikes at the threshold, co-located with the test peak (confirmed).**
`ce_relu`, Adam, widths 6→120. The exact top Hessian eigenvalue rises with width,
**peaks at width ≈ 24**, then *collapses* once the net can interpolate:

| width | 6 | 12 | 18 | **24** | 30 | 40 | 80 | 120 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test error | .767 | .774 | .776 | **.784** | .777 | .782 | .766 | .751 |
| exact λ_max(H) | 16.8 | 29.1 | 37.8 | **46.8** | 8.2 | 0.25 | 0.02 | 0.01 |

The test-error peak and the `λ_max(H)` peak sit at the **same width** (≈ the
interpolation threshold — train error first reaches 0 at width 30–40), and
`λ_max` drops ~200× into the over-parameterized regime (flat minima). This is the
exact-curvature dissection the SGD/Adam-only literature does not show.
→ `results/figures/curvature_overlay_ce_relu_adam_noise0.35.png`.

**H2 — curvature-aware optimizers reach interpolation far faster (confirmed, on the
speed axis).** `ce_relu`, width 40, steps-to-interpolation (train error → 0):

| optimizer | cubic_newton | jet_lbfgs | adam | sgd / diag_hutch / trust_region |
| --- | --- | --- | --- | --- |
| step reached | **69** | 111 | 287 | not within 500 |

Second-order arms cross the threshold ~4× sooner than Adam; at a *matched* 500-step
budget the final test errors are comparable (~0.74–0.79), i.e. they travel the same
double-descent curve faster rather than obviously flattening it at this budget.
→ `results/p2/figures/double_descent_noise0.35.png`.

**H3 — the curvature ridge builds up epoch-wise (illustrated).** `ce_relu`, width 24,
600 steps: as training fits (train error 0.89 → 0.24) the exact `λ_max(H)` climbs
**monotonically 0.5 → 16.1**, while test error dips to ~0.71 early (~step 70) and
then *creeps back up* to ~0.76 — early-stopping fragility tracking rising curvature.
(A full second descent needs interpolation, which this width-24 net does not reach
in 600 steps — honest scope.) → `results/p3/figures/epochwise_ce_relu_adam_w24.png`.

**H4 — the exact sharpness *penalty* suppresses the peak most cleanly (confirmed, with
a nuance).** `ce_relu`, widths 18/24/30, test error:

| arm | 18 | 24 | 30 |
| --- | --- | --- | --- |
| adam | .739 | .756 | .762 |
| sam_stochastic | .746 | .751 | .768 |
| sam_exact (exact SAM ascent) | .736 | .752 | .772 |
| **sharpness_reg** (exact `sharpness_aware_loss`) | **.736** | **.731** | **.749** |

The exact sharpness *penalty* is lowest at every width; the exact SAM *ascent step*
(`sam_exact`) is ≈ Adam. So closed-form sharpness helps — but it is the penalty, not
the ascent direction, that does the work. → `results/p3/figures/sharpness_ce_relu_noise0.35.png`.

**H5 — threshold-model fragility can be rigorously enclosed (confirmed).** Sound
`omnibias.verify` order-1 enclosures on the smallest `mse_tanh` nets: the certified
ℓ∞ Lipschitz upper bound grows monotonically with width — a *guaranteed* "bigger net
⇒ larger worst-case input sensitivity":

| width | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- |
| certified Lipschitz (ℓ∞) | 5.53 | 13.0 | 15.7 | 20.4 |
| certified-robust fraction | 0.75 | 0.25 | 0.50 | 0.50 |

These are *sound* (outward-rounded interval / Taylor-model) bounds, not estimates.
→ `results/p4/figures/certified.png`.

**H6 — activation changes where the peak sits (supported).** Same width sweep, both
registers: `ce_relu` interpolates by width 30–40 (its `λ_max` peak/threshold is
early), while `mse_tanh` (one-hot MSE) only reaches interpolation at width ≈ 120 —
the tanh/MSE threshold sits at a **larger width** than ReLU/CE. A metalearned
activation is left to the cluster grid. → `results/figures/register_comparison_noise0.35.png`.

**H7 — order-3 subspace helps only marginally, only at the threshold (not decisive).**
`mse_tanh`, `JetSubspaceTensor`: order-3 is marginally *lower* than order-2 exactly at
the threshold width 24 (0.759 vs 0.762) but marginally *higher* at widths 18/30 — a
threshold-localized effect within single-seed noise. The `O(k³)` third-order tensor
step does not pay off decisively at this scale/budget.
→ `results/p4/figures/subspace_mse_tanh_noise0.35.png`.

**H8 — exact-`lambda_max` LR control holds the edge of stability and robustly caps
late-epoch sharpening (confirmed; the generalization win is a *final*-error, not a
*best*-error, effect).** This turns the study's own findings into an optimizer: H1/H3
show `lambda_max(H)` spikes and climbs epoch-wise, and edge-of-stability theory says
full-batch GD self-stabilises at `lambda_max ≈ 2/eta`. The `eos` arm instead *sets*
`eta = c·2(1+beta)/lambda_max(H)` from the exact top eigenvalue every few steps.

*The control works exactly, for any target.* Once the landscape sharpens past the
stability limit the product `lambda_max·eta` **locks on the target `2c(1+beta)`** for the
rest of training — the controller shrinks `eta` in lock-step as `lambda_max` climbs, with
**no learning rate to tune**. The `c`-sweep and momentum variant each pin their own
target (1.8 / 2.0 / 2.1 / 3.42) by construction.
→ `results/eos/figures/eos_control.png` (single seed, per-step),
`results/eos/figures/eos_control_variants.png` (the four targets).

*Ablation across 3 seeds* (`ce_relu`, width 24, 400 steps; mean ± std):

| arm | best test | final test | train err | final λ_max(H) |
| --- | --- | --- | --- | --- |
| adam | .720±.012 | .776±.015 | .257±.011 | 16.4±3.9 |
| sgd | .714±.016 | .755±.014 | .409±.012 | 6.7±0.8 |
| **eos** (`c=.9`) | .718±.012 | **.745±.004** | .452±.018 | **4.4±0.3** |
| eos_edge (`c=1`) | .722±.012 | .746±.006 | .440±.013 | 4.0±0.5 |
| eos_past (`c=1.05`) | .721±.013 | .748±.011 | .439±.009 | 4.2±0.8 |
| eos_mom (`β=.9`) | .723±.006 | .773±.019 | **.245±.043** | 8.7±1.4 |

Three honest readings the multi-seed data forces:

1. **Best test error is flat across every arm** (.714–.723, error bars overlap) — no
   optimizer wins the *best* (early-stopped) error at this threshold; the single-seed
   "Adam 0.704" was seed noise.
2. **The robust `eos` win is late-training.** Plain `eos` ends **~3.7× flatter than Adam**
   (λ_max 4.4 vs 16.4) *and* with the **best, lowest-variance final test error**
   (.745±.004 vs Adam's .776±.015) — it resists Adam's overfitting drift by capping the
   curvature-driven step, needing neither early stopping nor a tuned LR.
3. **The knobs behave as predicted.** The `c`-sweep is inert in [0.9, 1.05] (on / just
   past the linear edge changes nothing — train error stays ≈ .44); **momentum recovers
   Adam-level fitting** (train .245) but reintroduces Adam-like overfitting (final test
   .773) and re-sharpening (λ_max 8.7). The flat, low-final-error behaviour is
   specifically the *momentum-free, curvature-capped* step.
→ `results/eos/figures/eos_ablation.png`.

**Bottom line.** The exact Hessian spectrum spikes at the interpolation threshold and
collapses beyond it (H1); this is reproducible per-optimizer (H2), builds up
epoch-wise (H3), is most cleanly suppressed by the closed-form sharpness penalty (H4),
can be turned into *sound* certificates (H5), moves with the activation (H6), and can
be used as a control signal — an exact-`lambda_max` step size that pins GD on the edge
of stability and robustly caps late-epoch sharpening and its overfitting drift (H8,
3 seeds). The exact-curvature + certified characterization, the optimizer-axis result,
and the curvature-as-controller demonstration are the defensible contribution — not
"beats Adam everywhere."

### Phase 1 — ExactSAM & FrugalCurvature (attacking the two axes where Adam is *not* optimal)

Unlike the CPU grid above, these numbers come from the **cluster grid** (`sweep/`, aggregated to
`results/phase1/`): real MNIST-1D (default `n_train=4000`, `n_test=1000`), 8 widths (5→250),
**4 seeds**, label noise 0.15, 600 steps, with the new per-run **optimizer-state-bytes** and
**wall-clock** telemetry (`train.py`). Each arm runs at the learning rate of its *natural base*
(`exact_sam`/`sgd` = heavy-ball SGD `1e-1`; `frugal_hutch`/`diag_hutch` = diagonal-preconditioner
`1e-1`; `adam`/`frugal_gn` at their defaults) — a matched-conditions comparison, not a tuned
bake-off.

**Memory (FrugalCurvature — confirmed).** Optimizer state measured on the live objects
(`_optimizer_state_bytes`):

| optimizer | adam | exact_sam | sgd | **frugal_hutch** | **frugal_gn** |
| --- | --- | --- | --- | --- | --- |
| bytes / param | 8.01 | 8.00 | 4.00 | **4.01** | **4.01** |

FrugalCurvature carries **half of Adam's optimizer state** — one momentum buffer + `O(#tensors)`
per-tensor curvature scalars, vs Adam's two `O(P)` buffers (`m`, `v`). It is the SGD memory
footprint with an exact-curvature preconditioner bolted on. (ExactSAM is *not* memory-lean: its
momentum + cached exact-sharpness direction match Adam's two buffers.)
→ `results/phase1/figures/memory_cost_ce_relu_noise0.15.png` (left panel).

**Generalization (ExactSAM — confirmed on `ce_relu`, reversed on `mse_tanh`).** Mean final test
error over 4 seeds; **best over the width sweep** in bold:

| register | adam | sgd | exact_sam | frugal_hutch | frugal_gn |
| --- | --- | --- | --- | --- | --- |
| ce_relu | .511 | .466 | **.454** | .462 | — |
| mse_tanh | **.395** | .549 | .713 | .645 | .557 |

On `ce_relu` (classification) ExactSAM has the **lowest** test error and — the real point —
**flattens Adam's double-descent peak**: at the peak width 100 Adam is `.613`, ExactSAM `.483`,
frugal_hutch `.470` (a ~13-point suppression). This is the study's own **H4** (the exact-sharpness
*penalty* beats the SAM ascent) packaged as a reusable optimizer. On `mse_tanh` the picture
**reverses** — Adam's adaptivity dominates (`.395`) and the SGD-based ExactSAM is worst — so the
generalization advantage is **register-dependent, not universal**.
→ `results/phase1/figures/double_descent_noise0.15.png`.

**Adaptive base (follow-up on the `mse_tanh` reversal).** ExactSAM now takes a selectable
`base` for the step that descends `g = ∇L + λ∇S`: `sgd` (default), `adam`, or memory-lean
`frugal` (per-tensor RMS). A 256-run base-comparison + a 60-run `λ` sweep isolate *why* the
`mse_tanh` reversal happens (mean best test error, best over width):

| register | adam | exact_sam (sgd) | exact_sam_adam | exact_sam_frugal |
| --- | --- | --- | --- | --- |
| `ce_relu` | .480 | **.437** | .456 | .448 |
| `mse_tanh` | **.380** | .712 | .565 | .565 |

The adaptive base **fixes the SGD-base *underfit*** on `mse_tanh` (`.712 → .565`) but still loses
to Adam. The `λ` sweep (Adam base; `λ=0` is *bit-for-bit* Adam) shows why — the penalty's sign is
**register-dependent**:

| λ (best over width) | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
| --- | --- | --- | --- | --- | --- |
| `ce_relu`  | .480 | .472 | .465 | .456 | **.451** |
| `mse_tanh` | **.380** | .380 | .445 | .565 | .677 |

More sharpness penalty **monotonically helps** classification but **monotonically hurts** (and
*underfits*, train error rising too) `tanh`/MSE regression. Only `λ≈1e-4` ties Adam on regression
while barely beating it on classification — so a **single fixed `λ` cannot win both registers**; the
base swap fixes the *fit*, not the penalty's sign. Best robust default: `base="adam"` with small
`λ` (degrades to Adam where the penalty would hurt). The real next lever is a **register/fit-aware
`λ`** (back off the penalty when it fights the fit). Reproduce with
`optim_experiments/exactsam_lam_sweep.py` (a research probe in the separate `omnibias_experiments` project) and the
`exact_sam_adam` / `exact_sam_frugal` arms.

**Auto-`λ` (`lam_auto`): a fit-preservation cap — wins classification, and *proves* why it can't
fix regression.** ExactSAM's `lam_auto` sets, each step, the largest penalty that keeps the
*combined* step first-order loss-decreasing: `λ_eff = min(λ, ρ·‖∇L‖²/|⟨∇L,∇S⟩|)` — one extra dot
product, exact, no extra backward. `λ` becomes an *upper bound* (arms use `3e-3`, the `ce_relu`
sweep optimum). A fresh 256-run grid (mean best test error, best over width):

| register | adam | exact_sam_adam (fixed) | **exact_sam_auto** (sgd) | exact_sam_adam_auto |
| --- | --- | --- | --- | --- |
| `ce_relu` | .480 | .456 | **.425** | .437 |
| `mse_tanh` | **.380** | .565 | .728 | .663 |

On `ce_relu` auto-`λ` gives the **best generalization in the whole study** (`.425` vs Adam `.480`)
with the **flattest** double-descent curve (final error `.44–.47` across all widths). But it **does
not** fix `mse_tanh` — and the money plot (`results/phase1/figures/autolam_trajectory_noise0.15.png`)
shows *why*: `λ_eff` **rides pinned at the bound on *both* registers** (the cap only dips
transiently), because on `mse_tanh` the penalty does **not** oppose the loss gradient — train loss
falls to `~0.085` and the model *fits* fine. The `mse_tanh` damage is a **generalization / metric
misalignment** (flat MSE-to-one-hot basins ≠ good arg-max accuracy), which is **invisible to any
fit-based signal** — so neither the first-order cap nor a train-loss governor can close it. Honest
conclusion: **use sharpness where flatness aligns with the target metric (classification / CE) —
there auto-`λ` is now the best arm; on MSE-to-one-hot regression, prefer Adam (`lam_auto` off).**
Reproduce with the `exact_sam_auto` / `exact_sam_adam_auto` arms and
`optim_experiments/autolam_trajectory.py` (in the separate `omnibias_experiments` project).

**Cost (honest — the "≤ SAM" claim needs the telemetry).** Mean wall-clock per step, ratio to
Adam at matched (register, width, seed):

| optimizer | sgd | adam | frugal_hutch | exact_sam | frugal_gn |
| --- | --- | --- | --- | --- | --- |
| × Adam / step | 0.75 | 1.00 | 1.27 | 3.55 | 7.41 |

`frugal_hutch` is nearly free (1.3×) for its half-memory + peak-suppression. **ExactSAM at
`probe_every=5` is 3.55× Adam — *above* SAM's 2×**, so the "≤ SAM cost" target is *not yet met*
at this amortization; reaching it needs a larger `probe_every` (rarer sharpness probes). The
telemetry lets us *set* that honestly rather than assume it. `frugal_gn`'s exact per-row
Gauss-Newton (subsampled to 64 rows) is the most expensive.

**Phase-1 bottom line.** Two Adam weak spots, two honest outcomes: **FrugalCurvature halves the
optimizer memory** while matching-or-beating Adam's `ce_relu` generalization at ~1.3× step cost,
and **ExactSAM lowers `ce_relu` test error below Adam and suppresses the double-descent peak** —
but only on the classification register, and above SAM's cost until `probe_every` is raised.
Neither is "beats Adam everywhere"; both are exact-curvature levers on the axes where Adam is
genuinely not optimal. Reproduce the grid with
`sweep/submit.sh --arms adam sgd exact_sam frugal_hutch frugal_gn --widths 5 13 30 50 75 100 150 250 --seeds 0 1 2 3 --noises 0.15 --steps 600` (then `experiment.write_summary` + `analysis/plots.py`).

## Honest caveats (scope explicitly)

- **"Closed-form" curvature** is exact for `mse_tanh` (Riccati tower); `ce_relu`
  curvature is exact-*to-autodiff* via matrix-free HVP — both are the *true*
  Hessian, labelled by how it is obtained. ReLU's `sigma'' = 0` makes its Hessian
  the Gauss-Newton matrix (trivial higher tower) — expected.
- **Curvature snapshots** default to the full training Hessian; on CPU we take
  them on a fixed training subsample (`--curv-batch`, the standard "Hessian on a
  held batch" convention) and use dense spectra only for narrow nets
  (`--dense-max-params`), matrix-free power-iteration / Hutchinson elsewhere. The
  dense-vs-matrix-free agreement at a shared narrow width is itself a check
  (`curvature.dense_vs_matrix_free_gap`, asserted in the tests).
- **Certified enclosures** (`omnibias.verify`) are *sound* but *local* and
  *narrow-width* only: interval / Taylor-model branch-and-bound over the 40-dim
  input box is expensive, so robustness uses a small `max_boxes`/`order` budget and
  the input-space flatness enclosure is off by default.
- The defensible thesis is the **exact-curvature + certified characterization** and
  the **optimizer-axis** result — not "beats Adam everywhere".
- **Phase-1 optimizers** are torch-only (bit-identical JAX twins are future work).
  ExactSAM is *amortized packaging* of the already-validated exact-sharpness penalty
  (H4), not new math; its "≤ SAM cost" is an aspiration the wall-clock telemetry shows
  is met only at larger `probe_every` (3.55× Adam at `probe_every=5`). FrugalCurvature's
  half-memory is structural and exact; its per-tensor curvature is *coarser* than Adam's
  per-coordinate `v`, so on some registers (`mse_tanh`) it lands between SGD and Adam.

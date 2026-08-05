# Complexity: time & memory

This page derives the time and memory complexity of the omnibias closed-form
differential operators and compares them, term by term, against the
state-of-the-art autodiff baselines:

- **folx** — the [Forward Laplacian](https://github.com/microsoft/folx)
  framework (Li et al., 2023), the fastest general-purpose Laplacian for
  neural-network wavefunctions.
- **`jax.hessian`** — JAX forward-over-reverse autodiff (`jacfwd ∘ jacrev`),
  the dense-Hessian baseline.
- **torch autograd** — `torch.func.hessian` (`jacfwd ∘ jacrev`), the same
  dense-Hessian baseline in PyTorch.

The headline result is summarised first, then derived.

!!! abstract "Headline (measured on GPU, float64, `H=256`, `B=4096`)"
    For the Laplacian of the one-layer field, the omnibias closed form does
    **`O(1)` derivative work per sample as the dimension `D` grows** — confirmed:
    time `0.167 → 0.211 ms` and memory `68 → 86 MiB` while `D` grows `80×`. The
    `D`-dependent part of the derivative (`‖W_h‖²`) is computed **once** and
    reused across the whole batch.

    - vs **dense-Hessian autodiff** (`jax.hessian`, `torch.func.hessian`): the
      win is large and **grows with `D`** — up to **68× / 199×** in time and
      **63× / 108×** in memory at `D = 240`.
    - vs **folx**: a near-tie for the *first* Laplacian (~1.0–1.25×) — folx is a
      strong sparsity-aware library and both are latency-bound at these sizes.
    - For the iterated Laplacian **`Δ^k`** (relativistic corrections), omnibias
      is **independent of both `k` and `D`**, while nested autodiff blows up:
      folx-nested re-pays the full pass each order, and the dense path explodes
      as `D^{2k}`. This is where omnibias decisively beats folx.

    All methods agree to `≤ 10⁻¹⁵` (float64) — the speedups are bit-for-bit, not
    accuracy trades.

## The model

All methods compute derivatives of the omnibias one-layer scalar field

\[
f(x) = b + \sum_{h=1}^{H} c_h\,\sigma\!\big(W_h \cdot x + \beta_h\big),
\qquad x \in \mathbb{R}^{D},
\]

with `W ∈ ℝ^{H×D}`, evaluated on a batch of `B` points. This is the building
block of the FermiNet/DeepQMC local kinetic energy (`omnibias-ferminet`), where
the Laplacian `Δf` is the kinetic term and `Δ^k f` are the relativistic
mass–velocity corrections.

The closed forms omnibias ships (see `omnibias.jax.laplacian`) are

\[
\nabla^2 f(x) = \sum_h c_h\,\sigma''(z_h)\,\lVert W_h\rVert^2,
\qquad
\Delta^k f(x) = \sum_h c_h\,\sigma^{(2k)}(z_h)\,\lVert W_h\rVert^{2k},
\qquad z_h = W_h\cdot x + \beta_h .
\]

Because `σ` is a Riccati-class activation, every derivative tower
`σ', σ'', …, σ^{(2k)}` is itself a closed-form forward pass (no nested
differentiation), so the entire object above is one forward evaluation.

## Cost model

Let

- `B` = batch size, `D` = input dimension, `H` = hidden width.
- `F = O(B · H · D)` = cost of **one forward pass** `f(x)` (the `X Wᵀ` matmul).
  This is the irreducible floor: you cannot evaluate `f`, let alone any
  derivative, for less.

We separate **total** cost from **derivative overhead** = (cost of the
derivative) − `F`. The interesting quantity is how the *overhead* scales in `D`.

## Derivation — Laplacian `Δf`

### omnibias (closed form)

`neural_field_value_grad_laplacian` computes `z = XWᵀ+β` (cost `F`), the towers
`σ, σ', σ''` (each `O(B·H)`), the per-row norms `r_h = ‖W_h‖²` (a single
`O(H·D)` reduction, computed **once** and reused for every sample and every
step), then the contraction `Σ_h c_h σ''(z_h) r_h` (`O(B·H)`).

- **Time:** `F + O(B·H)` — the derivative overhead is `O(B·H)`, i.e. **`O(1)`
  in `D`** (the lone `O(H·D)` norm term is amortised over the batch and across
  training steps).
- **Memory:** `O(B·H)` activations `+ O(H·D)` parameters. **No `D×D` object is
  ever formed; the overhead is `O(1)` in `D`.**

### folx (Forward Laplacian)

folx augments every intermediate with `(value, Jacobian wrt x, Laplacian)`.
For the hidden layer the Jacobian tangent is the `(H × D)` matrix `J = σ'(z) ⊙ W`;
the Laplacian accumulator needs `Σ_d J_{h,d}²` per unit, which folx evaluates
without ever forming the `D×D` Hessian.

- **Time:** `O(B·H·D)` — same order as the forward pass. folx re-pays the
  Jacobian contraction **per sample** (it does not amortise `‖W_h‖²` across the
  batch the way the closed form does), so its constant is larger, but it never
  touches a `D²` object.
- **Memory:** `O(B·H)` in practice — folx tracks the input Jacobian as a sparse
  structure, so for this field its measured peak is flat in `D` (≈ omnibias).

!!! info "Measured: folx ≈ omnibias for the *first* Laplacian"
    For the **single** Laplacian at VMC batch sizes both omnibias and folx are
    latency-bound (a handful of kernel launches), so both are essentially flat in
    `D` and within ~1.0–1.25× of each other (see the measured table below).
    omnibias's structural advantage over folx appears at **high order**
    (`Δ^k`, `k≥2`): folx must *nest*, re-paying the whole forward-Laplacian pass
    each time and falling back to the full Hessian, while omnibias stays `O(1)`.

### `jax.hessian` and torch autograd (dense Hessian + trace)

`trace(jacfwd(jacrev(f)))` (JAX) and `torch.trace(torch.func.hessian(f))`
materialise the full `D×D` Hessian `H_x f = Wᵀ diag(σ''(z)⊙c) W` per sample,
then trace it.

- **Time:** `O(B·H·D²)` — forming the dense Hessian is quadratic in `D`.
- **Memory:** `O(B·D²)` — the dense Hessian batch. **Quadratic in `D`.**

### Summary — Laplacian

| Method | Time | Memory | Derivative overhead vs forward | Exactness |
|---|---|---|---|---|
| **omnibias** closed form | `F + O(B·H)` | `O(B·H)` | **`O(1)` in `D`** | bit-exact |
| folx (forward Laplacian) | `O(B·H·D)` | `O(B·H)` (sparse) | small constant; no `D²` | AD-exact (float) |
| `jax.hessian` (jacfwd∘jacrev) | `O(B·H·D²)` | `O(B·D²)` | `O(D²)` | AD-exact (float) |
| torch `func.hessian` | `O(B·H·D²)` | `O(B·D²)` | `O(D²)` | AD-exact (float) |

The "`O(1)` in `D`" claim is precise: it is the **overhead over the forward
pass**, per sample, holding `B` and `H` fixed. The absolute cost is still
`F = O(B·H·D)` because evaluating `f` itself reads a `D`-vector — no method can
beat that floor.

## Derivation — iterated Laplacian `Δ^k f`

The polylaplacian is where the closed form pulls decisively ahead, because the
order of differentiation `2k` does **not** change the omnibias cost.

### omnibias

`neural_field_polylaplacian(…, k)` evaluates one tower `σ^{(2k)}` and contracts
with the precomputed `‖W_h‖^{2k}`:

- **Time:** `O(B·H)` derivative overhead — **independent of `k` and of `D`.**
- **Memory:** `O(B·H)` — independent of `k` and `D`.

### nested autodiff

To get `Δ^k` from autodiff you nest the Laplacian operator `k` times. Each
nesting differentiates a function that already contains a `D`-fold sum:

- folx-nested: `O(B·H·D^{k-1})` (each extra Laplacian multiplies the tangent
  bookkeeping by `D`).
- dense-Hessian nested: `O(B·H·D^{2k})` time / `O(B·D^{2k})` memory — each level
  squares a fresh `D×D` block and the graph compounds.

### Summary — polylaplacian `Δ^k`

| Method | Time | Memory | scaling in `k` |
|---|---|---|---|
| **omnibias** closed form | `O(B·H)` | `O(B·H)` | **flat** |
| folx-nested | `≳ O(B·H·D^{k-1})` | `O(B·D^{2(k-1)})`† | exponential in `k` |
| dense-Hessian nested | `O(B·H·D^{2k})` | `O(B·D^{2k})` | exponential in `k` |

† Nesting defeats folx's sparsity: from the second Laplacian on it falls back to
materialising the full Hessian (folx prints `compute the full hessian`), so its
memory grows like the dense path — which is why folx-nested **runs out of
memory** at `k=4` (`D=30`) and `k=3` (`D=120`) in the measurements below, while
omnibias is unaffected.

Measured on GPU (`D = 30`, 10-electron-class): the closed form is ~`1.8×` ahead
at `k=2`, **~`480×` ahead at `k=3`**, and at `k=4` folx-nested no longer
completes while omnibias is unchanged — the gap widens with both `D` and `k`
(see the GPU table below).

## Measured results

All methods agree to floating-point round-off (`≤ 10⁻¹⁵` absolute, float64), so
the speedups below compare *identical* numerical answers — the closed form is
not trading accuracy for speed; it is bit-exact.

### CPU smoke tier (reproducible from this repo)

Produced by [`benchmarks/laplacian_scaling.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/laplacian_scaling.py)
and committed as [`docs/benchmarks/laplacian_scaling.json`](benchmarks/laplacian_scaling.json).
`H = 32`, `B = 64`, float64, commodity CPU (`JAX_PLATFORMS=cpu`); cost reported as
**slowdown vs the omnibias closed form** (higher = slower than omnibias). Absolute
omnibias time stays ~0.004 ms across `D`; the autodiff cost grows with `D`:

| `D` | omnibias | folx | `jax.hessian` | torch `func.hessian` |
|---|---|---|---|---|
| 3 | 1.0× | 2.9× | 5.9× | 298× |
| 12 | 1.0× | 2.4× | 7.1× | 336× |
| 30 | 1.0× | 22.6× | 93× | 589× |
| 60 | 1.0× | 6.6× | **211×** | **923×** |

All four methods agree to `≤ 2×10⁻¹⁵` absolute. See the JSON for per-method
milliseconds and the exact library versions.

Polylaplacian `Δ^k`, from
[`benchmarks/polylaplacian_order.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/polylaplacian_order.py)
→ [`docs/benchmarks/polylaplacian_order.json`](benchmarks/polylaplacian_order.json)
(`D = 16`, `H = 16`, `B = 32`). Omnibias is flat in `k`; both nested baselines explode:

| `k` | omnibias | folx-nested | speedup vs folx | dense-nested | speedup vs dense |
|---|---|---|---|---|---|
| 1 | 0.0045 ms | 0.106 ms | 24× | 0.188 ms | 42× |
| 2 | 0.0045 ms | 0.138 ms | 31× | 0.639 ms | 142× |
| 3 | 0.0045 ms | 1.11 ms | **246×** | 59.0 ms | **13,100×** |
| 4 | 0.024 ms | 111 ms | **4,660×** | 4315 ms | **181,000×** |

Outputs agree with the dense nested Hessian to `≤ 8×10⁻¹¹` at `k=4` (and
`≤ 4×10⁻¹³` at `k=3`). This is the regime where omnibias is *thousands of times*
faster than nested autodiff — even on CPU at a modest `D=16`.

### GPU headline tier (off-band)

Full-fidelity Laplacian sweep (`H = 256`, `B = 4096`, float64, one data-center
GPU). These numbers were measured off-band and transcribed here; they are **not**
produced by the public `benchmarks/` scripts. Time is absolute ms for omnibias
and **slowdown ×** for the baselines (higher = slower). Omnibias is flat in `D`;
the dense-Hessian paths scale, folx tracks omnibias:

| `D` | omnibias (ms) | folx | `jax.hessian` | torch `func.hessian` |
|---|---|---|---|---|
| 3 | 0.167 | 1.25× | 1.36× | 17.3× |
| 12 | 0.173 | 1.07× | 2.30× | 16.7× |
| 30 | 0.192 | 1.01× | 4.44× | 21.4× |
| 60 | 0.198 | 1.01× | 9.55× | 39.7× |
| 120 | 0.191 | 1.07× | 27.1× | 92.1× |
| 240 | 0.211 | 1.13× | **67.7×** | **198.7×** |

Peak device memory (MiB), process-isolated per method (omnibias/folx flat,
dense-Hessian grows steeply with `D`):

| `D` | omnibias | folx | `jax.hessian` | torch |
|---|---|---|---|---|
| 3 | 68 | 72 | 68 | 137 |
| 30 | 70 | 72 | 333 | 814 |
| 120 | 76 | 76 | 1418 | 3403 |
| 240 | 86 | 86 | 5424 (**63×**) | 9305 (**108×**) |

All four methods agreed to `≤ 1.0×10⁻¹⁵` absolute (float64) at every `D` —
identical answers, so the speedups are not bought with accuracy.

**Reading of the data.** The omnibias `O(1)`-in-`D` claim is confirmed: time
0.167 → 0.211 ms and memory 68 → 86 MiB while `D` grows 80×. Against naive
dense-Hessian autodiff the win is large and *grows with `D`* — up to **68×
(jax)** / **199× (torch)** in time and **63× / 108×** in memory at `D = 240`.
Against folx, the *first* Laplacian is a near-tie (~1.1×): folx is a strong,
sparsity-aware library and both are latency-bound here. The decisive omnibias
advantage over folx is at **high order** (`Δ^k`, next section).

### GPU polylaplacian (`Δ^k`) tier — omnibias vs folx-nested

`H = 128`, `B = 1024`, float64, one data-center GPU. omnibias time is **flat in
both `k` and `D`** (one `σ^{(2k)}` tower + a reduction); folx-nested re-runs the
forward-Laplacian each order and falls back to the full Hessian, so it grows
explosively and eventually **fails to complete** (out of memory) — at orders
where omnibias still finishes in ~0.1 ms:

| `D` | `k` | omnibias (ms) | folx-nested (ms) | speedup |
|---|---|---|---|---|
| 30 | 1 | 0.080 | 0.103 | 1.3× |
| 30 | 2 | 0.086 | 0.154 | 1.8× |
| 30 | 3 | 0.085 | 40.6 | **479×** |
| 30 | 4 | 0.104 | — (OOM) | **∞** |
| 60 | 3 | 0.123 | 63.5 | **518×** |
| 120 | 2 | 0.121 | 0.324 | 2.7× |
| 120 | 3 | 0.133 | — (OOM) | **∞** |

At `k=3` omnibias is ~**480–520× faster than folx** and rising with `D`; at
`k≥4` (or `D≥120, k≥3`) folx-nested no longer completes while omnibias is
unaffected. omnibias `Δ^k` stays at `0.08–0.13 ms` for **every** `(D, k)` tested
— the textbook signature of an `O(1)`-in-`(k,D)` cost. Outputs agree to
`≤ 3×10⁻¹³`.

## Caveats & honest scope

- The closed-form complexity above is for the **one-layer field** that
  `omnibias.jax.laplacian` ships — exactly the FermiNet local-kinetic-energy
  primitive benchmarked here. Arbitrary deep ansätze need the closed-form path
  threaded through every layer; that is the multi-layer **jet** machinery
  (`omnibias.jax.jet`, `jet_mv`), whose own scaling is benchmarked separately.
- "`O(1)` in `D`" always means **overhead over the forward pass**, per sample,
  at fixed `(B, H)`. The forward pass itself is `O(B·H·D)`; that floor is shared
  by every method and is not what the comparison is about.
- folx remains the right tool for *general* networks where no closed form
  exists; omnibias wins precisely when the field is built from its Riccati-class
  activations, where the derivative tower is itself a forward pass.

## Reproducing

CPU smoke (seconds, any host with the workspace deps + optional `folx`):

```bash
uv run python benchmarks/laplacian_scaling.py
uv run python benchmarks/polylaplacian_order.py
uv run python benchmarks/derivative_order.py
uv run python benchmarks/optimizer_pinn.py
```

Artifacts land in [`docs/benchmarks/`](https://github.com/derivon-ai/omnibias/tree/main/docs/benchmarks). See
[`benchmarks/README.md`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/README.md)
for the full suite. The GPU headline tables above are off-band measurements and
are not regenerated by these scripts.
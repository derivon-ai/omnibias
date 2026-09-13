# Complexity: time & memory

This page derives the time and memory complexity of the omnibias closed-form
differential operators and compares them, term by term, against the
autodiff baselines measured here:

- **folx** — the [Forward Laplacian](https://github.com/microsoft/folx)
  framework for neural-network wavefunctions.
- **`jax.hessian`** — JAX forward-over-reverse autodiff (`jacfwd ∘ jacrev`),
  the dense-Hessian baseline.
- **torch autograd** — `torch.func.hessian` (`jacfwd ∘ jacrev`), the same
  dense-Hessian baseline in PyTorch.

The headline result is summarised first, then derived.

!!! abstract "Headline (measured on GPU, float64, `H=256`, `B=4096`)"
    For the Laplacian of the one-layer field, the omnibias closed form does
    **`O(B·H)` contraction work after computing the current weight norms**.
    Measured time was `0.167 → 0.211 ms` and memory `68 → 86 MiB` while `D`
    grew `80×`. The
    `D`-dependent part of the derivative (`‖W_h‖²`) is computed **once** and
    reused across the whole batch.

    - vs **dense-Hessian autodiff** (`jax.hessian`, `torch.func.hessian`): the
      win is large and **grows with `D`** — up to **68× / 199×** in time and
      **63× / 108×** in memory at `D = 240`.
    - vs **folx**: a near-tie for the *first* Laplacian (~1.0–1.25×) — folx is a
      strong sparsity-aware library and both are latency-bound at these sizes.
    - For the iterated Laplacian **`Δ^k`** (relativistic corrections), omnibias
      has **`O(B·H·k)` polynomial evaluation cost**, plus the forward matmul and
      current weight norms. The nested baselines become much more expensive
      in the measured cases. This is the main advantage demonstrated here.

    The first-Laplacian GPU results agree to `≤ 10⁻¹⁵` in float64. This is
    numerical agreement within a tolerance, not bit-for-bit identity. The
    higher-order errors are reported separately below.

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

For sigmoid and tanh, the requested derivative `σ^{(2k)}` is a polynomial
of degree `2k+1` in one activation value. Evaluating it needs no nested
differentiation. Computing one derivative and materialising every derivative
through that order are different workloads.

## Cost model

Let

- `B` = batch size, `D` = input dimension, `H` = hidden width.
- `F = O(B · H · D)` = the conventional dense-matmul cost of **one forward
  pass** `f(x)` (the `X Wᵀ` matmul).

We separate **total** cost from **derivative overhead** = (cost of the
derivative) − `F`. The interesting quantity is how the *overhead* scales in `D`.

## Derivation — Laplacian `Δf`

### omnibias (closed form)

`neural_field_laplacian` computes `z = XWᵀ+β` (cost `F`), `σ''`
(`O(B·H)`), the per-row norms `r_h = ‖W_h‖²` (an `O(H·D)` reduction), then
the contraction `Σ_h c_h σ''(z_h) r_h` (`O(B·H)`). The implementation computes
the norms on each call and shares them across that batch. They may be cached
only while `W` is unchanged; training updates invalidate such a cache.
`neural_field_value_grad_laplacian` additionally returns the gradient, whose
matrix multiplication costs another `O(B·H·D)`.

- **Time:** `F + O(H·D + B·H)`. After the current weight norms are available,
  the activation and contraction work is `O(B·H)`, independent of `D`.
  With the norms included, the per-sample overhead is `O(H·D/B + H)`.
- **Forward working arrays:** `O(B·H + H)`, in addition to the `O(B·D)` input
  and `O(H·D)` parameters. No `D×D` Hessian is formed. Compiler workspaces and
  reverse-mode training storage are separate from these array counts.

### folx (Forward Laplacian)

folx augments every intermediate with `(value, Jacobian wrt x, Laplacian)`.
The affine hidden-layer Jacobian is `W`; after activation it is
`J = σ'(z) ⊙ W`. The activation chain rule contributes
`σ''(z_h) Σ_d W_{h,d}²` to each hidden-unit Laplacian, without forming the
`D×D` Hessian.

- **Time:** `O(B·H·D)` — same order as the forward pass. folx re-pays the
  Jacobian contraction **per sample** (it does not amortise `‖W_h‖²` across the
  batch the way the closed form does), so its constant is larger, but it never
  touches a `D²` object.
- **Memory:** depends on the Jacobian representation and compiler. A dense
  hidden-layer Jacobian contains `O(B·H·D)` entries; sparsity can reduce this.
  The nearly flat process-level measurements below are observations at those
  sizes, not a general `O(B·H)` memory bound.

!!! info "Measured: folx ≈ omnibias for the *first* Laplacian"
    For the **single** Laplacian at VMC batch sizes both omnibias and folx are
    latency-bound (a handful of kernel launches), so both are essentially flat in
    `D` and within ~1.0–1.25× of each other (see the measured table below).
    omnibias's structural advantage over folx appears at **high order**
    (`Δ^k`, `k≥2`): folx must *nest*, re-paying the whole forward-Laplacian pass
    each time and, in this benchmark, falling back to the full Hessian.
    Omnibias instead evaluates an order-dependent polynomial.

### `jax.hessian` and torch autograd (dense Hessian + trace)

`trace(jacfwd(jacrev(f)))` (JAX) and `torch.trace(torch.func.hessian(f))`
materialise the full `D×D` Hessian `H_x f = Wᵀ diag(σ''(z)⊙c) W` per sample,
then trace it.

- **Time:** `O(B·H·D²)` — forming the dense Hessian is quadratic in `D`.
- **Memory:** `O(B·D²)` — the dense Hessian batch. **Quadratic in `D`.**

### Summary — Laplacian

| Method | Time | Memory | Derivative overhead vs forward | Exactness |
|---|---|---|---|---|
| **omnibias** closed form | `F + O(H·D + B·H)` | `O(B·H + H)` working arrays | `O(B·H)` after current norms | analytic formula, floating evaluation |
| folx (forward Laplacian) | `O(B·H·D)` | Jacobian/sparsity dependent | no explicit `D²` Hessian required | AD-exact (float) |
| `jax.hessian` (jacfwd∘jacrev) | `O(B·H·D²)` | `O(B·D²)` | `O(D²)` | AD-exact (float) |
| torch `func.hessian` | `O(B·H·D²)` | `O(B·D²)` | `O(D²)` | AD-exact (float) |

The dimension-independent part is the activation/contraction work **after
the current weight norms have been computed**. At fixed `B` and `H`, the
`O(H·D)` norm computation still grows with `D`. The total conventional dense
cost remains `O(B·H·D)`. The table excludes shared inputs and parameters from
working-array counts; measured process/device memory includes more than these
arrays.

## Derivation — iterated Laplacian `Δ^k f`

The polylaplacian is where the closed form pulls decisively ahead in these
benchmarks: increasing `k` lengthens a polynomial evaluation without nesting
Laplacian transforms.

### omnibias

`neural_field_polylaplacian(…, k)` evaluates the single derivative `σ^{(2k)}`
and contracts with `‖W_h‖^{2k}`. For sigmoid/tanh and cached coefficients:

- **Time:** `F + O(H·D + B·H·k)`. Horner evaluation uses `O(k)` multiply-adds
  per activation. Computing integer powers of the norms adds at most
  `O(H·log k)` multiplications using exponentiation by squaring, subsumed by
  the polynomial term. Special activations such as `exp` can have lower cost.
- **Forward working arrays:** streaming Horner evaluation needs `O(B·H + H)`
  array storage, plus `O(k)` coefficients. The emitted polynomial graph grows
  with `k`; compiler fusion and reverse-mode saved intermediates affect actual
  peak memory. A forward array count is not a training-memory guarantee.

Coefficient generation is separate preprocessing: the current integer
recurrences take `O(k²)` arithmetic operations on a cache miss for order
`2k`, with growing integer sizes, then round each coefficient once. This cost
is excluded from warm timings. Floating representations have finite order
limits; “arbitrary order” describes the recurrence, not unlimited float64
representability.

A complete activation tower through order `N`, with cached coefficients,
takes `O(B·H·N²)` Horner work when evaluating all its polynomials separately,
and its output alone contains `O(B·H·N)` entries. The current generic jet
helper calls each activation
fastpath separately, so it does not promise one shared activation evaluation
in eager execution; a compiler may eliminate repeated evaluations. The
opt-in `compose_jet_riccati` instead uses one activation value and costs
`O(deg(P)·N²)` to propagate the whole composed jet. General `compose_jet`
remains cubic in `N`.

### nested autodiff

The baselines here nest a Laplacian transform `k` times. The dense version
builds Hessians before tracing; the folx version repeatedly applies its
forward-Laplacian transform. Their intermediate graphs become expensive in
the measured runs. A fully materialised order-`2k` derivative tensor would
contain `D^{2k}` entries, but that is **not** a proved time or memory bound
for these compiled implementations: contraction order, sparsity, and compiler
elimination matter. Other AD methods need not construct that tensor.

### Summary — polylaplacian `Δ^k`

| Method | Time | Forward working arrays | scaling in `k` |
|---|---|---|---|
| **omnibias** closed form | `F + O(H·D + B·H·k)` | `O(B·H + H + k)` with streaming Horner | linear polynomial work |
| folx-nested | implementation dependent | implementation dependent† | rapid growth measured |
| dense-Hessian nested | implementation dependent | implementation dependent | rapid growth measured |

† In this benchmark, nesting triggers a full-Hessian fallback (folx prints
`compute the full hessian`). The reported GPU runs exhaust memory at `k=4`
(`D=30`) and `k=3` (`D=120`), while omnibias completes those cases. These
observations do not establish a universal memory exponent for folx.

Measured on GPU (`D = 30`, 10-electron-class): the closed form is ~`1.8×` ahead
at `k=2`, **~`480×` ahead at `k=3`**, and at `k=4` folx-nested no longer
completes while omnibias takes `0.104 ms` — the measured gap widens with `D` and `k`
(see the GPU table below).

## Measured results

These methods evaluate the same mathematical operators. Their floating-point
answers agree within the errors reported for each experiment below; the
agreement is not bit-for-bit. An analytic formula removes finite-difference
truncation error, but still has floating-point rounding and conditioning
error. Shared polynomial coefficients do not guarantee identical backend
outputs across native activation kernels, compiler fusion, and devices.

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
(`D = 16`, `H = 16`, `B = 32`). Omnibias rises from `0.0045` to `0.024 ms`
over this range; both nested baselines grow much faster:

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
and **slowdown ×** for the baselines (higher = slower). Omnibias varies modestly
over this `D` range; the dense-Hessian paths grow, and folx tracks omnibias:

| `D` | omnibias (ms) | folx | `jax.hessian` | torch `func.hessian` |
|---|---|---|---|---|
| 3 | 0.167 | 1.25× | 1.36× | 17.3× |
| 12 | 0.173 | 1.07× | 2.30× | 16.7× |
| 30 | 0.192 | 1.01× | 4.44× | 21.4× |
| 60 | 0.198 | 1.01× | 9.55× | 39.7× |
| 120 | 0.191 | 1.07× | 27.1× | 92.1× |
| 240 | 0.211 | 1.13× | **67.7×** | **198.7×** |

Peak device memory (MiB), process-isolated per method (omnibias/folx vary
modestly here; dense-Hessian grows steeply with `D`):

| `D` | omnibias | folx | `jax.hessian` | torch |
|---|---|---|---|---|
| 3 | 68 | 72 | 68 | 137 |
| 30 | 70 | 72 | 333 | 814 |
| 120 | 76 | 76 | 1418 | 3403 |
| 240 | 86 | 86 | 5424 (**63×**) | 9305 (**108×**) |

All four methods agreed to `≤ 1.0×10⁻¹⁵` absolute (float64) at every `D` —
agreement within the stated tolerance, not identical output bits.

**Reading of the data.** Time changes from 0.167 → 0.211 ms and memory from
68 → 86 MiB while `D` grows 80× in these measurements. This finite sweep
does not establish an asymptotic bound. Against
dense-Hessian autodiff the win is large and *grows with `D`* — up to **68×
(jax)** / **199× (torch)** in time and **63× / 108×** in memory at `D = 240`.
Against folx, the *first* Laplacian is a near-tie (~1.1×): folx is a strong,
sparsity-aware library and both are latency-bound here. The decisive omnibias
advantage over folx is at **high order** (`Δ^k`, next section).

### GPU polylaplacian (`Δ^k`) tier — omnibias vs folx-nested

`H = 128`, `B = 1024`, float64, one data-center GPU. The measured omnibias
times remain near `0.1 ms` over the listed `(D, k)` cases, despite the
order-dependent polynomial work. The folx-nested baseline becomes much
slower and eventually **fails to complete** (out of memory):

| `D` | `k` | omnibias (ms) | folx-nested (ms) | speedup |
|---|---|---|---|---|
| 30 | 1 | 0.080 | 0.103 | 1.3× |
| 30 | 2 | 0.086 | 0.154 | 1.8× |
| 30 | 3 | 0.085 | 40.6 | **479×** |
| 30 | 4 | 0.104 | — (OOM) | unavailable |
| 60 | 3 | 0.123 | 63.5 | **518×** |
| 120 | 2 | 0.121 | 0.324 | 2.7× |
| 120 | 3 | 0.133 | — (OOM) | unavailable |

At `k=3`, the reported `D=30` and `D=60` cases give approximately
**480–520× speedups over folx-nested**. At `(D,k)=(30,4)` and `(120,3)`, the
baseline fails to complete and omnibias finishes; no finite timing ratio is
available for those rows. Omnibias takes `0.080–0.133 ms` across the listed
cases. Reported outputs agree to `≤ 3×10⁻¹³` where both methods complete.

## Caveats & honest scope

- The closed-form complexity above is for the **one-layer field** that
  `omnibias.jax.laplacian` ships — exactly the FermiNet local-kinetic-energy
  primitive benchmarked here. Arbitrary deep ansätze need the closed-form path
  threaded through every layer; that is the multi-layer **jet** machinery
  (`omnibias.jax.jet`, `jet_mv`), whose own scaling is benchmarked separately.
- Dimension-independent contraction work assumes the current weight norms
  are available. Computing those norms costs `O(H·D)`; the forward dense
  matmul costs `O(B·H·D)`; the requested order contributes polynomial work.
- folx supports general networks beyond the one-layer closed form studied
  here. The measured advantage does not imply a win for every Riccati network,
  order, batch size, or device.
- The polylaplacian script JIT-compiles zero-argument closures over fixed
  inputs and weights. These warm evaluation timings do not measure a
  parameter-update training loop. Further benchmarks should pass changing
  inputs and parameters as runtime arguments and report compilation, warm
  evaluation, and parameter-gradient costs separately.

### Comparisons still needed

[JAX's `jax.experimental.jet`](https://docs.jax.dev/en/latest/jax.experimental.jet.html)
already propagates truncated Taylor polynomials without repeated first-order
AD. It is an existing correctness oracle in `packages/omnibias-jax/tests/test_jet.py`
and `benchmarks/singularity_tracking.py`; the latter times repeated `jacfwd`,
not `experimental.jet`. A broader performance claim needs matched-order,
matched-output timing against Taylor mode, including the Riccati fastpath.

The [NeurIPS 2024 STDE paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd2eb5250696753ea37141bbd89bb569-Abstract-Conference.html)
uses randomized Taylor-mode contractions for high-order differential
operators. Comparing against it requires reporting estimation variance and
cost at matched error, alongside deterministic operator timings. Neither the
existing nested-AD tables nor a low-order Taylor-mode agreement test establishes
superiority over that method.

The external [torch-jet project](https://github.com/f-dangel/torch-jet)
implements Taylor mode in PyTorch and hosts the NeurIPS 2025 paper
*Collapsing Taylor Mode Automatic Differentiation*. Its operator-specific
acceleration is another relevant baseline for the proposed comparison.
An import alias for `omnibias.torch.jet` is not a comparison to that library.

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

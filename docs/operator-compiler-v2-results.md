# Compiler execution and precision: v2 evidence

The supported compiler now provides bounded activation lowering, an independent
native graph for parameter higher gradients, and explicit Keras computation and
objective precision. The [operator guide](operator-compiler.md) and
[Keras precision contract](jet-compiler.md#explicit-numerical-precision) describe
the interfaces. The [v1 results](operator-compiler-results.md) remain historical
evidence; timings from different source snapshots or protocols are not pooled.

## Order-32 cold-start acceptance

The coupled network has widths 16 → 16 → 16 → 1, batch 32, float64, and a
32nd-order directional request. Automatic lowering selects reusable activation
regions through the public Torch API. Three fresh CPU processes each completed
capture, compilation, value validation, parameter gradients, 20 warm samples per
timed stage, optimizer updates, and tensor-memory profiling within the requested
120-second and 3,072-MiB engine limits.

| Replicate | Engine wall time | Forward compilation and first call | First gradient call | Warm forward | Warm backward | Complete SGD step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 119.753 s | 53.975 s | 61.263 s | 0.5595 ms | 1.5590 ms | 2.2851 ms |
| 1 | 119.653 s | 53.674 s | 61.371 s | 0.9459 ms | 1.5317 ms | 2.2433 ms |
| 2 | 118.368 s | 53.002 s | 60.983 s | 0.5654 ms | 1.5666 ms | 2.3683 ms |

Peak self-reported process RSS was 1,081–1,085 MiB; the parent's sampled peak
was 1,073–1,075 MiB. The separately tracked CPU tensor-storage peak was
1,864,224 bytes. These are different memory measures. Worst sampled scaled
value error against the independent 80-digit reference was 2.97×10⁻¹⁵.
Derivative-only optimizer steps changed original parameters.

The slowest worker had only **0.247 seconds of margin**. This establishes the
gate on this machine and workload, not a portable deadline guarantee. The
median of process forward medians was 0.5654 ms; the higher replicate-1
latency remains in the evidence. Do not treat the earlier 0.369-ms feasibility
probe as the release measurement.

The independent reference cost 7.163 seconds once for all three workers. Total
report wall time was 364.965 seconds. A single worker plus reference generation
would take 125.531–126.916 seconds; the 120-second acceptance budget explicitly
excludes the separately reported reference.

## Cache protocol and measurement scope

Protocol `omnibias/operator-compiler/v2` records references keyed by exact
operands, derivative request, oracle source hash, and oracle version. Engine
workers validate this record before execution. Stage boundaries are flushed
before and after expensive work, so timeouts identify the active stage.

Cold runs start with empty isolated cache directories and permit normal backend
reuse during that worker. Persistent runs reuse on-disk caches across workers.
The explicit `disabled` diagnostic suppresses backend cache layers. Earlier
failed experiments called that last behavior "cold": disabling caches prevented
regional reuse between compilation calls. Those failed reports are preserved
with their original environment and source snapshots, and are not combined with
the corrected cold-start measurements. No unsafe guards or larger budgets were
used to obtain the passing runs.
An earlier singleton diagnostic also overlapped a parameter-AD test run; it is
retained for completeness and is not controlled deadline evidence. The later
three-worker cache-disabled experiment and all accepted timing sweeps ran
without concurrent validation jobs.

The cold acceptance run preceded a canonical license-header restamp. A manifest
records every before/after file hash and verifies identical Python ASTs. The
subsequent comparison sweep uses its own complete source snapshot. No timing
ratio combines these reports.

Measurements use an Intel Core i7-11850H, CPU affinity 2, one tensor-computation
thread, and one Inductor compilation worker. The recorded versions are Python
3.12.13, Torch 2.14.1+cu130 with CPU execution, JAX 0.10.2, Keras 3.15.1, folx
0.2.30, and jet-for-pytorch 1.0.0. This is a controlled workstation measurement,
not GPU certification. Warm Torch training compiles the derivative forward and
uses Python loss/backward/SGD orchestration; JAX compiles a whole functional
training step. Those training scopes do not earn cross-framework speedup ratios.
Torch parameter resets and `zero_grad` occur outside the timed samples; the
reported step includes derivative evaluation, loss, backward, and `SGD.step`.

## Matching-source performance comparisons

Every accepted method below has three fresh processes and at least 20
synchronized samples per timed stage, with independent accuracy checks before
acceptance. Tables show medians of process medians. These family sweeps use
persistent disk caches starting from a new matrix directory, with sequential
workers and no concurrent test or benchmark jobs. First-call costs and failures
remain separate in the raw reports.

### Ridge contraction

The ridge model has D=H=16 and batch 32. All four orders pass the existing
within-20%-or-5-µs allowance against their equivalent handwritten kernels, for
both Torch and strict Keras/JAX. All accepted cells also complete derivative-only
parameter-gradient and optimizer checks.

| Operator | Torch compiler | Torch handwritten | Strict Keras/JAX compiler | JAX handwritten | Dense nested JAX | Nested folx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Δ | 0.06424 ms | 0.06631 ms | 0.02266 ms | **0.02130 ms** | 0.03497 ms | 0.02145 ms |
| Δ² | 0.06332 ms | 0.06826 ms | **0.02374 ms** | 0.02511 ms | 0.85638 ms | 0.04502 ms |
| Δ³ | 0.06676 ms | 0.07256 ms | 0.02465 ms | **0.02420 ms** | 144.12771 ms | 0.74868 ms |
| Δ⁴ | 0.06652 ms | 0.07465 ms | **0.02759 ms** | 0.02798 ms | Memory budget | 157.84975 ms |

The automatically generated strict Keras/JAX contraction is approximately
**5,846× faster than dense nested JAX for Δ³** and **5,721× faster than nested
folx for Δ⁴**, with matching runtime operands, source snapshot, and precision
gate. Relative to the specialized handwritten JAX contraction, it is about
0.98× and 1.01× as fast, respectively: essentially comparable in these small
timings. The advantage is recovering that specialized contraction automatically
from ordinary model structure. These are warm forward comparisons for this
ridge workload, not claims about arbitrary programs or training speedups.

Torch's corresponding warm forward ratios are approximately 2,159× and 2,373×
against those two nested baselines. The faster JAX implementations remain in
the table; no headline ratio replaces the fastest applicable baseline.

The ridge report contains 69 successful cells and three memory-budget failures,
all dense nested JAX at order eight. A failed baseline receives no timing ratio.

### Disjoint mixed derivatives

For `sin(sum_i tanh(w_i*x_i+b_i))`, the compiler proves coordinate independence
and propagates branch first derivatives into a univariate outer program.

| Mixed order | Torch compiler | Handwritten formula | Generic fixed-seed mixed jet |
| ---: | ---: | ---: | --- |
| 4 | 0.04228 ms | **0.04044 ms** | 0.09746 ms |
| 8 | 0.04702 ms | **0.04221 ms** | Graph budget refused |
| 24 | 0.04973 ms | **0.04505 ms** | 16,777,216 coefficients; refused |
| 32 | 0.04982 ms | **0.04574 ms** | 4,294,967,296 coefficients; refused |

All compiler and handwritten cells pass value, parameter-gradient, and optimizer
checks. Orders 24/32 use 25/33 outer coefficients without constructing the generic
mixed layout. At order four, the compiler is 2.30× faster than the reference jet;
the handwritten formula remains faster at every tested order. Refused layouts
have no speedup ratio.

The report preserves 27 successful cells, 15 refusals, three timeouts, and three
worker errors. Nested Torch at order eight times out during compilation/first
execution. At order four, it completes forward/accuracy/warm evaluation but exits
without a complete result during gradient validation; the cause is unclassified.
Those partial observations are not accepted forward-and-training comparisons.

### Affine bottleneck transport

The 32 → 2 → 16 → 1 model has tanh after the first two maps. The compiler
transports spatial operators through the first affine map and retains the live
metric `AAᵀ` in the parameter-gradient graph. All 30 cells pass the complete
accuracy, gradient, and training protocol.

| Operator | Torch compiler | Torch handwritten | Strict Keras/JAX compiler | folx | torch-jet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laplacian | 0.10876 ms | 0.12029 ms | 0.03793 ms | **0.03520 ms** | 0.10154 ms |
| Biharmonic | 0.15745 ms | 0.16517 ms | **0.05346 ms** | 0.81403 ms | 7.53559 ms |

For the biharmonic request, the Torch compiler is 47.9× faster than torch-jet
within PyTorch and 1.05× faster than the handwritten Torch kernel. Strict
Keras/JAX is 15.2× faster than nested folx within JAX. The ordinary Laplacian
remains faster with folx. A handwritten transported JAX kernel was not measured;
do not infer an advantage over that implementation from the Torch comparison.

Torch complete derivative-loss/backward/SGD medians are 0.51969 ms for the
compiler, 0.51949 ms for handwritten code, and 10.73034 ms for torch-jet.
Tracked CPU tensor-storage peaks are 272,072 bytes for compiler/handwritten
and 25,543,720 bytes for torch-jet, approximately 93.9× less tracked storage.
This is not the process-RSS ratio. JAX functional training medians are recorded
separately (0.21664 ms for Keras/JAX and 4.45332 ms for folx); they use a different
compilation scope from the Torch training loop.

### General coupled networks

The 16 → 16 → 16 → 1 network requires ordinary directional Taylor propagation.
The reference compiler uses the same kernel-lowering option but disables the
structural operator selection. The handwritten baseline is the existing
`mlp_jet` implementation.

| Directional order | Torch compiler | Reference compiler | Handwritten jet | JAX Taylor mode | torch-jet | Nested Torch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.09481 ms | 0.08938 ms | 0.09539 ms | **0.03654 ms** | 0.09413 ms | 0.18107 ms |
| 16 | **0.16468 ms** | 0.16506 ms | 0.17475 ms | 0.25584 ms | Incomplete 3-process cell | Refused by order budget |
| 32 | 0.53802 ms | **0.52863 ms** | Timeout | Timeout | Capture error | Refused by order budget |

All compiler and reference-compiler cases pass the complete protocol. The
compiler remains close to its equivalent reference path and handwritten kernel
where those complete. JAX Taylor mode wins at order four; the Torch compiler is
1.55× faster at order 16 on this forward workload. The order-32 compiler is
approximately 1.8% slower than its reference path. Its median backward is
1.44858 ms and its timed derivative-loss/backward/SGD step is 2.19336 ms.

At order 32, handwritten jets and JAX Taylor mode complete forward evaluation
and accuracy checks but all three workers time out during parameter-gradient
preparation. The installed torch-jet version fails capture with
`Multiplicity should be an integer, but got 253.00000000000003.` These outcomes
do not show that those systems cannot run with other settings or larger budgets;
partial warm-forward results remain in the raw evidence without accepted timing
ratios. At order 16, torch-jet has two complete workers and one gradient-stage
timeout, so its three-process comparison is also ineligible.

The coupled report contains 38 successful cells, six explicit refusals, seven
timeouts, and three capture errors. Across all four families, the sweep has
**204 cells: 164 successful, 21 refused, 10 timed out, six errors, and three
memory-budget failures**. All 39 Torch compiler cells and all 18 strict
Keras/JAX compiler cells succeed. Every one of the eight ridge allowance gates
passes. No cell reports a source change. The largest sampled scaled error among
Torch compiler cells is 2.97×10⁻¹⁵.

## Execution and training contracts

`compile_jet` and `compile_operator` accept `kernel_lowering="auto"|"inline"|
"regional"|"packed"`. Automatic mode uses regional directional activation
kernels when the estimated inline graph exceeds 2,048 nodes, with packed
reductions as the compatibility fallback when the public region API is absent.
Graph and arithmetic budgets still count work inside regions.

Ordinary eager execution retains native parameter higher gradients.
`program.hybrid(...)` compiles its pure forward and recomputes the native graph
in backward, including differentiable argument clones that preserve runtime
identity sharing and dependent directions. Its outer wrapper is intentionally
outside fullgraph compilation. `program.compile_parameter_hvp(loss_fn, ...)`
applies `jvp(grad(loss))` before compilation to an inline native graph. It returns
one HVP per unique program parameter, including zeros for unused parameters,
holds spatial inputs and buffers fixed, and does not populate `.grad`.

These paths retain the originally captured semantics even if Python source
attributes change later, and remain independent of `program.compile()` state.
Capture requires distinct example objects for independent argument roles;
ambiguous capture-time aliases to another input, parameter, or buffer are
diagnosed. Runtime aliases remain supported.

Keras keeps `precision="source"` as its default. `precision="strict"` uses
model-local native numerical primitives, preserves declared FP32/FP64 arithmetic
and symbolic dtypes, and rejects explicit downcasts or incompatible policies.
`PrecisionMeanSquaredError` plus the model's public loss hook preserves objective
and gradient precision through compiled training. Supported objective trees,
weights, sample weights, serialization, and optimizer continuation have explicit
tests. Reporting metrics have a separate precision contract. External losses
retain their existing Keras behavior; ambiguous objective trees, implicit masks,
and unsupported additional losses are rejected in the strict objective path.

## Numerical and implementation validation

| Validation scope | Result |
| --- | ---: |
| Core, including slow checks | 2,568 passed |
| Torch default suite | 1,339 passed; 9 slow cases deselected |
| Seven existing Torch slow optimizer cases | 7 passed |
| Complete parameter-AD suite, including order 16/32 | 10 passed |
| Keras full package, Torch backend | 295 passed |
| Keras full package, JAX backend | 292 passed; 3 Torch-specific skips |
| Keras full package, TensorFlow backend | 292 passed; 3 Torch-specific skips |
| Additional cross-backend compiler parity | 22 passed; 1 default-interpreter TensorFlow skip |
| Maintained packaging, documentation, and benchmark protocols | 111 passed |
| Strict mypy, prescribed core/Torch/JAX scope | 204 source files passed |
| Ruff, whole `packages` and `tests` trees | Passed |
| Three clean wheels and isolated installation probes | Passed |

The complete parameter-AD run overlaps eight default Torch cases; its two
order-16/32 cases account for the remaining two slow checks. They use
`aot_eager` and independent exponential formulas. Separate lower-order HVP
cases exercise Inductor. High-order Inductor HVP compilation is not covered by
the ordinary first-gradient performance deadline. The cross-backend skip only
concerns TensorFlow's absence from the default interpreter; the separate
TensorFlow Keras environment completed its full suite.

Tests cover shared parameters, repeated runtime arguments, dependent directions,
live buffers, mutation detection, structured outputs, unused parameters,
previously compiled programs, and captured semantics after source attribute
changes. Strict Keras tests check actual FP64 arithmetic and gradients against
independent high-precision calculations, compiled SGD updates, source variable
identity, native/analytic jet pullbacks, weighted objective trees, and save/load
optimizer continuation. No framework packages or global policies were patched.
The former export-order and import-order failures are fixed without numerical
changes. Canonical license headers, generated inventory, and strict documentation
builds also pass.

The independent 100-digit activation grid covers tanh and sigmoid at orders
1, 2, 4, 8, 16, and 32, including exact symmetry zeros and saturated tails. The
largest sampled recurrence value relative error is 8.49×10⁻¹²; the largest
sampled bias-gradient relative error is 6.56×10⁻¹⁵. These are sampled errors,
not global bounds. Nonlinear incoming GELU paths and derivatives through order
32 have additional 120-digit core regressions, and strict Keras tests check
exact and approximate GELU tail semantics with zero absolute tolerance.

The polynomial provider remains an explicit option with separate accuracy
results. Its sampled relative value error reaches 1.13×10⁴⁷ in a saturated
order-32 case; it returns approximately −1.51×10¹⁷ for sigmoid's exactly zero
32nd derivative at zero. None of those failures are admitted as fast timing
results. Recurrence remains the default.

## Limits of the evidence

The engine differentiates its documented supported tensor programs. It does not
capture arbitrary Python or establish GPU behavior. Exact derivative formulas
still incur floating-point rounding; FP64 is not exact arithmetic. Regional
compilation does not remove first-call cost, and hybrid execution recomputes the
native graph during backward. Neither the structural reductions nor these
execution changes establish a universal 5,000× advantage.

## Evidence and reproduction

The [v2 summary](benchmarks/operator_compiler_v2_summary.json) recomputes
eligibility and comparisons from each input report separately. The
[v2 evidence archive](benchmarks/operator_compiler_v2_evidence.tar.gz) contains
raw results, all samples, exact runtime operands, independent references and
their costs, source snapshots, failure stages, optional-baseline licenses,
validation commands/logs, wheel builds/probes, and reproduction instructions.
Historical cache-disabled failures and the corrected cold experiment remain
separate inputs. Public summaries replace local workspace/home paths; raw
provenance is retained inside the archive.

The matrix manifest and archived `run_matrix.py` specify the four family sweeps
and every environment override. An individual cold acceptance run uses:

```bash
CUDA_VISIBLE_DEVICES='' TORCHINDUCTOR_COMPILE_THREADS=1 \
  taskset -c 2 .venv/bin/python benchmarks/operator_compiler.py \
  --families coupled --orders 32 --methods compiler --cache cold \
  --replicates 3 --repeats 20 --timeout 120 --rss-limit-mib 3072 \
  --output coupled32.json
```

Use an available CPU affinity on other machines and report that difference.
Results are written to `artifacts/` or `OMNIBIAS_SCRATCH`; use a fresh output
directory when reproducing evidence. Optional baseline dependencies are not
installed or replaced by the harness. The separate accuracy grid is
`benchmarks/operator_accuracy.py`, and clean wheel reproduction uses
`scripts/build_operator_wheels.py --output-dir artifacts/operator-wheels-v2`
with a new empty output directory. Setup costs, reference generation, engine
worker wall time, and whole-report elapsed time are retained rather than folded
into warm-forward speedups.

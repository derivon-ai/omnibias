# Structural operator compiler: measured scope

This page preserves the **v1** measurements and their original limitations.
The [v2 execution and precision results](operator-compiler-v2-results.md) cover
regional order-32 compilation, explicit Keras precision, and parameter higher
gradients. The two evidence sets are not pooled.

The experimental compiler now automatically selects ridge contractions,
compressed disjoint-branch rules, affine bottleneck transport, bounded apolar
quotients, or the existing Taylor fallback. Supported derivatives remain
trainable through the original model parameters. The [implementation guide](operator-compiler.md)
states the mathematical and execution contracts.

The result is a structural optimization of higher-order differentiation. These
measurements do **not** establish a universal 5,000× speedup or differentiation
of arbitrary Python. Handwritten kernels and existing AD implementations win
several comparisons below.

## Protocol and evidence

Measurements were made on an Intel Core i7-11850H CPU, float64, batch 32, one
thread per worker. Each eligible cell has three fresh worker processes and
20 synchronized samples per stage, after an independent 80-digit reference
check at three input points (`rtol=2e-9`, `atol=1e-280`). All coordinates and
parameters are runtime operands, recorded in the raw reports. Derivative-only
gradients and optimizer updates are checked before accepting a complete cell.

Results are medians of process medians. The initial family runs overlapped on
distinct physical CPU cores; they are controlled workstation measurements, not
an otherwise-idle-machine certification. Workers record CPU affinity. Disk
compiler caches were persistent in the family sweeps. A separate ridge order-four
run used isolated cold caches. Capture and first compilation/execution are
recorded separately from warm execution. Each worker has a 120-second wall-time and 3,072 MiB
RSS budget, including imports, reference calculation and compilation.

Versions: Python 3.12.13, PyTorch 2.14.1+cu130 (CPU execution), JAX 0.10.2,
folx 0.2.30, and jet-for-pytorch 1.0.0. The torch-jet compiled baseline uses its
public graph-capture and common-subexpression-elimination path before
`torch.compile`. Coordinate seeding is inside compiled baseline wrappers.

The [summary](benchmarks/operator_compiler_summary.json) and
[evidence archive](benchmarks/operator_compiler_evidence.tar.gz) retain individual
samples, failures, configurations, exact source snapshots, runtime operands,
versions, licenses and restoration instructions. A dirty base commit alone does
not identify the measured source. Different source snapshots are kept separate.

## Ridge contraction and release gate

The ridge model has D=H=16. The first sweep recovered the contraction
automatically, but failed the within-20%-or-5-µs allowance against the
handwritten Torch function. Profiling found avoidable input-tree and
source-primal shape bookkeeping. The final implementation skips tree rebuilding
for proved flat positional inputs, retaining every type/dtype/shape guard, and
replays terminal shape operations directly on the derivative. Standard in-place
`Module.compile` avoids an extra module wrapper while preserving hooks; the
benchmark applies this entrypoint uniformly to all module methods.

| Operator | Final compiler | Unchanged handwritten function | Release gate |
| --- | ---: | ---: | --- |
| Δ | 0.07357 ms | 0.08483 ms | Pass |
| Δ² | 0.08846 ms | 0.09179 ms | Pass |
| Δ³ | 0.07767 ms | 0.08217 ms | Pass |
| Δ⁴ | 0.08065 ms | 0.08184 ms | Pass |

Each row comes from a new three-process/20-sample report after the source fix;
the failed first sweep is retained, not overwritten. Treat these small timing
differences as evidence for the allowance, not a general throughput ranking.
The order-four cold-cache repeat also passes. For that compiler case,
capture took a median 1.34 s and compilation plus first execution 9.27 s;
the persistent-cache repeat recorded 1.27 s and 1.74 s, respectively. These
setup costs are excluded from warm derivative timings.

The original sweep's multiple-baseline comparisons remain useful with their
own source snapshot: Δ³ took 0.1015 ms versus 170.15 ms for nested dense JAX;
Δ⁴ took 0.1131 ms versus 198.94 ms for nested folx. Those are approximately
1,676× and 1,759×, not a new ≥5,000× compiler result. Handwritten JAX recurrence
contractions were faster, at 0.02838 ms and 0.03134 ms. No ratio combines the
final compiler timing with an earlier baseline run.

## Affine bottleneck

The model has widths 32 → 2 → 16 → 1, with tanh after the first two maps.
The compiler transports the operator into the two-dimensional latent space.
Its biharmonic state has nine components per scalar, with the live metric
`AAᵀ` connected to parameter gradients.

| Requested operator | Compiler | Handwritten | folx | Compiled torch-jet |
| --- | ---: | ---: | ---: | ---: |
| Laplacian | 0.1710 ms | 0.1465 ms | **0.04583 ms** | 0.1484 ms |
| Biharmonic | **0.1892 ms** | 0.2056 ms | 1.0374 ms | 10.2445 ms |

The Laplacian row is from the initial sweep; the biharmonic row is a final-source
repeat with all four methods. For the biharmonic case this is **5.48× over folx**
and **54.14× over torch-jet**, and 1.09× relative to the handwritten contraction. The folx
comparison crosses framework implementations; the torch-jet comparison stays
within PyTorch. Folx wins the ordinary Laplacian case.

For complete Torch loss/backward/SGD steps, the final biharmonic repeat recorded
0.6857 ms for the compiler, 0.6348 ms for handwritten code and 14.6610 ms for
torch-jet. Thus the handwritten training step remains faster; forward speedups
must not be reused as training speedups.

In the biharmonic forward/backward profiler pass, tracked CPU tensor-storage
peak was 214,728 bytes for both compiler and handwritten implementations,
versus 25,543,720 bytes for torch-jet: approximately 119× less tracked storage.
This excludes untracked allocator/workspace memory. Process RSS and XLA's
static buffer estimates are separate fields, not interchangeable peak measures.

## Disjoint mixed derivatives

The model is `sin(sum_i tanh(w_i*x_i+b_i))`; the request differentiates once
in every coordinate. The compiler proves featurewise independence and uses
branch first derivatives plus a scalar outer tower.

| Mixed order | Compiler | Handwritten formula | Generic fixed-seed mixed jet |
| ---: | ---: | ---: | ---: |
| 4 | 0.06436 ms | **0.04988 ms** | 0.11965 ms |
| 8 | 0.06575 ms | **0.05156 ms** | Graph budget refused |
| 24 | 0.06608 ms | **0.05837 ms** | 16,777,216 coefficients; refused |
| 32 | 0.06598 ms | **0.06136 ms** | 4,294,967,296 coefficients; refused |

Orders four/eight are from the initial sweep; orders 24/32 are final-source
repeats, with each row comparing matching sources. Orders 24 and 32 use 25 and
33 outer coefficients, respectively, and complete
parameter backward and optimizer steps. This demonstrates an exponential
reduction relative to that generic layout for this structure. A refused
baseline has no timing ratio. At order four, the measured improvement over the
generic jet is 1.86×; the handwritten formula remains faster.

## Coupled networks and Taylor-mode comparisons

The coupled model has widths 16 → 16 → 16 → 1. Its directional request requires
the general Taylor fallback.

| Order/execution | Compiler | Existing handwritten jet | Reference compiler | JAX Taylor mode | torch-jet |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4, compiled | 0.1006 ms | 0.1166 ms | 0.09227 ms | **0.03771 ms** | 0.1008 ms |
| 16, compiled | **0.1934 ms** | 0.2194 ms | 0.1987 ms | 0.2939 ms | Initial sweep: one timeout |
| 32, eager | 21.61 ms | **16.92 ms** | 23.03 ms | Not measured | Not measured |

The order-16 row uses the final-source repeat; the other rows retain their
initial-sweep source snapshots. The compiler is near the equivalent reference
jet and 1.52× faster than JAX Taylor mode at order 16 on this workload; JAX
Taylor mode wins at order four. The initial order-32 compiled cells for compiler, reference, handwritten jet and JAX
Taylor mode reached the worker time budget. Torch-jet raised a multiplicity
validation error at order 32. These outcomes do not prove that larger budgets
or other implementations cannot run those cases. Eager order 32 passes values,
parameter gradients and optimizer updates. Compiled order-32 throughput remains
unmeasured under this protocol.

Cross-framework training ratios are intentionally absent: the Torch harness
compiles derivative forward and uses Python loss/backward/SGD orchestration;
the JAX harness compiles the whole functional training step. Both scopes and
their raw times are retained in the evidence.

## Numerical accuracy and frontend limits

The separate 100-digit activation grid covers tanh/sigmoid at orders
1, 2, 4, 8, 16 and 32, including zero, moderate inputs and saturated tails.
The largest sampled relative value error of the recurrence provider was
8.49×10⁻¹². Its largest sampled relative bias-gradient error was 6.56×10⁻¹⁵.
These are sampled errors, not global bounds. Exact parity zeros are checked
separately from relative errors. The polynomial opt-in shows severe high-order
cancellation on this grid, including a nonzero result of about −1.51×10¹⁷ for
sigmoid's theoretically zero 32nd derivative at zero.

The stable tanh recurrence initializes its slope from
`4*sigmoid(2*z)*sigmoid(-2*z)`. It retains the tested 32nd derivative at
`z=20`, about −3.65×10⁻⁸, where subtraction from a rounded unit tanh value
would produce zero. Native parameter gradients and analytic pullbacks have
independent tail regressions.

Keras graph lifting and training tests pass in separate Torch, JAX and
TensorFlow processes. This installed Keras 3.15.1 applies a float64-to-float32
promotion rule in some non-TensorFlow public operations. A saved Keras/JAX
Dense-tanh reproduction has 1.12×10⁻⁷ relative second-derivative error against
the ideal float64 function even with `JAX_ENABLE_X64=1` and floatx set to
float64. It is rejected by the strict float64 timing gate. The implementation
preserves source semantics; no installed framework was patched to obtain an
accepted benchmark. See the [frontend precision contract](jet-compiler.md#keras-preserve-batching-and-original-variables).

## Verification and remaining limits

The final core/Torch run passed 3,839 tests, with seven slow tests deselected by
the repository default. The independent export-order check still fails in the
preexisting `core/verified/locality.py` working-tree change; that unrelated file
was preserved. Ruff passes apart from the separately excluded preexisting
locality-test import ordering, and strict mypy passes 202 source files.

The Keras jet/operator suites pass 103 tests on Torch and 100 on each of JAX and
TensorFlow; the latter two skip three Torch-specific cases. Docs snippets and
benchmark/report protocol tests pass 43 cases. Clean wheels for core, Torch and
Keras were built from source copies, verified by file hashes and installed into
an isolated target. Wheel-only imports, derivative backward and original
variable identity checks passed. Test logs and build/probe manifests are bundled.

The interfaces remain experimental. GPU performance, universal Python capture,
and compiled order-32 throughput under the stated budget are not established.

## Reproduction

Use `benchmarks/operator_compiler.py` with the archived report's exact config.
Optional baselines must be installed in an isolated environment; the harness
does not install or replace frameworks. The archive explains how to restore
the exact measured sources. The evidence packager rechecks every source hash
and recomputes the acceptance gates without pooling different reports:

```bash
python benchmarks/operator_report.py \
  --inputs artifacts/operator-evidence/ridge.json artifacts/operator-evidence/mixed.json \
    artifacts/operator-evidence/bottleneck.json artifacts/operator-evidence/coupled.json \
    artifacts/operator-evidence/coupled_eager32.json \
    artifacts/operator-evidence/ridge_gate_2.json artifacts/operator-evidence/ridge_gate_4.json \
    artifacts/operator-evidence/ridge_gate_6.json artifacts/operator-evidence/ridge_gate_8.json \
    artifacts/operator-evidence/ridge_cold4.json artifacts/operator-evidence/bottleneck_final4.json \
    artifacts/operator-evidence/mixed_final.json artifacts/operator-evidence/coupled_final16.json \
  --accuracy artifacts/operator-release/operator_accuracy.json \
  --output docs/benchmarks/operator_compiler_summary.json \
  --archive docs/benchmarks/operator_compiler_evidence.tar.gz
```

Supply benchmark reports explicitly if that directory also contains diagnostic
JSON files. The activation accuracy probe is `benchmarks/operator_accuracy.py`.

# Merge-readiness audit: presentation and architecture

This audit follows the [open-core migration](open-core-migration.md). It covers
package presentation, executable examples, installation boundaries, the formal
kernel, equation-discovery placement and a bounded GPU optimization diagnostic.
It does not establish scientific priority or validate every downstream product.

## Decisions

| Question | Decision and reason |
| --- | --- |
| Keep the Lean kernel? | Yes, as optional certificate infrastructure. Ordinary training has no Lean dependency. The kernel checks supported finite rational obligations, not an entire physical model. |
| Agent or skill per package? | These are `SKILL.md` maintenance instructions loaded when relevant. They do not launch autonomous agents. `AGENTS.md` routes readers to one capability rule and the relevant skill. |
| Where should equation discovery live? | In the existing extracted symbolic consumer repository. Backend derivative algebra remains here; candidate construction, fitting, selection and scientific validation belong to the consumer. No additional repository or distribution is needed for the existing implementation. |
| Explain the two collapses publicly? | Yes. The README provides the intuition, equations and limitations. A paper should add derivations, related work, proofs and reproducible evaluations. Essential usage context should not wait for publication. |
| Add native CUDA now? | Profile a specific residual/operator first. Compare eager execution, compiler fusion and a native candidate before accepting an additional kernel maintenance burden. |

## Repairs in this pass

- Replaced the sixteen short primitive READMEs with package-specific motivation,
  capabilities, runnable examples, installation instructions, numerical contracts,
  validation routes and licensing links.
- Expanded the twenty-six consumer READMEs and both research-distribution READMEs.
  Their application-specific validation requirements remain distinct.
- Added reproducible bias-collapse and temperature-collapse illustrations in SVG,
  PNG and animated GIF formats. The animations are explicitly explanatory, not
  performance charts. Static alternatives avoid requiring motion to understand them.
- Corrected an invented SOS API name, outdated holonomic recurrence ownership,
  obsolete symbolic notebook links and misleading dependency wording.
- Added symbolic's missing `fractional` and `integral` feature extras, and included
  them in `all`. Optional runtime functionality should have a documented installation
  path rather than rely on a development environment's test dependencies.
- Clarified the optional Lean source-checkout/toolchain requirement and trust boundary.

The existing documentation test runner already discovers `packages/*/README.md`;
these examples now participate in the same CI gate as the root documentation.
Consumer example execution is separate from installation-isolation checks: a
combined ecosystem environment can establish that examples execute, but cannot
establish that each package declares its complete dependency closure.

## Equation-discovery coverage and its boundaries

The extracted implementation contains neural-jet candidate generation, mixed
field partials, sparse equation fitting, integral-column construction, selected
fractional operators and structured expression families. These are useful,
substantial capabilities. They need operator-specific validation rather than a
single claim that every calculus operator is supported.

| Capability | Distinction to preserve |
| --- | --- |
| High-order and mixed derivatives | Derivatives of the represented supported model; derivative accuracy and fit accuracy are different. |
| Candidate equation generation | Sparse fitting and model selection are numerical inference, not automatic proofs of symbolic identities. |
| Integral features | Domain, measure and quadrature assumptions are part of the result. |
| Fractional features | Operator definition, boundary/history convention and analytic-class versus grid approximation matter. |
| Limits and asymptotics | A rational horizontal-asymptote routine does not establish a general symbolic limit solver. |
| Exact recurrence algebra | The shared exact fitter belongs in the Apache difference primitive; the discovery product may consume it. |

Before advertising a unified calculus engine, publish a consumer-owned matrix of
operators, supported expression classes, singularities, domains, approximation
orders and held-out tests. A novelty claim also requires a literature comparison.
These are product work, not reasons to return the consumer to this monorepo.

## Native acceleration: a bounded next step

The reproducible diagnostic is
[`benchmarks/cuda_fusion_probe.py`](https://github.com/derivon-ai/omnibias/blob/codex/pinn-substrate-split/benchmarks/cuda_fusion_probe.py).
Its [recorded output](../benchmarks/cuda_fusion_probe.json) compares eager and
compiled PyTorch on 20,000 float64 tanh inputs at orders four and eight. It records
CUDA-event medians after warmup, compilation plus first execution separately,
and five-point value/gradient comparisons against an independent 80-digit oracle.
It is a local activation-only diagnostic; timing variability and small workloads
prevent extrapolation to training throughput or arbitrary devices. Compilation
can lose on one case and win on another.

A promising first native target is a fused activation recurrence or specialized
Laplacian contraction with enough repeated tensor work to justify fusion. Keep a
portable fallback and require value, parameter-gradient, higher-gradient, dtype,
layout, stream, memory and workload-scaling checks. Avoid materializing a dense
mixed tensor if the requested operator is only a contraction.

PyTorch integration needs dispatcher registration, fake/meta behavior and an
explicit autograd contract; JAX needs tracing-compatible shape information and
explicit derivative rules at its FFI boundary. See the official
[PyTorch custom operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
and [JAX FFI guide](https://docs.jax.dev/en/latest/ffi.html).
Neither a C++ wrapper nor CUDA alone guarantees the fastest implementation.

## Remaining release work

This PR does not publish new versions or create consumer remotes. Before a future
release, publish coordinated dependency versions, confirm rendered package-index
long descriptions, and establish consumer CI in the eventual repository locations.
The extracted source remains local until that separate step is authorized.

Native kernels, a comprehensive equation-discovery operator matrix, full scientific
priority comparisons and industrial application benchmarks are future development,
not implemented features of this presentation pass. Preserve those distinctions
when reviewing marketing language.

## Validation recorded for this pass

The repository suite passed **736 tests**. Strict typing passed for **185 source
files**; Ruff, license headers, generated inventories, capability evidence,
agent-context checks and the strict documentation build passed. The sixteen
primitive README examples executed with **zero failures and zero skips**.

The standalone symbolic fractional/integral profile installed from wheels and
passed its import-origin checks. Full build and consumer-example logs are retained
under `artifacts/readme-audit/`; the PR records the final wheel and consumer results.
These checks supplement, rather than replace, the earlier migration's numerical
suites and license-boundary audit.

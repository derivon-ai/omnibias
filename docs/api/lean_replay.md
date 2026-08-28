# Lean replay checker (phase1-lean-replay)

A finite, **generic** replay checker in the Mathlib-free Lean kernel that
genuinely re-executes a straight-line `Interval` derivation, instead of
merely trusting a Python-reported conclusion — plus a companion
domain-subdivision coverage certificate for branch-and-bound / bisection
searches.

Two payload kinds feed the existing `omnibias.core.proof.lean_check`
bridge, following exactly its existing dispatch / `by decide` convention:

* **`interval_replay_trace`** — an ordered `ReplayTrace` of `literal` /
  `sub` / `mul` steps over `Interval` primitives (`ReplayStep`, built with
  `ReplayRecorder`). `Omnibias/Replay.lean`'s `replayOk` re-executes every
  `sub`/`mul` step against the kernel's own proven `ZInterval` arithmetic
  and checks each recorded envelope genuinely contains the exact
  recomputation. Scoped to exactly this vocabulary: `add` / `div` /
  `sqrt` / `reciprocal` replay is future work, gated on a concrete call
  site and a matching kernel lemma. The one instrumented call site is
  `record_ldlt_diagonal_trace`, a thin, independent mirror of the
  diagonal-pivot recurrence in `omnibias.core.verified.eig_operator`
  (verified to reproduce it bit-for-bit; it does not import, call, or
  alter that module).
* **`domain_subdivision`** — a `DomainSubdivisionCertificate` recording the
  finite list of leaf boxes a branch-and-bound search visited.
  `Omnibias/Subdivision.lean`'s `coversNoGaps` is a purely combinatorial
  check that the leaves tile the claimed starting domain edge-to-edge,
  with no gap and no overlap; it says nothing about any individual leaf's
  own analytic conclusion.

`theorem_prover_verified` is the **same flag** every other omnibias
certificate kind earns from `check_certificate` — it means the same thing
here as everywhere else (a genuine `lake build` re-checked this finite
obligation), so this module reuses it rather than introducing a new,
narrower flag. It is set only by a genuine kernel pass and is never
forged; `record_ldlt_diagonal_trace` and `ReplayRecorder` never touch it.

Two distinct layers catch two distinct classes of tampering: a
structurally malformed trace or subdivision (a bad operand index, an
out-of-order leaf list) is refused by the Python dataclasses themselves;
a numerically tampered but well-formed `ReplayTrace` step is caught only
by the Lean kernel's `decide` genuinely failing to typecheck; a
`DomainSubdivisionCertificate` with a real gap is refused by
`omnibias.core.proof.lean_check.generate_obligation` itself, before any
Lean is emitted, mirroring the existing positive-definite-pivot /
poisedness pattern of only asking Lean to re-derive an obligation Python
already believes holds.

```python
from omnibias.core.proof.replay import (
    DomainSubdivisionCertificate,
    SubdivisionLeaf,
    record_ldlt_diagonal_trace,
    seal_replay_certificate,
)

matrix = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]
pivots, trace = record_ldlt_diagonal_trace(matrix)
assert pivots is not None  # every pivot's sign was certified
assert len(trace.conclusions) == 3  # one entry per pivot D_jj

# Seal the trace as a v1 certificate without invoking `lake` (fast, always
# available); pass run_lean=True to genuinely drive the Lean kernel when a
# toolchain is present -- `theorem_prover_verified` is set only then.
report = seal_replay_certificate(
    trace, claim="LDLT diagonal-pivot recurrence replay", run_lean=False
)
assert report.theorem_prover_verified is False  # run_lean=False never earns it

leaves = (
    SubdivisionLeaf(0, 4, "count_below(1/4) == 0"),
    SubdivisionLeaf(4, 10, "count_below(1/2) == 1"),
    SubdivisionLeaf(10, 16, "count_below(1) == 2"),
)
subdivision = DomainSubdivisionCertificate(resolution=16, leaves=leaves)
assert subdivision.covers_no_gaps() is True
```

No new package: the recorder lives in `omnibias.core.proof.replay`, and
the Lean checkers live in
`formal/omnibias-verified-kernel/Omnibias/{Replay,Subdivision}.lean`.

## API

::: omnibias.core.proof.replay
    options:
      show_root_heading: false
      heading_level: 3

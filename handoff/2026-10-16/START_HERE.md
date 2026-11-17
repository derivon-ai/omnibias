# Omnibias scientific-ML research handoff — 2026-10-16

This is a **research checkpoint**, not a release or a claim that the agreed
implementation plan is complete. Read this file and `NEXT_AGENT_PROMPT.txt`
before executing experiments. The separate transfer ZIP contains the research
workspace, preserved failures, frozen sources, raw evidence available at its
cutoff, and an offline Git bundle. `MANIFEST.json` records every exported file.

## Restore and orientation

1. Pull branch `codex/omnibias-gpu-handoff-20261016` from
   `git@github.com:derivon-ai/omnibias.git`. The exact commit is in the ZIP's
   `export_state.json`. Offline alternative:
   `git clone -b codex/omnibias-gpu-handoff-20261016 git/omnibias.bundle omnibias`.
2. Verify the ZIP using this commit's `verify_export.py`, then extract it.
3. Read `RESULTS_AND_LESSONS.md`, `NEGATIVE_RESULTS.json`, `AGREED_SCOPE.md`, and
   `A100_LSF_CONTINUATION.md` in this directory.
4. Research files are under `research/bias-collapse-research/` in the ZIP.
   The most recent implementation work is in `completion_execution_20261013/`.
5. Read that workspace's `SESSION_STATE.txt`,
   `completion_requirements_checklist_v15.json`, and the latest authoritative
   source clarification:
   `diagnostics/continuation_post_checkpoint_20261016T044839Z.json`.
   Follow `SOURCE_MAP.md` for the later proposals and evidence.

The repository snapshot includes earlier Gaussian/core research and other
pre-existing workspace changes as well as the operator optimization work. It
preserves the working state for transfer; it does not assert that every included
branch was implemented or validated in the final optimization experiment.

## Non-negotiable evidence distinctions

- **Source review, AST checks, hashes, mocks, and checksum verification are not
  numerical qualification or benchmark passes.**
- The original frozen private stage recorded 68 CPU checks and 68 checks on an
  asserted NVIDIA T1200. These predate subsequent contract/runtime corrections.
  Do not attribute those passes to the new combined candidate.
- The newest shared/worker/owner/interpreter/saved24 finite source compositions
  are independently reviewed. Operational publications/case plans remain
  unavailable; actual numerical/scientific/resource/performance acceptance is
  not established by those reviews.
- The 78-entry checklist preserves historical statuses. Its older descriptions
  of source work can be stale. Later reviewed source includes containing-domain
  and whole-block reuse; this does not turn their UNEXECUTED tests into passes.
- No new public promotion through `omnibias.jax.experimental` is earned. The
  `_experimental_stage` is private, with its historical exports/gates preserved.
- General nonlinear PDE-to-LP reduction, a global neural-network optimizer,
  universal accuracy/speed superiority, and historical mathematical priority
  remain unachieved/unestablished. They are not prerequisites secretly solved
  by this checkpoint.

## Live laptop experiment and export cutoff

At handoff preparation the original laptop process was still running. Its
schedule has 10,548 main records, 36 cache records and 234 PDE records: **10,818
total**. Unified-exec session ID 24105 is local to that old session; it cannot
be resumed by that number on another machine.

The dated filename-only frontier used for this export is recorded in
`export_state.json`. Existing result/monitor file pairs are **not** a count of
passed, independently audited or scientifically accepted runs. In-progress
attempt files and later outputs are explicitly excluded. A new machine must
not treat an incomplete export as a completed original schedule.

The old process is left running. Do not restart its schedule on the cluster or
replace its failures. Start a separately named, newly frozen A100 experiment
with new hardware/environment/source bindings. The ZIP's packaging window is
recorded because source export/compression can contend with a live benchmark;
do not assume overlapping laptop timings establish a quiet final comparison.

## Immediate priorities

1. Preserve history and establish the A100/LSF environment without changing old
   manifests or applying broad path-replacement edits to historical evidence.
2. Compose a runnable final candidate from the reviewed private proposals;
   operational bindings intentionally still need actual target validation.
3. Execute pending numerical, contract, gradient, high-precision, resource and
   saved-science suites before claiming the candidate qualifies.
4. Prioritize practical derivative orders **2–6** for new optimization work.
   Preserve the wider agreed experiment obligations and frozen laptop schedule.
5. Finish matched evidence, complete training/PDE targets, report all failures,
   and integrate only exact final-package configurations that earn their gate.

Do not rereview tens of thousands of historical source assertions as a
substitute for running the missing numerical tests. Do not repeatedly retry a
known unsuccessful configuration without a specific changed hypothesis and a
new experiment identity.

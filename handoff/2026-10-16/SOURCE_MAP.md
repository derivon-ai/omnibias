# Source and evidence map

Paths in this file are relative to exported `research/bias-collapse-research/`.
`C/` below means `completion_execution_20261013/`.

## Read these first, rather than rediscovering the source pipeline

| Role | Path |
|---|---|
| Latest authoritative source/status clarification | C/diagnostics/continuation_post_checkpoint_20261016T044839Z.json |
| Latest checkpoint | C/diagnostics/continuation_checkpoint_20261016T044542Z.json |
| Source-preservation ledger316 | C/diagnostics/continuation_source_ledger_20261016T044542Z.json |
| Independent ledger review | C/diagnostics_continuation_044542_independent_20261016/independent_metadata_review_v1.json |
| All78 requirement rows | C/completion_requirements_checklist_v15.json |
| Historical CPU68 | C/diagnostics/package_validation_cpu_05.json |
| Historical assertedGPU68 | C/diagnostics/package_validation_gpu_04.json |
| Original current experiment protocol | C/protocol.json |
| Cache/PDE protocols | C/protocol_cache_lifecycle.json; C/protocol_pde_completion.json |
| Latest candidate proposal/recipe | C/pending_candidate_identity/proposal_manifest_v1.json; C/pending_candidate_identity/source_recipe_v1.json |

The ledger and checkpoint contain exact hashes, origins and snapshots. Preserve
them unchanged. An old absolute source path is provenance, not a relocation
instruction. Work in a NEW target-specific composition if paths/environment change.

## Reviewed source closures, actual runtime acceptance still pending

| Component | Concrete source/proposal | Digest or scope |
|---|---|---|
| Whole strict owner | C/pending_identity_strict_training_owner/proposal_manifest_v1.json | a18234485eb2f4a64202a732602a8673280912fef084807f6a9b702aa156e6be |
| Owner V4 source manifest | C/pending_identity_strict_training_owner/source_preparation/launcher_review_input_v4.json | 09e6f51ddf6c04e9ea5362d5727f6e1f3aa2475081242ad4d83b13e218097df2 |
| Strict sharedV2 | C/pending_identity_strict_training_shared_revision_v2/proposal_manifest_v2.json | Resolve file name if needed from source ledger digest ea626b21af6c247e3b97614b507d3c4db48254dd42aa6dba860c5238b0de9441 |
| Strict worker | C/pending_identity_strict_training_worker/proposal_manifest_v1.json | 328242a0b99d2c1049a4451d5634d75d646df55ecd746ab25939d23e8b0af9e8 |
| Saved24V4 | C/pending_identity_saved24_validation_revision_v4/proposal_manifest_v4.json | cba1f009724c549741bca9affb7ac4fc3ff18a6d98b39190eeba400cb364b5e9 |
| Interpreter identity | C/pending_identity_strict_training_owner/interpreter_identity.py | 283b3926924c9d33e9a5f3e80e12ade05e9e6cc462449f1724f1ccd59a87f768 |
| GPU saved audit | C/pending_identity_strict_training_owner/gpu_audit.py | a2687a190c3dca5b23c14e9526685cf7b8b1892dbfcfc8f47ddb8ed6b69d3cb2 |
| Whole-block external consumer | C/pending_identity_wholeblock_domain_reuse/revision_v1/proposal_manifest_phase1_v1.json | f566ca2711951ca7b68a4c43d8ca8e447a01bb553493cd68e885bf649fc2e6ed |
| Saved-science composition | C/pending_identity_moment_science_revision_v2/proposal_manifest_v2.json | d5200d3a822a9901e234f087deae8516bc542672c62553267fd8dad8a078de3b |
| Containing-domain reuse | C/pending_runtime_domain_reuse/proposal_manifest_v4.json | Reviewed by C/diagnostics/domain_reuse_root_source_review_v4.json;18+18 numerical cases UNEXECUTED |
| Whole-block reuse | C/pending_identity_wholeblock_domain_reuse/revision_v1/runtime_block_domain_reuse.py | New whole-scan reuse source; numerical cases UNEXECUTED |
| Contract hardening | C/pending_contract_hardening/revision_manifest_9.json | Nine-source proposal/57 regression definitions, not executed |
| Runtime-prepared | C/pending_runtime_prepared/proposal_manifest_v2.json |18 numerical cases UNEXECUTED; broader reuse is in later proposals above |

Some proposal basenames in historical discussions are shorthand. Use the
ledger's exact path+SHA record, not a guessed filename; the archive includes the
whole tree. `metadata` views do not authorize native child execution.

Current owner source pins: lane7e695375950533fea7a5f18f469b251385cd36fa93025ab2d72fa67a2eea726c;
core dfffcd90a3c002f683e57fc8d772a4fa66674cbfdda4f0b44c6a15535d698f6f;
interpreter283b...; audit a268.... Owner248 source/snapshot pairs and309 aliases
are integrity/composition checks, not numerical tests. Source publication slots,
current37/core181/sixversions/hardware bindings and native caseplans remain actual
target-validation obligations. Do not replay source review to inflate progress.

Known reviewed fixes: helper configuration before use; worker source-receipt
joins; canonical binary separate from lexical venv invocation; fresh deadline
immediately before Popen; failed-constructor accounting retained; one shared
audit-work debit; all owned GPU allocations checked; descriptor55 consistency;
failed-read charges retained; success cannot survive failed cleanup.

Historical V1/V2/V3 defects and reviewer errors remain preserved. See
NEGATIVE_RESULTS.json. No caps were raised and no scientific tolerance loosened.

## Mathematical and implementation entry points

C/EXECUTION_DERIVATIONS.txt; C/VARPRO_MATHEMATICS.txt; C/planner.py; C/bounds.py;
C/streaming.py; C/device_training.py; C/prefetch.py; C/variable_projection.py;
C/explicit_fft.py; C/cache_manager.py; C/low_rank_operators.py.
Older notes can predate a source correction; actual held source and current
review metadata control that narrow implementation detail.

Historical experimental directories contain complete reports and negative raw
records: operator_selection_20261011, operator_specialization_20261011,
ridge_taylor_comparison_20261011, original_kernel_audit_20261011,
generic_api_audit_20261012, reuse_planning_20261012, reuse_execution_20261012,
overlap_companion_20261011, modal_cubic_benchmark_trials and the root LaTeX
reports/certificates/research records. Read the final scoped report before raw logs.

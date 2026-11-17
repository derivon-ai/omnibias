# Continue on the shared A100/LSF cluster

The human reports about20 NVIDIA A100 allocations with20–80GB memory and IBM
LSF, shared with other users. Exact CPU/RAM, driver, CUDA and queue configuration
are not yet known.20GB may be a MIG slice, not a whole A100. Record actual model,
UUID, MIG profile/visibility and assigned physical CPU topology inside the job.

## Restore first

- Verify ZIP and Git commit. Keep the exported tree immutable as historical
  evidence. Copy a new working experiment directory for target-specific work.
- Copy BOTH repository and research tree. Code-only Git pull is insufficient:
  pending prototypes, frozen sources, references, failures and raw results live
  in the separate research export.
- Read repo AGENTS.md and relevant skills. Recreate Python3.12 and the locked
  workspace (`uv sync --all-packages --group docs` is the repo development command).
  Historical measured JAX/jaxlib version was0.10.2; the old CUDA13 plugins were
  installed in an isolated /tmp target and are not portable executable binaries.
  Recreate compatible plugins/libraries on the cluster; do not silently upgrade
  and attribute results to the old toolchain. Any necessary change gets a new
  environment/protocol identity. The archive includes old environment records.
- Enable JAX binary64 before arrays. Assert the actual GPU backend/device.
  CPU fallback under a GPU label was a previous setup error, not GPU evidence.
- Historical absolute paths and hashes must stay preserved. New working copies
  need an explicit old-to-new root map, regenerated current-source/environment
  bindings and a new protocol. Do not sed-rewrite old ledgers or pretend hashes
  remain valid. PUB=None and process authority are intentional guard boundaries.

## Allocation policy

Prefer an exclusive whole GPU, MPS disabled, reserved host cores and adequate
RAM; queue names/model strings and exclusivity permissions are site-specific.
An illustrative LSF GPU requirement is:

```
-gpu "num=1:mode=exclusive_process:mps=no:j_exclusive=yes"
```

Choose a queue and CPU affinity via local policy. This is an example request,
not an executed job or a claim that every LSF site supports identical settings.
Full A100 and each MIG profile are distinct hardware targets. Do not pool their
timings. Available RAM does not automatically raise frozen3GiB process budgets.

Run independent workload groups on separate isolated allocations. Keep all
methods and the three fresh process repetitions for each matched group serial,
with frozen rotating order and equal resources. Avoid simultaneous compilation,
profiling, backup or validation on measured resources. Record allocation IDs,
resource scopes, contention and monitor limitations. Do not assume twenty GPUs
make one derivative program twenty times faster: multi-device execution is a
separate implementation/qualification question.

Useful official references:
- https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=o-gpu
- https://docs.nvidia.com/datacenter/tesla/mig-user-guide/mig-device-names.html
- https://docs.jax.dev/en/latest/multi_process.html
- https://docs.jax.dev/en/latest/installation.html

## Do first on the new target

1. Establish a resource-isolated development allocation, actual versions and
   numerical/backend assumptions. Preserve a machine-readable environment record.
2. Resolve the reviewed candidate/consumer/science/worker/owner source DAG into
   a NEW actual composition. Use exact existing proposals/reviews, not guessed
   receipts; the copied historical publications intentionally remainNone.
3. Run correctness, negative contract, small complete gradients/high precision,
   low-order stability, rollback/resume, low-rank and readout/VP suites. Qualify
   actual emitted outputs/objective/first gradients. A process that refuses is
   evidence of refusal; distinguish conservative proof limits from wrong values.
4. Freeze calibration/validation/measurement data and A100 schedule only after
   the candidate is stable. Compare equally structured native/analytic Taylor,
   fused manual, canonical existing compiler and applicable folx baselines.
5. Prioritize2–6 spatial derivatives and total residual/training costs. Finish
   the wider agreed tests/failure classifications separately; do not republish
   a selected kernel win as universal advantage or PDE time-to-target superiority.

## Laptop continuation and source-export timing

Session24105 and its incomplete original schedule remain on the laptop. Do not
submit that old queue again on A100. Ask the human/original agent for a final
evidence delta once its remaining jobs finish. Existing records retain their
source/hardware identity and every failure. Archive creation is local work that
may contend with laptop timings; the export records its interval. Overlap
requires scoped audit and cannot silently become quiet final-source evidence.

Native execution, GPU monitoring/identity/MIG handling and interval analysis on
the cluster still require actual implementation tests. The strict old owner
was source-reviewed for the laptop premises; it is not automatically an A100/MIG
resource certificate. Keep the original separate actor/evidence budget scopes.

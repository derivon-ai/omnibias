# Hardware: what was measured and what was not

The planner chooses a program by predicted time. How far that prediction can be trusted on a device depends on whether the device was
measured, so this page lists the devices that were, the outcome, and the devices that were not. A device that is not listed is
**unmeasured**, not assumed to behave like its neighbour.

## What was run

Five devices, one run each, JAX 0.10.2, float64, the same harness as the [dispatch gate](operator-planner.md#dispatch-regret): 40
problems per device (14 Laplacian, 6 bi-Laplacian, 8 divergence, 6 log-determinant, 6 residual-Gram; none reused from the gate), every
feasible program timed in turn, 300 s cap per program. The *regret* of a cost model on a problem is the measured time of the program it
ranks first over the best measured time. Three models are scored on each device: the generic constants the planner uses when no
calibration is loaded; the frozen A100 model of the gate (`a100_transfer`, in the record); and a 5-fold cross-validated fit on the
device's own 40 problems. The rule registered before any device was read: a calibration file ships for a device class only if its
cross-validated median regret is at most 1.10, its worst at most 1.5 and the median no worse than the generic model's.
Record: [`hardware_matrix.json`](benchmarks/hardware/hardware_matrix.json).

| device | GEMM float64 / float32 / TF32 (TFLOP/s) | bandwidth (GB/s) | launch (µs) | generic model: median / worst regret | own fit, cross-validated: median / worst | calibration shipped |
|---|---|---:|---:|---|---|:-:|
| A100-SXM4-80GB | 17.5 / 18.7 / 123.9 | 1,639 | 49 | 1.00 / 3.03 | 1.00 / 3.03 (+1 unmeasured) | no |
| A100-SXM4-40GB | 17.8 / 18.7 / 123.1 | 1,328 | 53 | 1.00 / 41.74 | 1.00 / 15.63 (+1 unmeasured) | no |
| H100 80GB HBM3 (SXM) | 55.3 / 51.6 / 365.4 | 2,950 | 27 | 1.02 / 3.90 | 1.02 / 3.21 (+1 unmeasured) | no |
| A30 | 7.6 / 9.6 / 54.6 | 801 | 27 | 1.00 / 2.64 | 1.00 / 2.64 (+1 unmeasured) | no |
| CPU (JAX CPU backend, one compute node) | 0.08 / 0.22 / 0.21 | 17 | 10 | 1.00 / 1.45 | 1.00 / 3.12 (+1 unmeasured) | no |

**No calibration file ships for any device.** The worst-case regret of the own-device fit is 2.6 to 15.6 on every device, against the
registered 1.5. "+1 unmeasured" is one problem (`lap2/0`) on which the fit ranked a program first that has no time, because an
earlier program in that cell hit the 300 s cap and the cell ended; the registered scoring counts it as infinite regret and the table
shows the largest finite regret beside it. The outcome does not depend on that choice: the finite worst case already exceeds 1.5 on
every device.

## How to read it

- **The median is 1.0 everywhere and means little.** Most problems have one obviously good program, and any ranking finds it. The
  tail is what the gate is about, and the tail is not under control on these 40 problems per device with a model fitted on 40.
- **The winning program changes with the device.** `projected` wins 21 of 40 on the A100-80GB, 17 on the A100-40GB, 16 on the H100,
  24 on the A30 and 25 on the CPU; `dense_jacobian` wins 1, 4, 4, 4 and 3, `forward_laplacian` 7, 5, 4, 4 and 4
  (`winners` in the record). That is the reason a per-device model exists; it is not evidence that this one is good enough.
- **The A100-40GB is not the A100-80GB.** The generic model's worst regret is 41.7 on the first and 3.0 on the second, and the model
  fitted on the 80GB device and scored on the 40GB one (a sibling of the same class) has median 1.02. A calibration does not transfer
  between two cards of the same family without a check.
- **Launch latency sets the floor.** 49 µs on the A100, 27 µs on the H100 and the A30, 10 µs on the CPU. Below about ten times that
  floor a program's cost is the launch, not the arithmetic, and the ranking is decided by how many launches a program makes.
- **TF32.** The probe measures a TF32 GEMM at 6 to 7 times the float32 rate on the A100 and H100; whether a derivative program may
  use it is a precision question, not a speed one: see [precision](precision.md).

## What was not measured

| device or back end | status |
|---|---|
| H100 PCIe, H200, L40, V100 SXM2, V100 PCIe | requested on the shared cluster on 22 Oct 10:07 PDT (one job each); none had started 15.5 hours later (23 Oct 01:36 PDT), when they were cancelled. **Unmeasured.** |
| TPU | no access. **Unmeasured.** |
| AMD GPUs (ROCm) | not attempted. **Unmeasured.** |
| PyTorch on any of the above | **not in this matrix**: the PyTorch planner ranks with the heuristic and has no per-device calibration; its tables are per call and per device in [operator planner](operator-planner.md#pytorch) and the baselines results. |
| float32, other activations, networks that are not dense chains | the problems are float64 tanh MLPs; the [precision](precision.md) tables and [coverage](operator-coverage.md) cover other cases per program, not per device. |
| multi-device runs | measured on the A100 only: see [distributed](distributed.md). |

Each row is one run on one node of a shared cluster; a ratio near 1 sits inside the run-to-run variation of a launch-bound call.

## Using it

Run `mode="find"` on a device that is not in the table above and wherever a wrong choice costs more than a measurement: it times the
feasible programs and does not use the cost model. `explain()["cost_model"]` says `"defaults"` when no calibration is loaded, which is
the case on every device today; a CPU with JAX reads `"defaults+cpu class prior"` (constants set from the hardware of one
core, not fitted, see [the planner page](operator-planner.md#dispatch-regret)) and PyTorch on an accelerator reads
`"defaults+counted torch launches"` (the kernels one PyTorch call launches, counted). To calibrate a new device, `omnibias.core.plan_calibration.fit_cost_model` takes the same measurement
records the harness writes (`omnibias_experiments/hardware_20261022/`).

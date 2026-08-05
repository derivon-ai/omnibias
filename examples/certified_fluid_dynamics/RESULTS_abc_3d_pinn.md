# Track A results: 3D incompressible Navier-Stokes PINN vs exact ABC flow

Validated numerical solution of **one specific 3D incompressible Navier-Stokes
instance** with omnibias closed-form derivatives and hard incompressibility.
This is **not** a global-regularity statement (`unproven_claim = False`); it is
a trained neural field scored against a known exact solution. See
[`../../docs/scope-and-guarantees.md`](../../docs/scope-and-guarantees.md) §3.

Driver: [`run_abc_3d_pinn.py`](run_abc_3d_pinn.py).

## Problem

Velocity is represented as `u = curl(A)` via `VectorPotentialField`, so
`div u = 0` holds **by construction** (closed-form curl). The prebuilt
`NavierStokes(form="primitive_3d")` residual is trained on interior collocation
plus an initial condition, and scored against the exact **decaying
Arnold-Beltrami-Childress** solution (an exact unsteady NS solution, `f = 0`):

```
U(x)   = (sin z + cos y,  sin x + cos z,  sin y + cos x)
u(x,t) = exp(-nu t) U(x),   p(x,t) = -0.5 exp(-2 nu t) |U|^2
```

## Environment

Run on a GPU node (`python`, torch 2.9.0+cu128, float64).

| | GPU | Wall time |
|---|---|---|
| v2 (reported) | NVIDIA H200 | 11m 26s (20k steps) |
| v1 (ablation) | NVIDIA A100-40GB | ~25m (diverged) |

## Result (v2)

Config: `K=6`, `time_hidden=128`, `time_depth=1`, `batch=4096`, `nu=0.05`,
`T=1.0`, `lr=3e-3` with cosine decay, `ic_weight=25`, 20 000 Adam steps.

| Metric | Value | Meaning |
|---|---|---|
| **rel-L2 velocity error** | **0.0097** | < 1% vs exact decaying ABC over `t in [0, 1]` |
| **max \|div u\|** | **0.0** | exact incompressibility (hard cage) |
| RMS momentum residual | 0.069 | interior NS residual |
| final IC loss | 2.0e-6 | initial condition matched |

Convergence (total loss): `25.2 (step 0) -> 2.47 (5k) -> 0.111 (10k) ->
0.0537 (15k) -> 0.0099 (20k)`; IC loss `1.0 -> 5.6e-5 (10k) -> 2.0e-6 (20k)`.

## Ablation: LR decay matters

The first run (v1: fixed `lr=2e-3`, `ic_weight=10`, no decay) became unstable
late in training (PDE loss into the thousands) and ended at rel-L2 = **2.52**
(worse than the zero field). Adding cosine LR decay (v2) fixed it. Incompressibility
stayed at machine zero in **both** runs -- that is structural, not learned.

## Reproduce

```bash
# smoke (CPU, seconds)
python -m examples.certified_fluid_dynamics.run_abc_3d_pinn --smoke

# full run on a GPU node (submit through your cluster's GPU batch wrapper)
python -u -m examples.certified_fluid_dynamics.run_abc_3d_pinn \
  --steps 20000 --K 6 --lr 3e-3 --ic-weight 25 \
  --out-dir "artifacts/omnibias_runs/abc3d_gpu_v2"
```

Large artifacts (checkpoint `abc3d_field.pt`, full `metrics.json`) are persisted
under `artifacts/omnibias_runs/abc3d_gpu_v2/`, not committed to the repo.

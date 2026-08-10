# Beyond DeepMind (Phase 5 structure layer)

After CCF Rung-2 (`whole_line_certified=True`), omnibias adds a **structure
layer** that DeepMind’s single-patch PINN loop does not automate:

| Layer | Module | Role |
| --- | --- | --- |
| 5a | `omnibias.pinn.partition` + `phase5_beyond.partitioned_near_far_residual_report` | Near/far partitioned self-similar patches; must beat single-patch residual |
| 5b | `omnibias-tab` meta-features via `ansatz_router_meta_features` | Soft-tree router over family / rung / dictionary size |
| 5c | `omnibias-logic` checklist via `obligation_planner_report` | MaxSAT ordering of **finite** CAP/Lean obligations only |

```python
from omnibias.pinn.jax.discovery import phase5_beyond as p5

gate = p5.phase5_entry_from_status({"gates": {"rung2_earned": False}})
assert gate.allowed is False
blocked = p5.blocked_phase5_bundle("waiting_for_rung2")
assert blocked["honesty"]["navier_stokes_proof_claim"] is False

# Helpers are earnable only after Rung-2; they never claim Clay NS.
part = p5.partitioned_near_far_residual_report(
    single_patch_residual=1e-2,
    partitioned_residual=1e-3,
    residual_threshold=1e-2,
)
assert part["skill"] > 0.0
```

!!! danger "Entry gate"
    Phase 5 must not start while Rung-1 residual is the blocker. Discrete
    packages never replace CubicGN / Martens–Grosse on the fluid residual.
    Continuum NS / RH / Yang–Mills literals are forbidden in the obligation
    planner.

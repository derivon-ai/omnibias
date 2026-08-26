# 08-12 Jet-MPC optimizer

## 1. Thesis and status

Receding-horizon control on the directional jet of
`phi(s) = L(theta + s d)`: the planned law is the first control of
08-11 LQR; the applied step is that control boxed by an input limit
and an optional Lagrange trust radius.

- **Status**: shipped (G1–G4 CI; receding LQR-with-box; not plant MPC)
- **Depends on**: 08-01, 03-12, 08-11
- **Blocks**: none

### Operator card

- **Benefit.** Look-ahead is the 08-11 horizon; constraints are a
  box and a sound remainder radius; only `u_0` is applied.
- **How it works.** Call `scalar_finite_horizon_lqr`. Clip `u_0` to
  `[-u_max, u_max]` and, if requested, to the certified truncation
  radius. Refuse if the applied remainder exceeds a tolerance.
- **Strength.** Quadratic bowls and jets with a known `|phi^(N+1)|`.
- **When to use.** After 08-11 when the Newton / LQR step must
  respect a trust or actuator box.
- **When not.** As chemical-plant / vehicle MPC. As a general
  nonlinear programme. As a global solver. Not CCF stretch.
- **Accuracy floor.** Quadratic model + first-control box. A full
  state-horizon QP is not this spec.

## 2. Where it lands

`omnibias.core.control_mpc` plus thin
`omnibias.{torch,jax}.optim_mpc` twins. No new package. Reuses
`omnibias.core.control_lqr` and `omnibias.core.line_search`
remainder radii.

## 3. Prior art in omnibias

- Spec 08-11 — the unconstrained receding law *is* `K_0`. Do not
  re-derive the Riccati.
- Spec 03-12 — Lagrange remainder / certified radius.
- Spec 08-10 — PID on the same jet. Different gains.
- Spec 08-09 — I/O filter after a step. Distinct.
- `omnibias.control` CBF-QP — plant safety, not a `theta` trainer.

**Confirmed gap.** No trainer that recedes an 08-11 law under a box
and a remainder trust.

## 4. Mathematics

Unconstrained receding LQR applies `u_0 = -K_0 phi'(0)` from the
horizon-`N` scalar Riccati of 08-11. This spec sets

```
s* = clip(u_0, -limit, limit)
limit = min(u_max, r_trust)
```

where `r_trust` is optional and comes from
`certified_truncation_radius` (03-12) when a bound on
`|phi^(N+1)|` is supplied. If `refuse_uncertified` and the Lagrange
remainder of `s*` exceeds `remainder_tol`, the step is `0`.

Founding **bias collapse** (`delta -> 0`) supplies the jet. No
temperature collapse (`beta -> inf`, feasibility). Faà di Bruno is
the chain rule. This is **not** a plant MPC and **not** a dense
multi-control QP over state constraints along the horizon.

## 5. Worked example

`phi(s)=(s-1)^2` from `0`. Unconstrained (`u_max=4`) recovers
`s*=1`. `u_max=0.25` boxes to `0.25` and reports `reason="boxed"`.
Both reports set `receding=True`.

## 6. Proposed API

Shipped.

```python
from omnibias.core.control_mpc import JetMPCConfig, mpc_step_from_derivatives

report = mpc_step_from_derivatives(
    (1.0, -2.0, 2.0),
    config=JetMPCConfig(q=0.0, r=0.0, qf=1.0, horizon=1, u_max=0.25),
)
```

Torch / jax: `jet_mpc_step(loss_fn, params, direction, config=...)`.
Eager; do not `jit` / `torch.compile` the driver.

## 7. Practical use cases

1. **Boxed Newton** on a bowl.
2. **Trust-clipped** LQR when a `|phi^(N+1)|` bound is known.
3. **Not** autonomous-vehicle trajectory MPC. **Not** CCF Hilbert.

## 8. Acceptance gates

- **G1 recede / box.** Unconstrained bowl recovers `s*=1`.
  `u_max=0.25` boxes. `receding` is true.
- **G2 skill.** Random bowls median `|s*-c|<1e-12`. Box clip and
  remainder refuse both fire.
- **G3 honesty.** Payload forbids global min, plant MPC, general QP,
  DARE, activation-Riccati-as-model, stretch, skipped chain rule.
- **G4 parity.** Torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/jet_mpc.py`
- Smoke: `docs/benchmarks/jet_mpc_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/jet_mpc/`
- CI smoke after implementation.

## 10. Honesty and scope

- Not a global minimum of a deep nest. Faà di Bruno is the chain rule.
- Hilbert × dictionary remains `~1e-1`; this trainer does not set or
  weaken `CCF_STRETCH_RESIDUAL_GATE`.
- Not plant MPC. Not a general horizon-QP. Unconstrained law is 08-11.
- Bias collapse supplies the jet; it does not certify a continuum
  solution.
- Distinct from 08-09 (I/O enclosure of the net, not of `phi`).

## 11. Open questions and risks

- **A full state-constrained horizon QP is leftover**, not this
  ship. Tests must not claim it.
- **Falsifier.** If unconstrained MPC disagrees with 08-11 `K_0`,
  the receding law is wrong; do not ship.

## 12. Implementation checklist

- [x] `mpc_step_from_derivatives` reusing 08-11
- [x] Torch / jax `jet_mpc_step` twins
- [x] Tests: G1–G3, LQR agreement, box, remainder, parity
- [x] `benchmarks/jet_mpc.py` plus smoke JSON
- [x] Index row in `theory/README.md`

---

## Repo invariants this spec must respect

Check these before submitting an implementation.

- **Pure core**: no torch, jax, tensorflow or keras imports from
  `omnibias.core`. Pure-Python math lives there and every backend imports it.
- **Bit-identical twins**: torch and jax implementations must agree exactly.
  Polynomial coefficients come from `omnibias.core.polynomials`; never fork them
  per backend.
- **Default dtype**: use the framework default
  (`torch.get_default_dtype()` / `keras.config.floatx()`), never a hardcoded
  `float32`.
- **Vendor-neutral language**: no scheduler commands, vendor names, internal
  hostnames, usernames, or absolute local paths in tracked files. Artifacts go
  to `$OMNIBIAS_SCRATCH`, defaulting to a repo-relative `artifacts/`.
  `packages/omnibias-core/tests/test_no_leakage.py` enforces this across the
  whole readable surface.
- **Terminology**: distinguish the founding bias collapse (`delta -> 0`, yields a
  derivative) from temperature collapse (`beta -> inf`, yields a 0/1 step). The
  older penalty phrasings are retired and guarded by `tests/test_terminology.py`.
  Every new `**/relaxation.py` must carry the cross-reference note and be listed
  in `PENALTY_FILES` in
  `packages/omnibias-core/tests/test_concept_terminology.py`.
- **Executable docs**: if any part of this lands in `docs/` or a package README,
  every fenced Python block is executed by `tests/test_docs_snippets.py`. Verify
  calls against real signatures; opting out needs a directive with a stated
  reason.
- **Earned flags**: `theorem_prover_verified` is set only by a genuine
  `lake build` pass of the Mathlib-free kernel, and `mathlib_verified` only by a
  genuine pass of the Mathlib-backed project. Asserting either without a pass
  blocks the verdict.
- **Typing tier**: the strict CI gate covers `core`, `torch`, `jax` and
  `ferminet`. Newly authored modules should be written strict-clean regardless
  of tier; curated beta modules can be added to
  `scripts/mypy_strict_allowlist.txt` once
  `mypy --strict --follow-imports=silent <file>` is clean.

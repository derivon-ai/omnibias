# 09-27 Parameter-space jets

## 1. Thesis and status

Treat a PDE parameter `μ` as a **jet coordinate** so one forward
pass yields mixed partials `∂^{α,β} u / ∂x^α ∂μ^β`, not only
query-coordinate derivatives at a frozen `μ`.

- **Status**: concept
- **Depends on**: 01-10, 09-01
- **Blocks**: none

Related, not re-specified: 09-16 (MAML on `theta`), 09-20 (path of
problems), 09-22 (Newton-on-`x`).

### Operator card

- **Benefit.** Sensitivity, continuation, and Newton-on-`μ` without
  finite differences in the parameter.
- **How it works.** Put `μ` on the same `mlp_jet_mv` / OMBU trunk as
  `x` (or a jet branch). One multivariate jet is every mixed partial
  up to total order `N`.
- **Strength.** Parametric heat / Burgers slabs already shipped.
- **When to use.** After a first-bet gate; when `∂u/∂μ` is the object.
- **When not.** As a ParamPINN package. Today's `pde_params` encoder
  plus autodiff is **not** this spec. Not NS. Not CCF.
- **Accuracy floor.** Closed form only if `μ` goes through the tower.
  Jet truncation `R_N`.

## 2. Where it lands

`omnibias.pinn.operator` — extend the DeepONet trunk / field so `μ`
is a jet coordinate. Fails the package test (same domain as the
shipped operator submodule). No new package.

## 3. Prior art in omnibias

- `omnibias.pinn.operator` — `ParametricOperatorSlab`,
  `make_parametric_heat_slab` / `make_parametric_burgers_slab`,
  `ConditioningSpec.n_parameters`, branch head `pde_params`.
- DeepONet **query** jets: `∂^α G(u)(y)` is closed form (trunk jet ×
  branch coeffs). The branch that eats `μ` is an encoder, not a jet
  trunk.
- `omnibias.{torch,jax}.jet_mv` — `mlp_jet_mv`, `compose_jet_mv`,
  `jet_partials`.
- Spec 09-16 — inner exact Hessian in `theta`, not mixed `x`–`μ` jets.
- Spec 09-20 — homotopy in a problem parameter `tau` on `theta`.
- Spec 09-22 — Newton-on-`x`, not Newton-on-`μ`.

**Confirmed gap.** No API that returns `∂^{α,β} u / ∂x^α ∂μ^β` from
one tower pass with the method labelled closed-form. Concatenating
`μ` into a generic MLP and calling autodiff is ordinary AD.

## 4. Mathematics

Let `u = u(x, μ)` be represented by an OMBU / jet network on the
joint coordinate `z = (x, μ)`. The founding **bias collapse**
(`delta -> 0`) supplies `sigma^(n)` along each affine direction in
`z`. `mlp_jet_mv` of total order `N` is every mixed partial with
`|α| + |β| <= N`.

Closed-form claim holds **iff** every path from `μ` to `u` is a
composition the jet kernels know (affine + registered `sigma` /
`compose_jet_mv`). A dense encoder on `pde_params` that is not a jet
net makes `∂/∂μ` **autodiff**, and the spec must say so.

No temperature collapse. Error: omitted remainder `R_N` in `μ` and
in `x`.

Uses that **call** this jet and live in other files:

- Newton-on-`μ` → 09-22 with `x` replaced by `μ` (do not duplicate).
- Continuation in `μ` → 09-20 (do not duplicate).
- Few-shot across a `μ`-family → 09-16 (do not duplicate).

## 5. Worked example

Heat `u_t = μ u_xx` on a Fourier mode, exact `u = e^{-μ k^2 t} sin(kx)`
with `k = 1`, `t = 1`, `μ = 1`. Then `u_μ = -k^2 t u = -u`.

A jet trunk that takes `(x, t, μ)` and is exact on this mode (or a
fitted OMBU that matches the mode to `1e-10` in value) must return
`∂u/∂μ` within `1e-8` of `-u` at a named interior point, and must
beat a central FD in `μ` (`h = 1e-4`) on `|∂u/∂μ + u|` or match it
within `2x` while reporting the method label `closed_form`.

If the implementation uses autodiff through a non-jet `pde_params`
encoder, G3 fails even if the number is accurate.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/pinn/operator/{torch,jax}/parameter_jets.py  — proposed
@dataclass(frozen=True)
class ParameterJetSpec:
    space_order: int
    param_order: int
    method: str = "closed_form"  # closed_form | autodiff

def mixed_jet(field, coords, parameters, *, spec: ParameterJetSpec):
    """Partials ∂^{α,β} u. Raises if spec.method is closed_form and
    parameters do not enter a jet trunk."""
```

JAX: `mlp_jet_mv` on concatenated `(coords, parameters)`; no Python
loop over mixed multi-indices.

## 7. Practical use cases

1. **Sensitivity** `∂u/∂μ` on parametric heat vs FD-in-`μ`.
2. **Newton-on-`μ`** to match a sensor (driver is 09-22).
3. **UQ readout:** `u`, `∂u/∂μ`, `∂²u/∂μ²` in one pass.
4. **Not** Navier–Stokes parameter identification at continuum.

## 8. Acceptance gates

- **G1 identity.** Worked Fourier heat mode: `|∂u/∂μ + u| < 1e-8`
  at a named point when the trunk is exact on that mode.
- **G2 skill.** `make_parametric_heat_slab` (or Burgers): mixed-jet
  `∂u/∂μ` MAE vs a high-order FD reference is strictly below the
  `h=1e-3` central-FD MAE, five seeds, skill vs `∂u/∂μ = 0` positive.
- **G3 method label.** A test asserts `closed_form` is refused unless
  `μ` enters `mlp_jet_mv` / OMBU; autodiff path cannot set that flag.
- **G4 honesty.** `parampinn_package` false; `ns_claim` false;
  `stretch_claim` false.
- **G5 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/parameter_space_jets.py`
- Smoke: `docs/benchmarks/parameter_space_jets_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/parameter_space_jets/`

## 10. Honesty and scope

- Not a new package named ParamPINN / P²INN. The parametric **slab**
  is already shipped; this spec is mixed jets.
- Closed form is a property of the `μ` path, not of the word
  "parametric."
- Not CCF stretch. Not NS global regularity.
- Certificate tier: empirical. Optional later TM enclosure in `μ`
  is 09-05 / 09-24, not this file.

## 11. Open questions and risks

- **Branch vs trunk.** Putting `μ` only in the DeepONet branch and
  jetting the trunk in `y` does not give closed-form `∂/∂μ`.
- **Width.** Joint `(x, μ)` jets grow as multivariate order `N`.
- **Falsifier.** G2: FD-in-`μ` wins, or G3: a PR labels autodiff
  `closed_form`.

## 12. Implementation checklist

- [ ] Jet path for `parameters` in `omnibias.pinn.operator`
- [ ] Method-label guard test
- [ ] Fourier-mode G1 test
- [ ] `benchmarks/parameter_space_jets.py` plus smoke JSON
- [ ] `__all__` update
- [ ] Index row in `theory/README.md`

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

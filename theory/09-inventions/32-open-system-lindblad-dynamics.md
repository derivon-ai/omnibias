# 09-32 Open-system Lindblad dynamics

## 1. Thesis and status

A time-independent GKSL generator is a linear semigroup: `rho(t) =
exp(t L) rho(0)` and `d^n rho / dt^n = L^n rho(t)` from one propagator
evaluation. This spec ships that semigroup in three registers
(float core, certified interval, torch/jax twins), a qpinn
density-matrix residual with a hard PSD cage, and mints `relaxation`
as the seventh named collapse. The thermal qubit steady population
is the shipped sigmoid occupancy; pure dephasing reproduces
einselection exactly.

- **Status**: shipped (G1–G7 CI; export, not a new package; Markovian
  GKSL declared, not derived)
- **Depends on**: 01-14, 04-03, 09-31
- **Blocks**: none

### Operator card

- **Benefit.** A certified propagator, a certified "this `rho(t)` is a
  valid density matrix" verdict, and a certified relaxation-time bound
  composed from primitives that already ship (`interval_matrix_exp`,
  `lohner_flow`, `interval_ldlt_pivots`, `occupancy`).
- **How it works.** Vectorize `rho` (column stacking), realify the
  complex superoperator once, and reuse the real interval LA stack.
  Arbitrary-order time derivatives are matrix powers of one `exp(t L)`.
  This is the *linear-semigroup* analogue of the sigma tower, labelled
  as such, not as the activation tower.
- **Strength.** Exact on the qubit T1/T2/thermal family and on pure
  dephasing; sound enclosure of `exp(t L)` for general finite `d`.
- **When to use.** Finite-dimensional Markovian master equations,
  certified decoherence-plus-relaxation timescales, qpinn training of
  a density matrix under a declared GKSL generator.
- **When not.** As a Born–Markov–secular derivation from a
  system-bath Hamiltonian, a non-Markovian claim, a measurement-problem
  resolution, or a continuum / thermodynamic-limit statement.
- **Accuracy floor.** Only the qubit Bloch family and pure dephasing
  are closed form. General `d` is a certified *numerical* propagator.

## 2. Where it lands

- `omnibias.core.lindblad` — float semigroup (pure Python + numpy).
- `omnibias.core.verified.lindblad` — certified propagator / flow /
  positivity / steady state / contraction.
- `omnibias.core.collapse.relaxation` — seventh named collapse.
- `omnibias.{torch,jax}.lindblad` — bit-identical differentiable twins.
- `omnibias.qpinn.{torch,jax}.equations.lindblad` plus
  `omnibias.qpinn.{torch,jax}.cage.density` — density-matrix residual
  and hard `G G^dag / Tr` cage.

No new package: the domain is the already-shipped quantum / verified /
collapse surface. qpinn is the existing residual home; T1 twins follow
the occupancy (04-03) shape.

## 3. Prior art in omnibias

- `omnibias.core.verified.lohner.interval_matrix_exp` — certified
  truncated Taylor `exp(M)` with a Lagrange tail; requires
  `||M||_inf / (order+2) < 1`.
- `omnibias.core.verified.lohner.lohner_flow` /
  `linear_field` / `constant_jacobian` / `naive_interval_flow` — QR
  frame versus wrapping-prone baseline.
- `omnibias.core.verified.linalg.interval_solve` /
  `inf_norm_matrix` / `matmul`.
- `omnibias.core.verified.eig_operator.interval_ldlt_pivots` /
  `is_positive_definite` — real symmetric only; unlocked for Hermitian
  `rho` by the realification `[[A, -B], [B, A]]`.
- `omnibias.core.collapse.einselection` — exact pure-dephasing
  solution, `T1 = inf`; this spec's general engine must reproduce it.
- `omnibias.core.occupancy.occupancy` / `FermiModel` — the thermal
  qubit excited population is `sigmoid(-beta * omega)`.
- `omnibias.qpinn` — unitary TDSE/NLS/Dirac residuals, hard norm cage,
  split-real encoding; **no** open-system residual (confirmed gap).
- `omnibias.core.collapse.schema` — a name earns a slot only by a
  distinct `(parameter, surviving_object)` pair.

**Confirmed gap.** No GKSL / Liouvillian / amplitude-damping /
thermal-density-matrix vocabulary exists beyond einselection's
pure-dephasing analytic solution.

## 4. Mathematics

The Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) equation on a
finite-dimensional Hilbert space is

```
d(rho)/dt = L[rho] = -i [H, rho]
    + sum_k gamma_k (A_k rho A_k^dag - 1/2 {A_k^dag A_k, rho})
```

with `gamma_k >= 0`. For time-independent `(H, A_k, gamma_k)`:

```
rho(t) = exp(t L) rho(0),
d^n rho / dt^n = L^n rho(t).
```

Column-stacking `vec(A X B) = (B^T kron A) vec(X)` yields a complex
`d^2 x d^2` superoperator. `ComplexInterval` is scalar-only, so the
certified path realifies once:

```
[[Re M, -Im M], [Im M, Re M]]  acting on  [Re v; Im v].
```

A Hermitian `rho = A + i B` is PSD iff the real symmetric
`[[A, -B], [B, A]]` is PSD (eigenvalues doubled). That is the
shipped `interval_ldlt_pivots` test.

**Qubit Bloch family (closed form).** Ground `|0>`, excited `|1>`,
`H = diag(0, omega)`, `A_down = |0><1|` at `gamma_down`,
`A_up = |1><0|` at `gamma_up`, optional `A_phi = sigma_z` at
`gamma_phi`:

```
T1^{-1} = gamma_down + gamma_up,
T2^{-1} = 1/(2 T1) + 2 gamma_phi,
p_e(t) = p_e_ss + (p_e(0) - p_e_ss) exp(-t / T1),
rho_01(t) = rho_01(0) exp(i omega t) exp(-t / T2),
p_e_ss = gamma_up / (gamma_up + gamma_down).
```

Detailed balance `gamma_up / gamma_down = exp(-beta * omega)` makes
`p_e_ss = occupancy(FermiModel(beta, mu=0), omega)`.

**Pure dephasing.** `H = 0`, `A = sigma_z`, `gamma = Gamma / 2`
reproduces einselection: off-diagonals decay as `exp(-Gamma t)`,
diagonals freeze. The kernel of `L` is then a *manifold* of diagonal
states, not a unique fixed point.

**Relaxation collapse.** Moving parameter `relaxation_rate`
(`gamma t -> inf`); surviving object a unique **steady state**.
On a sound enclosure `C(t)` of the contraction factor
`||exp(t L) - Pi||_inf` toward that unique `rho_ss`, with an explicit
caller `distance_budget` `eps > 0`:

- `C.hi < eps` -> `PROVED`; surviving object is the enclosed `rho_ss`
- `C.lo > eps` -> `DISPROVED`
- otherwise, or if uniqueness cannot be certified -> `BLOCKED`

This is **not** founding bias collapse (`delta -> 0`). The
`beta -> inf` zero-temperature limit of the thermal population is
the founding **temperature collapse** (feasibility sense) already
minted by occupancy / the founding spec; this module names it and
does not re-request that slot. The new slot is `relaxation`,
distinct from einselection on both axes (`relaxation_rate` /
`steady_state` versus `decoherence_rate` /
`einselected_distribution`). The disagreement gate: on pure
dephasing, einselection can `PROVE` while relaxation must refuse
(no unique steady state).

`propagator_enclosure` subdivides `[0, t]` so each step satisfies
`interval_matrix_exp`'s `||M||_inf / (order+2) < 1`, then composes
with `matmul`. No admissible step is a first-class refusal, not a
wrong bound. `lohner_flow` is used only when the generator is an
exact float matrix (`linear_field` takes floats); interval-valued
rates take the `interval_matrix_exp` composition path.

## 5. Worked example

Qubit amplitude damping, `omega = 1`, `gamma_down = 1`,
`gamma_up = 0`, `rho(0) = |1><1|`. Then `T1 = 1`, `p_e(t) = exp(-t)`,
`p_e_ss = 0`. At `t = 5`, `p_e = exp(-5) ≈ 6.74e-3`. The certified
propagator must contain that population. The thermal detailed-balance
sibling with `beta = 2`, `gamma_down = 1` has
`p_e_ss = occupancy(FermiModel(2.0, 0.0), 1.0) ≈ 0.119202922`.

Pure dephasing `Gamma = 1`, equal superposition, `eps = 1e-6`:
einselection at `t = 20` is `PROVED`; relaxation collapse on the
same model refuses (steady-state manifold).

## 6. Proposed API

```python
# omnibias.core.lindblad — implemented
@dataclass(frozen=True)
class LindbladModel:
    hamiltonian: tuple[tuple[complex, ...], ...]
    jumps: tuple[tuple[tuple[complex, ...], ...], ...]
    rates: tuple[float, ...]

def liouvillian(model) -> NDArray: ...
def propagator(model, time: float) -> NDArray: ...
def density_matrix(model, rho0, time: float) -> NDArray: ...
def apply_lindblad(model, rho) -> NDArray: ...
def time_derivative_tower(model, rho0, time, *, order: int) -> tuple[NDArray, ...]: ...
def steady_state(model) -> NDArray: ...
def qubit_bloch_solution(...) -> NDArray: ...
def thermal_steady_population(*, omega: float, beta: float) -> float: ...
def dissipative_gap(model) -> float: ...  # proposer, never a verdict
def honesty_payload() -> dict[str, bool]: ...

# omnibias.core.verified.lindblad — implemented
def liouvillian_enclosure(model) -> IntervalMatrix: ...
def propagator_enclosure(model, time, *, order: int = 12) -> IntervalMatrix | None: ...
def density_matrix_enclosure(model, rho0, time) -> ComplexInterval matrix | None: ...
def trajectory_enclosure(model, rho0, *, h, n_steps, order=12) -> ...
def trace_enclosure(rho) -> Interval: ...
def hermiticity_residual_enclosure(rho) -> Interval: ...
def positivity_verdict(rho) -> ObligationVerdict: ...
def steady_state_enclosure(model) -> ComplexInterval matrix | None: ...
def contraction_enclosure(model, time) -> Interval | None: ...
def certified_relaxation_time(model, *, distance_budget, t_max) -> RelaxationTime: ...

# omnibias.core.collapse.relaxation — implemented
RELAXATION_SPEC = CollapseSpec(
    name="relaxation",
    parameter="relaxation_rate",
    limit="inf",
    surviving_object="steady_state",
    failure="contraction_not_certified",
    home="omnibias.core.collapse.relaxation",
    register="measure",
)
def relaxation_collapse(model, *, time, distance_budget) -> ObligationVerdict: ...

# omnibias.{torch,jax}.lindblad — implemented, bit-identical
# order is a static Python int (jit / vmap / grad safe)
```

Default dtype is the framework default, never a hardcoded `float32`.
No torch/jax imports from `omnibias.core`.

## 7. Practical use cases

1. **Certified T1.** Amplitude-damping qubit: enclose `p_e(t)` and
   decide when it is below a declared budget.
2. **Thermal occupation.** Detailed-balance rates; the certified
   steady population must enclose `occupancy(FermiModel(beta, 0), omega)`.
3. **Einselection as a special case.** Pure dephasing must match
   `reduced_density_matrix` to 1e-14 in float and sit inside the
   certified propagator.
4. **qpinn open-system residual.** Train `rho(t)` against
   `d rho/dt - L[rho]` with a hard PSD/trace cage.
5. **Not** a system-bath derivation, a non-Markovian solver, a
   single-outcome claim, or a continuum field theory.

## 8. Acceptance gates

- **G1 exactness.** Pure dephasing reproduces
  `einselection.reduced_density_matrix` to 1e-14; qubit amplitude
  damping matches the Bloch closed form; `d^n rho/dt^n = L^n rho(t)`
  matches mpmath numerical differentiation for `n <= 6`.
- **G2 soundness.** Every enclosure contains a dense deterministic
  grid **and** a random sample of float truth.
- **G3 state validity.** `trace_enclosure` contains 1,
  `hermiticity_residual_enclosure` contains 0, `positivity_verdict`
  is `PROVED` for a full-rank mixed state and `BLOCKED` for a pure
  state (a zero eigenvalue makes `PROVED` unreachable).
- **G4 certified relaxation time.** `T_certified(eps) >= T_float`
  against a float spectral-abscissa oracle; a non-contracting model
  returns a halt, not a number.
- **G5 the 04-03 bridge.** Certified thermal-qubit `p_e_ss` encloses
  `occupancy(FermiModel(beta, mu=0), omega)`; float twin matches to
  1e-15; detailed balance holds.
- **G6 registry plus honesty.** `are_distinct(RELAXATION_SPEC,
  EINSELECTION_SPEC)` on both axes; disagreement gate on pure
  dephasing; every honesty key observed `False`; static source scan.
- **G7 wrapping.** `lohner_flow` beats `naive_interval_flow` on
  enclosure width at a fixed horizon by a measured factor.

Parity: T1 twins at `rtol=1e-13` with jit/vmap/grad smokes; qpinn
cross-backend at `rtol=1e-9`, `atol=1e-12`.

## 9. Benchmark plan

CPU smoke: `benchmarks/lindblad.py` writes
`docs/benchmarks/lindblad_smoke.json` (CI `cross_backend` job).
`--full` writes `$OMNIBIAS_SCRATCH/lindblad/lindblad_full.json`.
G1–G7 live in the smoke; cost / wall-time is recorded, not gated.

## 10. Honesty and scope

- Not a wave-function-collapse claim, not a measurement-problem
  resolution, not a Born-rule derivation, not a single-outcome claim.
- The GKSL form is a **caller input**. There is no Born–Markov–secular
  derivation from a system-bath Hamiltonian here
  (`markovian_model_declared_not_derived: True`).
- Not a non-Markovian claim. Not a general closed-form claim (only
  the qubit and pure-dephasing families are closed form).
- Do not conflate this with founding bias collapse (`delta -> 0`),
  temperature collapse (`beta -> inf`, the feasibility sense — named
  here as the T=0 thermal step, already minted, not re-requested), or
  Enclosure Collapse (`width -> 0` of a sound enclosure).
- Pure dephasing is einselection's job; this module reproduces it
  and refuses a unique-steady-state certificate on that manifold.
- `theorem_prover_verified` and `mathlib_verified` stay false.
- Navier-Stokes, Yang-Mills mass gap, RH, and P vs NP stay external.
- No thermodynamic limit, no continuum limit, no quantum-advantage
  claim.

## 11. Open questions and risks

- **Inf-norm contraction can be loose.** `||exp(t L) - Pi||_inf` may
  stay above a small `eps` because of wrapping even when the 2-norm
  spectral abscissa has already dropped. G4 is therefore
  "conservative versus a float oracle", not "tight T1". A falsifier
  is `T_certified < T_float`.
- **`interval_matrix_exp` step restriction.** Large `||t L||` needs
  subdivision; if no admissible step exists the API refuses rather
  than returning a wrong bound.
- **Pure-state positivity.** LDL^T cannot `PROVE` a rank-1 `rho`.
  `Inconclusive` is first-class, not a bug.
- **Rebrand challenge.** A reviewer may call relaxation Enclosure
  Collapse of a contraction scalar. The surviving object is a
  density matrix (the unique `rho_ss`), and the disagreement gate
  versus einselection is the defense.
- **Budget silently defaulting.** `distance_budget` is always an
  explicit caller argument.
- **Falsifier.** A `PROVED` positivity verdict on a state whose
  float eigenvalues include a negative, a thermal `p_e_ss` enclosure
  that misses occupancy, or any honesty key observed `True`.

## 12. Implementation checklist

- [x] `theory/09-inventions/32-open-system-lindblad-dynamics.md`
- [x] `omnibias.core.lindblad`
- [x] `omnibias.core.verified.lindblad`
- [x] `omnibias.core.collapse.relaxation` + engine `lindblad` kind
- [x] `omnibias.{torch,jax}.lindblad`
- [x] qpinn `_core/density.py`, equations, cage
- [x] tests (G1–G7, parity, qpinn cross-backend)
- [x] `benchmarks/lindblad.py` + smoke JSON + CI step
- [x] docs API + cookbook + mkdocs nav + collapse seventh row
- [x] regenerate `__all__` on every touched `__init__.py`
- [x] index row in `theory/README.md`

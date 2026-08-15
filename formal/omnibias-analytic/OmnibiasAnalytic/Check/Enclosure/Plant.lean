/-
Locked enclosure-trace plants (A/B/C/D).

Each plant is a finite rational DAG. `evalTrace` replays it. The NK plant
then applies the existing unique-root theorem on `[5/4, 7/4]`.

Scope (honest). These are planted rational identities and one compact-interval
root. They are not a continuum PDE claim, not analytic continuation of a
Dirichlet series, and not a general SOS engine.
-/

import OmnibiasAnalytic.Check.Enclosure
import OmnibiasAnalytic.Check.Kantorovich.Plant
import OmnibiasAnalytic.Tower

namespace OmnibiasAnalytic.Check

open QInterval

/-! ### A. Tower Horner of `sigmoidPoly 2` at `2/3` -/

/-- Horner of `[0, 1, -3, 2]` at `2/3`. -/
def towerHornerOps : List TraceOp :=
  [ .const (2 / 3)
  , .const 2
  , .const (-3)
  , .const 1
  , .const 0
  , .mul 1 0
  , .add 5 2
  , .mul 6 0
  , .add 7 3
  , .mul 8 0
  , .add 9 4
  ]

theorem tower_horner_coeffs : OmnibiasAnalytic.Tower.sigmoidCoeffList 2 = [0, 1, -3, 2] :=
  OmnibiasAnalytic.Tower.sigmoidCoeffs_two

theorem tower_horner_result :
    (evalTrace towerHornerOps).getLast? = some (point (-2 / 27)) := by
  native_decide

/-! ### B. NK bound DAG plus unique root of `x² - 2` -/

/-- Replay `|A(c²-2)|`, `κ = 2 Z2 r`, `p(r) = Y0 + Z2 r² - r`. -/
def nkBoundOps : List TraceOp :=
  [ .const (3 / 2)
  , .const (1 / 3)
  , .const 2
  , .const (1 / 4)
  , .const (2 / 3)
  , .mul 0 0
  , .sub 5 2
  , .mul 1 6
  , .abs 7
  , .mul 4 3
  , .mul 2 9
  , .mul 3 3
  , .mul 4 11
  , .add 8 12
  , .sub 13 3
  ]

theorem nk_trace_bounds :
    (evalTrace nkBoundOps)[8]? = some (point (1 / 12)) ∧
    (evalTrace nkBoundOps)[10]? = some (point (1 / 3)) ∧
    (evalTrace nkBoundOps)[14]? = some (point (-1 / 8)) := by
  native_decide

theorem nk_trace_unique_zero :
    ((evalTrace nkBoundOps)[8]? = some (point (1 / 12)) ∧
      (evalTrace nkBoundOps)[10]? = some (point (1 / 3)) ∧
      (evalTrace nkBoundOps)[14]? = some (point (-1 / 8))) ∧
    ∃! x : ℝ, x ∈ Set.Icc (5 / 4) (7 / 4) ∧ quadraticPlant x = 0 :=
  ⟨nk_trace_bounds, quadratic_plant_radii_unique_zero⟩

/-! ### C. Bernoulli `B₂ = 1/6` and named `zetaNeg1 = -1/12` -/

/-- `B₂ = 1 · 2 / (2² (2² - 1))`, then `zetaNeg1 := -B₂ / 2`. -/
def bernoulliOps : List TraceOp :=
  [ .const 1
  , .const 2
  , .const 4
  , .const 3
  , .mul 2 3
  , .recip 4
  , .mul 0 1
  , .mul 6 5
  , .const (-2)
  , .recip 8
  , .mul 7 9
  ]

theorem bernoulli_b2_zetaNeg1 :
    (evalTrace bernoulliOps)[7]? = some (point (1 / 6)) ∧
    (evalTrace bernoulliOps)[10]? = some (point (-1 / 12)) := by
  native_decide

/-! ### D. Exact LDLᵀ of `[[2, 1], [1, 2]]` -/

/-- Pivots `d₀ = 2`, `L₁₀ = 1/2`, `d₁ = 2 - L₁₀² d₀ = 3/2`. -/
def ldltOps : List TraceOp :=
  [ .const 2
  , .const 1
  , .recip 0
  , .mul 1 2
  , .mul 3 3
  , .mul 4 0
  , .sub 0 5
  ]

theorem ldlt_plant_pivots_pos :
    (evalTrace ldltOps)[0]? = some (point 2) ∧
    (evalTrace ldltOps)[6]? = some (point (3 / 2)) ∧
    (0 : ℚ) < 2 ∧ (0 : ℚ) < 3 / 2 := by
  native_decide

end OmnibiasAnalytic.Check

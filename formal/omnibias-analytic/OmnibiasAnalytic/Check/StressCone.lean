/-
Finite rational admissible-stress cone (Mathlib-backed).

``T`` lies in the interior of ``cone(v₁, v₂)`` when both Cramer
weights are strictly positive. This file proves that algebra over `ℚ`
for the locked generators ``e₁, e₂`` and ``T = (1, 1)``.

Not Clay (A)/(B), not a forced-blowup reproof, no `admit`.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

def coneDet (v1x v1y v2x v2y : ℚ) : ℚ := v1x * v2y - v1y * v2x

def coneLambda1 (Tx Ty v2x v2y det : ℚ) : ℚ :=
  (Tx * v2y - Ty * v2x) / det

def coneLambda2 (v1x v1y Tx Ty det : ℚ) : ℚ :=
  (v1x * Ty - v1y * Tx) / det

theorem locked_cone_interior :
    coneDet 1 0 0 1 = 1 ∧
      0 < coneLambda1 1 1 0 1 1 ∧
      0 < coneLambda2 1 0 1 1 1 := by
  unfold coneDet coneLambda1 coneLambda2
  constructor
  · norm_num
  constructor <;> norm_num

end OmnibiasAnalytic.Check

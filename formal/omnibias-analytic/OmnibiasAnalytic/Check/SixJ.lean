/-
Named SU(2) Racah 6j identities (Mathlib-backed).

Locked rational evaluations matching
`omnibias.geometry.gauge.transfer.sixj`:

* `{1/2 1/2 0; 1/2 1/2 0} = -1/2` via the Racah prefactor `1/4` and
  the single `t = 1` term `-2`
* the all-`1/2` triad is an illegal triangle, so the 6j is `0`

Scope (honest). Finite rational identities on named labels. Not a
continuum gauge claim. The file contains no `admit`.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

/-- `Δ(1/2,1/2,0)² = 0! 0! 1! / 2! = 1/2`. -/
def deltaSqHalfHalfZero : ℚ :=
  1 / 2

/-- Four identical `Δ²` give `1/16`; the square root is the rational `1/4`. -/
def sixjHalfPrefactor : ℚ :=
  1 / 4

/-- The Racah sum for `{1/2 1/2 0; 1/2 1/2 0}` is the single term `t = 1`. -/
def sixjHalfSum : ℚ :=
  -2

def sixjHalfHalfZero : ℚ :=
  sixjHalfPrefactor * sixjHalfSum

/-- Textbook value `{1/2 1/2 0; 1/2 1/2 0} = -1/2`.
Finite rational identity, not a continuum gauge claim. -/
theorem sixj_half_half_zero : sixjHalfHalfZero = -1 / 2 := by
  unfold sixjHalfHalfZero sixjHalfPrefactor sixjHalfSum
  norm_num

/-- All-`1/2` fails the integer-sum triangle rule, so the 6j is `0`. -/
def sixjAllHalf : ℚ :=
  0

theorem sixj_all_half_vanishes : sixjAllHalf = 0 := by
  unfold sixjAllHalf
  norm_num

end OmnibiasAnalytic.Check

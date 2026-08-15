/-
Newton-Kantorovich / Krawczyk lemmas (Mathlib-backed).

Two layers live here:

* Finite rational inequalities (`radiiPoly < 0`, `κ < 1`, Krawczyk
  containment). These are the leaf obligations of the existing
  `radii_polynomial` / `krawczyk` certificate classes. Existence remains a
  trusted input for those classes.
* 1-D existence on a compact interval: a contraction self-map of
  `Icc (c - r) (c + r)` has a unique fixed point (Banach), and the Newton
  operator `x ↦ x - A f(x)` converts that into a unique root of `f`.

Scope (honest). The existence theorems are 1-D statements about a named map
on a compact real interval. They do not state a continuum PDE.
-/

import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Dynamics.FixedPoints.Basic
import Mathlib.Tactic
import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Topology.MetricSpace.Lipschitz

namespace OmnibiasAnalytic.Check

open scoped NNReal
open Set Function

/-! ### Finite rational inequalities (existing certificate leaves) -/

/-- The **radii polynomial** `p(r) = Y0 + (Z0 + Z1) r + Z2 r² - r`. -/
def radiiPoly (Y0 Z0 Z1 Z2 r : ℚ) : ℚ := Y0 + (Z0 + Z1) * r + Z2 * r ^ 2 - r

/-- The **contraction constant** `κ(r) = Z0 + Z1 + 2 Z2 r`. -/
def contractionConst (Z0 Z1 Z2 r : ℚ) : ℚ := Z0 + Z1 + 2 * Z2 * r

/-- **Self-map bound.** The radii polynomial being negative at `r` is exactly the
Newton-Kantorovich self-map inequality `Y0 + (Z0 + Z1) r + Z2 r² < r`. -/
theorem radii_selfmap {Y0 Z0 Z1 Z2 r : ℚ} (hp : radiiPoly Y0 Z0 Z1 Z2 r < 0) :
    Y0 + (Z0 + Z1) * r + Z2 * r ^ 2 < r := by
  unfold radiiPoly at hp; linarith

/-- **Radii-polynomial contraction obligation.** Both finite conditions of the
existence test: the polynomial is negative and the contraction constant is `< 1`. -/
theorem radii_contraction {Y0 Z0 Z1 Z2 r : ℚ}
    (hp : radiiPoly Y0 Z0 Z1 Z2 r < 0) (hk : contractionConst Z0 Z1 Z2 r < 1) :
    radiiPoly Y0 Z0 Z1 Z2 r < 0 ∧ contractionConst Z0 Z1 Z2 r < 1 :=
  ⟨hp, hk⟩

/-- **Krawczyk strict containment (one coordinate).** The image interval
`[lo, hi]` lies strictly inside the box `[c - r, c + r]`. -/
theorem krawczyk_strictly_inside {c r lo hi : ℚ} (h1 : c - r < lo) (h2 : hi < c + r) :
    c - r < lo ∧ hi < c + r :=
  ⟨h1, h2⟩

/-! ### 1-D Newton operator and Banach on a compact interval -/

/-- Newton-like operator `T(x) = x - A f(x)`. -/
def newtonOp (A : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := x - A * f x

theorem newtonOp_fixedPt_iff {A : ℝ} {f : ℝ → ℝ} {x : ℝ} (hA : A ≠ 0) :
    IsFixedPt (newtonOp A f) x ↔ f x = 0 := by
  unfold IsFixedPt newtonOp
  constructor
  · intro h
    have : A * f x = 0 := by linarith
    exact (mul_eq_zero.mp this).resolve_left hA
  · intro hf
    simp [hf]

theorem mem_closedBall_Icc {c r x : ℝ} :
    x ∈ Icc (c - r) (c + r) ↔ |x - c| ≤ r := by
  simp [abs_le, le_sub_iff_add_le, add_comm]

theorem center_mem_Icc {c r : ℝ} (hr : 0 ≤ r) : c ∈ Icc (c - r) (c + r) := by
  simpa [mem_closedBall_Icc] using hr

/-- Crude self-map: `|T x - c| ≤ K |x - c| + |T c - c|`, so `Y0 + K r ≤ r`
puts the image back in the interval. This is weaker than the classical
quadratic radii bound (an extra `Z2 r²`); it is the estimate that follows
from a Lipschitz constant alone. -/
theorem mapsTo_Icc_of_center_lip
    {T : ℝ → ℝ} {c r Y0 : ℝ} {K : ℝ≥0}
    (hr : 0 ≤ r)
    (hY : |T c - c| ≤ Y0)
    (hLip : LipschitzOnWith K T (Icc (c - r) (c + r)))
    (hbound : Y0 + (K : ℝ) * r ≤ r) :
    MapsTo T (Icc (c - r) (c + r)) (Icc (c - r) (c + r)) := by
  intro x hx
  have hc := center_mem_Icc (c := c) hr
  have hdist : dist (T x) (T c) ≤ (K : ℝ) * dist x c := hLip.dist_le_mul x hx c hc
  have : |T x - c| ≤ (K : ℝ) * |x - c| + |T c - c| := by
    calc
      |T x - c| ≤ |T x - T c| + |T c - c| := abs_sub_le _ _ _
      _ = dist (T x) (T c) + |T c - c| := by simp [Real.dist_eq]
      _ ≤ (K : ℝ) * dist x c + |T c - c| := by gcongr
      _ = (K : ℝ) * |x - c| + |T c - c| := by simp [Real.dist_eq]
  have hxr : |x - c| ≤ r := (mem_closedBall_Icc).1 hx
  have : |T x - c| ≤ r := by nlinarith
  exact (mem_closedBall_Icc).2 this

/-- A contraction self-map of a compact real interval has a unique fixed point
there (Banach, Mathlib `ContractingWith.exists_fixedPoint'`). -/
theorem unique_fixedPoint_on_Icc
    {T : ℝ → ℝ} {c r : ℝ} {K : ℝ≥0}
    (hr : 0 ≤ r)
    (hmaps : MapsTo T (Icc (c - r) (c + r)) (Icc (c - r) (c + r)))
    (hK : K < 1)
    (hLip : LipschitzOnWith K T (Icc (c - r) (c + r))) :
    ∃! x, x ∈ Icc (c - r) (c + r) ∧ T x = x := by
  set s := Icc (c - r) (c + r)
  have hsc : IsComplete s := isClosed_Icc.isComplete
  have hcontr : ContractingWith K (hmaps.restrict T s s) :=
    ⟨hK, fun x y => hLip x.property y.property⟩
  have hc : c ∈ s := center_mem_Icc hr
  obtain ⟨y, hys, hyfix, _⟩ :=
    ContractingWith.exists_fixedPoint' (f := T) hsc hmaps hcontr hc (edist_ne_top _ _)
  refine existsUnique_of_exists_of_unique ⟨y, hys, hyfix⟩ ?_
  intro a b ha hb
  have hdist : dist a b ≤ (K : ℝ) * dist a b := by
    simpa [ha.2, hb.2] using hLip.dist_le_mul a ha.1 b hb.1
  have hK' : (K : ℝ) < 1 := hK
  have : dist a b = 0 := by nlinarith [dist_nonneg (x := a) (y := b)]
  exact eq_of_dist_eq_zero this

/-- Center displacement plus a Lipschitz constant `< 1` with `Y0 + K r ≤ r`
yield a unique fixed point on the interval. -/
theorem unique_fixedPoint_of_center_lip
    {T : ℝ → ℝ} {c r Y0 : ℝ} {K : ℝ≥0}
    (hr : 0 ≤ r)
    (hY : |T c - c| ≤ Y0)
    (hLip : LipschitzOnWith K T (Icc (c - r) (c + r)))
    (hK : K < 1)
    (hbound : Y0 + (K : ℝ) * r ≤ r) :
    ∃! x, x ∈ Icc (c - r) (c + r) ∧ T x = x :=
  unique_fixedPoint_on_Icc hr
    (mapsTo_Icc_of_center_lip hr hY hLip hbound) hK hLip

/-- Unique root of `f` from a contracting Newton operator. -/
theorem unique_zero_of_newton_contraction
    {f : ℝ → ℝ} {A c r : ℝ} {K : ℝ≥0}
    (hA : A ≠ 0) (hr : 0 ≤ r)
    (hmaps : MapsTo (newtonOp A f) (Icc (c - r) (c + r)) (Icc (c - r) (c + r)))
    (hK : K < 1)
    (hLip : LipschitzOnWith K (newtonOp A f) (Icc (c - r) (c + r))) :
    ∃! x, x ∈ Icc (c - r) (c + r) ∧ f x = 0 := by
  obtain ⟨x, ⟨hx, hT⟩, huniq⟩ := unique_fixedPoint_on_Icc hr hmaps hK hLip
  refine ⟨x, ⟨hx, (newtonOp_fixedPt_iff hA).1 hT⟩, ?_⟩
  intro y hy
  exact huniq y ⟨hy.1, (newtonOp_fixedPt_iff hA).2 hy.2⟩

theorem unique_zero_of_center_lip
    {f : ℝ → ℝ} {A c r Y0 : ℝ} {K : ℝ≥0}
    (hA : A ≠ 0) (hr : 0 ≤ r)
    (hY : |newtonOp A f c - c| ≤ Y0)
    (hLip : LipschitzOnWith K (newtonOp A f) (Icc (c - r) (c + r)))
    (hK : K < 1)
    (hbound : Y0 + (K : ℝ) * r ≤ r) :
    ∃! x, x ∈ Icc (c - r) (c + r) ∧ f x = 0 :=
  unique_zero_of_newton_contraction hA hr
    (mapsTo_Icc_of_center_lip hr hY hLip hbound) hK hLip

theorem hasDerivAt_newtonOp {A : ℝ} {f : ℝ → ℝ} {f' x : ℝ}
    (hf : HasDerivAt f f' x) :
    HasDerivAt (newtonOp A f) (1 - A * f') x := by
  change HasDerivAt (fun y => y - A * f y) (1 - A * f') x
  exact (hasDerivAt_id' x).sub (hf.const_mul A)

theorem deriv_newtonOp {A : ℝ} {f : ℝ → ℝ} {x : ℝ}
    (hf : DifferentiableAt ℝ f x) :
    deriv (newtonOp A f) x = 1 - A * deriv f x :=
  (hasDerivAt_newtonOp hf.hasDerivAt).deriv

/-- 1-D Krawczyk / Newton derivative bound: `|1 - A f'| ≤ K` on the interval
implies the Newton operator is `K`-Lipschitz there. -/
theorem lipschitzOnWith_newtonOp_of_deriv_le
    {f : ℝ → ℝ} {A c r : ℝ} {K : ℝ≥0}
    (hf : Differentiable ℝ f)
    (hbound : ∀ x ∈ Icc (c - r) (c + r), |1 - A * deriv f x| ≤ (K : ℝ)) :
    LipschitzOnWith K (newtonOp A f) (Icc (c - r) (c + r)) := by
  have hdiff : ∀ x ∈ Icc (c - r) (c + r), DifferentiableAt ℝ (newtonOp A f) x :=
    fun x _ => (hasDerivAt_newtonOp (hf x).hasDerivAt).differentiableAt
  refine (convex_Icc (c - r) (c + r)).lipschitzOnWith_of_nnnorm_deriv_le hdiff ?_
  intro x hx
  have hder := deriv_newtonOp (A := A) (hf x)
  rw [← NNReal.coe_le_coe]
  simpa [hder, coe_nnnorm, Real.norm_eq_abs] using hbound x hx

/-- 1-D Krawczyk existence: derivative bound `|1 - A f'| ≤ K < 1` plus the
crude self-map estimate yield a unique root on the interval. -/
theorem unique_zero_of_krawczyk_1d
    {f : ℝ → ℝ} {A c r Y0 : ℝ} {K : ℝ≥0}
    (hA : A ≠ 0) (hr : 0 ≤ r)
    (hf : Differentiable ℝ f)
    (hY : |newtonOp A f c - c| ≤ Y0)
    (hder : ∀ x ∈ Icc (c - r) (c + r), |1 - A * deriv f x| ≤ (K : ℝ))
    (hK : K < 1)
    (hbound : Y0 + (K : ℝ) * r ≤ r) :
    ∃! x, x ∈ Icc (c - r) (c + r) ∧ f x = 0 :=
  unique_zero_of_center_lip hA hr hY
    (lipschitzOnWith_newtonOp_of_deriv_le hf hder) hK hbound

end OmnibiasAnalytic.Check

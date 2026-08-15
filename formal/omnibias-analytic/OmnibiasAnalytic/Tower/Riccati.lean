/-
First derivatives of the omnibias Riccati family.

Pointwise identities on `ℝ`: `sigmoid' = σ(1-σ)`, `tanh' = 1-t²`,
`sech' = -t sech`, `gaussian' = -z g`, `softplus' = sigmoid`. Each map is
`C^∞`, so the iterated-derivative link theorems are well-defined at every
finite order.

Scope. These are algebraic / `C^∞` identities. This module does not state a
finite-difference collapse, a continuum PDE claim, or any asymptotic.
-/

import Mathlib.Analysis.Calculus.ContDiff.Basic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Tactic

namespace OmnibiasAnalytic.Tower

noncomputable section

/-- Probabilist's Gaussian `g(x) = exp(-x² / 2)`. -/
def gaussian (x : ℝ) : ℝ := Real.exp (-x ^ 2 / 2)

/-- Softplus `log(1 + exp x)`, whose derivative is `sigmoid`. -/
def softplus (x : ℝ) : ℝ := Real.log (1 + Real.exp x)

/-- Hyperbolic secant `sech = 1 / cosh` (Mathlib 4.31 has no `Real.sech`). -/
def sech (x : ℝ) : ℝ := (Real.cosh x)⁻¹

theorem contDiff_sigmoid : ContDiff ℝ ⊤ Real.sigmoid :=
  ContDiff.of_le _root_.contDiff_sigmoid le_top

theorem hasDerivAt_sigmoid (x : ℝ) :
    HasDerivAt Real.sigmoid (Real.sigmoid x * (1 - Real.sigmoid x)) x :=
  Real.hasDerivAt_sigmoid x

theorem deriv_sigmoid (x : ℝ) :
    deriv Real.sigmoid x = Real.sigmoid x * (1 - Real.sigmoid x) :=
  (hasDerivAt_sigmoid x).deriv

lemma one_add_exp_pos (x : ℝ) : 0 < 1 + Real.exp x := by positivity

lemma one_add_exp_ne_zero (x : ℝ) : 1 + Real.exp x ≠ 0 := (one_add_exp_pos x).ne'

lemma cosh_ne_zero (x : ℝ) : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'

theorem hasDerivAt_tanh (x : ℝ) :
    HasDerivAt Real.tanh (1 - Real.tanh x ^ 2) x := by
  have hx : Real.cosh x ≠ 0 := cosh_ne_zero x
  have hdiv : HasDerivAt (fun y => Real.sinh y / Real.cosh y)
      ((Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) / Real.cosh x ^ 2) x :=
    (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hx
  have hform :
      (Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) / Real.cosh x ^ 2 =
        1 - Real.tanh x ^ 2 := by
    rw [Real.tanh_eq_sinh_div_cosh]
    field_simp [hx]
  convert hdiv using 1
  · funext y; exact Real.tanh_eq_sinh_div_cosh y
  · exact hform.symm

theorem deriv_tanh (x : ℝ) : deriv Real.tanh x = 1 - Real.tanh x ^ 2 :=
  (hasDerivAt_tanh x).deriv

theorem contDiff_tanh : ContDiff ℝ ⊤ Real.tanh := by
  have htanh : Real.tanh = fun y => Real.sinh y / Real.cosh y := by
    funext y; exact Real.tanh_eq_sinh_div_cosh y
  rw [htanh]
  exact Real.contDiff_sinh.div Real.contDiff_cosh cosh_ne_zero

theorem hasDerivAt_sech (x : ℝ) :
    HasDerivAt sech (-Real.tanh x * sech x) x := by
  have hx : Real.cosh x ≠ 0 := cosh_ne_zero x
  have hinv : HasDerivAt (fun y => (Real.cosh y)⁻¹)
      (-Real.sinh x / Real.cosh x ^ 2) x :=
    (Real.hasDerivAt_cosh x).inv hx
  have hform : -Real.sinh x / Real.cosh x ^ 2 = -Real.tanh x * sech x := by
    rw [Real.tanh_eq_sinh_div_cosh]
    unfold sech
    field_simp [hx]
  convert hinv using 1
  · rfl
  · exact hform.symm

theorem deriv_sech (x : ℝ) : deriv sech x = -Real.tanh x * sech x :=
  (hasDerivAt_sech x).deriv

theorem contDiff_sech : ContDiff ℝ ⊤ sech :=
  Real.contDiff_cosh.inv cosh_ne_zero

theorem hasDerivAt_gaussian (x : ℝ) :
    HasDerivAt gaussian (-x * gaussian x) x := by
  have hinner : HasDerivAt (fun y => -y ^ 2 / 2) (-x) x := by
    have h := (hasDerivAt_pow 2 x).neg.div_const (2 : ℝ)
    have hval : (-(↑2 * x ^ (2 - 1)) / 2 : ℝ) = -x := by simp; ring
    exact h.congr_deriv hval
  have hexp : HasDerivAt (Real.exp ∘ fun y => -y ^ 2 / 2)
      (Real.exp (-x ^ 2 / 2) * -x) x :=
    (Real.hasDerivAt_exp (-x ^ 2 / 2)).comp x hinner
  have hfun : gaussian = Real.exp ∘ fun y => -y ^ 2 / 2 := rfl
  rw [hfun]
  refine hexp.congr_deriv ?_
  simp [mul_comm]

theorem deriv_gaussian (x : ℝ) : deriv gaussian x = -x * gaussian x :=
  (hasDerivAt_gaussian x).deriv

theorem contDiff_gaussian : ContDiff ℝ ⊤ gaussian :=
  Real.contDiff_exp.comp ((contDiff_id.pow 2).neg.div_const 2)

theorem hasDerivAt_softplus (x : ℝ) : HasDerivAt softplus (Real.sigmoid x) x := by
  have hsum : HasDerivAt (fun y => 1 + Real.exp y) (Real.exp x) x :=
    (Real.hasDerivAt_exp x).const_add 1
  have hlog : HasDerivAt (fun y => Real.log (1 + Real.exp y))
      (Real.exp x / (1 + Real.exp x)) x :=
    hsum.log (one_add_exp_ne_zero x)
  have hform : Real.exp x / (1 + Real.exp x) = Real.sigmoid x := by
    rw [Real.sigmoid_def]
    have hx : 1 + Real.exp (-x) ≠ 0 := one_add_exp_ne_zero (-x)
    field_simp [hx, one_add_exp_ne_zero x]
    rw [mul_add, mul_one, ← Real.exp_add, add_neg_cancel, Real.exp_zero]
    ring
  convert hlog using 1
  · rfl
  · exact hform.symm

theorem deriv_softplus (x : ℝ) : deriv softplus x = Real.sigmoid x :=
  (hasDerivAt_softplus x).deriv

theorem contDiff_softplus : ContDiff ℝ ⊤ softplus :=
  (contDiff_const.add Real.contDiff_exp).log one_add_exp_ne_zero

end

end OmnibiasAnalytic.Tower

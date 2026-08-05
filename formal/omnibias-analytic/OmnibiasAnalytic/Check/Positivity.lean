/-
Sound sum-of-squares / positivity lemmas (Mathlib-backed).

These are reusable *capability* lemmas that the integer `ZInterval` kernel cannot
express: genuine polynomial positivity facts closed by Mathlib's `positivity` and
`nlinarith`. They back the sum-of-squares style obligations a certificate may
carry (e.g. a certified-positive quadratic form / discriminant condition).

Scope. As everywhere in `Check`, a green build certifies the *implication* (if the
certificate's rational data is correct, the stated inequality holds). That the
data really encloses the intended analytic quantity is a trusted input.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

/-- **Sum of two squares is nonnegative** (the atomic SOS fact) over `ℚ`. -/
theorem sq_add_sq_nonneg (a b : ℚ) : 0 ≤ a ^ 2 + b ^ 2 := by positivity

/-- **Sum of three squares is nonnegative** over `ℚ`. -/
theorem sq_add_sq_add_sq_nonneg (a b c : ℚ) : 0 ≤ a ^ 2 + b ^ 2 + c ^ 2 := by
  positivity

/-- **Strict positivity of a squared term plus a positive slack** -- the shape a
certified SOS decomposition `q = (linear)² + slack` with `slack > 0` produces. -/
theorem sq_add_pos_pos {t s : ℚ} (h : 0 < s) : 0 < t ^ 2 + s := by
  nlinarith [sq_nonneg t]

/-- **Quadratic-form positivity via the discriminant (an SOS certificate).**
If the leading coefficient is positive and the discriminant is negative
(`b² < 4ac`), the quadratic `a x² + b x + c` is strictly positive for every `x`.
The witnessing sum-of-squares identity is `4a(ax²+bx+c) = (2ax+b)² + (4ac - b²)`. -/
theorem quad_pos {a b c x : ℚ} (ha : 0 < a) (hdisc : b ^ 2 < 4 * a * c) :
    0 < a * x ^ 2 + b * x + c := by
  nlinarith [sq_nonneg (2 * a * x + b), mul_pos ha ha]

end OmnibiasAnalytic.Check

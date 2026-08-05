/-
Sound rational enclosed-sign lemmas (Mathlib-backed).

These are the `ℚ` analogues of the integer `ZInterval` sign obligations in
`formal/omnibias-verified-kernel` (`Omnibias/Certificate.lean`). Working over `ℚ`
(rather than scaling rational endpoints to a common integer denominator before
entering the kernel) is the concrete gain from depending on Mathlib: the Python
bridge can emit the certificate's rational endpoints directly and close the
endpoint-sign side condition with `norm_num`.

Scope. Exactly as in the kernel: a green build certifies the implication "if the
certificate's rational bound is correct, the stated sign holds". That the bound
really encloses the intended analytic quantity is a trusted (unverified) input.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

/-- **Enclosed-quantity positivity.** A value at least as large as a strictly
positive lower bound is strictly positive. -/
theorem enclosed_pos {x lo : ℚ} (hlo : lo ≤ x) (h : 0 < lo) : 0 < x :=
  lt_of_lt_of_le h hlo

/-- **Enclosed-quantity negativity.** A value no larger than a strictly negative
upper bound is strictly negative (the dual obligation, used to *exclude* a
property). -/
theorem enclosed_neg {x hi : ℚ} (hhi : x ≤ hi) (h : hi < 0) : x < 0 :=
  lt_of_le_of_lt hhi h

end OmnibiasAnalytic.Check

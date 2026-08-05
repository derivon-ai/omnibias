/-
Kernel-verified positive-definiteness via the LDLᵀ inertia vector.

The strict-local-minimum certificate (`omnibias.verify.certify_trained_min`) proves a
symmetric interval Hessian is positive definite by an interval LDLᵀ factorisation: it
is PD exactly when every pivot interval `Dⱼ` of the factorisation is strictly positive
(`Dⱼ.lo > 0`).  Previously the Python bridge collapsed that whole inertia vector to a
*single* scalar `eig_min > 0` before it reached the kernel; this module lets the kernel
re-check the **full n-pivot inertia vector** instead -- a genuine interval-matrix
obligation rather than its scalar shadow.

Soundness / scope.  The interval LDLᵀ factorisation (that these intervals really are the
pivots `Dⱼ` of the symmetric matrix box) is computed by the *unverified* Python interval
arithmetic and is a **trusted input** here -- the same trust base as the previous
`eig_min` scalar, only finer-grained.  What the kernel certifies, sorry-free, is the
finite sign fact: *given* the pivot intervals, every one is strictly positive, so every
exact pivot drawn from them is positive, hence (Sylvester's law of inertia, `S = L D Lᵀ`
a congruence) every point matrix in the box is positive definite.  Interval `*` / `/` are
deliberately kept out of the Mathlib-free kernel; the division-bearing factorisation
stays on the trusted Python side, and only its `+`/comparison certificate lands here.
-/

import Omnibias.Certificate

namespace Omnibias

/-- Are all LDLᵀ pivot intervals strictly positive (each `lo > 0`)?  A `Bool`
decision procedure over the pivot list the Python interval LDLᵀ produced. -/
def allPivotsPos : List ZInterval → Bool
  | [] => true
  | d :: ds => decide (0 < d.lo) && allPivotsPos ds

/-- **Certified positive-definite LDLᵀ inertia.**  If every pivot interval of the
symmetric matrix box's LDLᵀ factorisation is strictly positive, then any exact pivot
value drawn from any of them is strictly positive.  With the factorisation as a trusted
Python input, this is the whole negative-inertia-zero certificate: by Sylvester's law of
inertia every point matrix in the box is positive definite.  Chains the proven
`ZInterval.pos_of_mem_of_lo_pos` per pivot. -/
theorem matrix_positive_definite_certified :
    ∀ (pivots : List ZInterval), allPivotsPos pivots = true →
      ∀ (x : Int) (d : ZInterval), d ∈ pivots → ZInterval.Mem x d → 0 < x := by
  intro pivots
  induction pivots with
  | nil =>
      intro _ x d hd
      cases hd
  | cons e es ih =>
      intro h x d hd hx
      simp only [allPivotsPos, Bool.and_eq_true, decide_eq_true_eq] at h
      cases List.mem_cons.mp hd with
      | inl heq => subst heq; exact ZInterval.pos_of_mem_of_lo_pos hx h.1
      | inr hmem => exact ih h.2 x d hmem hx

end Omnibias

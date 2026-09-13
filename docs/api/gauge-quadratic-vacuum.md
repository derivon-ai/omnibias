# Finite quadratic vacuum root certificates

`omnibias.geometry.gauge.transfer.quadratic_vacuum` checks a rational
matrix Riccati residual for a coupled harmonic reference. It does not
certify the nonlinear Wilson vacuum.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer.quadratic_vacuum import (
    quadratic_vacuum_root, replay_quadratic_vacuum_root_certificate,
)

result = quadratic_vacuum_root(
    [[4, -1], [-1, 4]],
    [[Q(248, 125), Q(-63, 250)], [Q(-63, 250), Q(248, 125)]],
    frequency_lower=Q(17, 10),
)
assert result["status"] == "PASS"
assert result["witness"]["arithmetic"]["root_error_upper"] == "19/214500"
assert replay_quadratic_vacuum_root_certificate(result["certificate"])
assert not result["continuum_claim"]
```

For symmetric rational stiffness \(K\) and root witness \(P\), derive
\(p=\min_i(P_{ii}-\sum_{j\ne i}|P_{ij}|)\) and
\(r=\max_i\sum_j|(K-P^2)_{ij}|\). The sufficient gate is
\(p>0\) and \(s^2\le p^2-r\), for the supplied positive
`frequency_lower` \(s\). It earns

\[
K\ge s^2I,\qquad
\|\sqrt K-P\|\le r/(p+s),\qquad
\|(X\mapsto\sqrt KX+X\sqrt K)^{-1}\|\le1/(2s).
\]

The last two inequalities follow from the convergent Sylvester integral
\(D=\int_0^\infty e^{-t\sqrt K}(K-P^2)e^{-tP}dt\). No commutation
assumption or floating-point eigenvalue is used. These bounds apply to the
operator norm. Negative square roots are refused even when their square is
correct. A failed sufficient gate is `INCONCLUSIVE`.

The exact Hamiltonian certified here is
\(H=\kappa(-\Delta)/2+q^TKq/(2\kappa)\) on \(\mathbb R^n\),
for any \(\kappa>0\). Its vacuum is the Gaussian with precision
\(\sqrt K/\kappa\); its gap is at least \(s\). Inputs must already
specify the complete positive coordinate space. Null directions are not
automatically discarded or treated as gauge.

`scale` is a positive rational normalization. Under
\((K,P,s)\mapsto(\sigma^2K,\sigma P,\sigma s)\), the normalized
root-error and Sylvester-inverse bounds are unchanged. No renormalization
transformation, future-scale induction or nonlinear remainder is inferred
from this arithmetic covariance.

The certificate type is `quadratic_vacuum_root_v1`. Canonical replay
recomputes the full payload and rejects changed bounds, claims, flags and
scope, even after rehashing. Analytic implications are written proofs;
`theorem_prover_verified` and `mathlib_verified` remain false.

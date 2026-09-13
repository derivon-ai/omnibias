# Scale-normalized Schur budgets

`scale_gap_budget` checks finite rational comparison data. It does not
establish that a Yang--Mills blocking transformation has these constants.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    scale_gap_budget, replay_scale_gap_budget_certificate,
)

steps = [[Q(1, 2), Q(1, 16 * 2**n), Q(1, 4 * 2**n), 1]
         for n in range(16)]
result = scale_gap_budget(1, 2, steps, geometric_envelope=[Q(1, 8), Q(1, 2)])
assert result["status"] == "PASS"
assert result["witness"]["conditional_all_stage_physical_floor"] == "3/2"
assert replay_scale_gap_budget_certificate(result["certificate"])
assert not result["actual_rg_steps_verified"]
assert not result["physical_gap_verified"]
assert not result["continuum_claim"]
```

Each step is `(spacing_ratio, compression_loss, relative_cross_norm,
high_buffer)`, denoted \((s,\epsilon,b,h)\). Its analytic input would be
the actual vacuum-centered comparison for the next lattice Hamiltonian,
divided by \(s\), with low block at least \((1-\epsilon)d\), the
**entire** complementary block at least \((1+h)d\), and cross norm at
most \(bd\). Here \(d\) is the preceding lattice gap floor.

For \(h>0,\epsilon\ge0\), completing a square bounds that comparison
from below by \((1-\delta)d\), where
\(\delta=\epsilon+b^2/h\). The finite recurrence is therefore

\[
a_{n+1}=s_na_n,\quad d_{n+1}=s_n(1-\delta_n)d_n,\quad
\frac{d_N}{a_N}=\frac{d_0}{a_0}\prod_{n<N}(1-\delta_n).
\]

The gate checks \(0\le\delta_n<1\), the Schur residual, exact spacing
cancellation and the finite product bound
\(\prod(1-\delta_n)\ge1-\sum\delta_n\).
For a supplied geometric envelope \(\delta_n\le c\theta^n\), it
checks only the **provided prefix** and the rational condition
\(c/(1-\theta)<1\). The field
`conditional_all_stage_physical_floor` is
\((d_0/a_0)(1-c/(1-\theta))\); using it beyond that prefix requires
an analytic induction establishing the same operator hypotheses and
envelope at every later scale. The future-envelope flag stays false.

Persistent losses can make every finite step positive while sending the
physical lower bound to zero. For \(s_n=1/2,\delta_n=1/2\),
\(d_N/a_N=(d_0/a_0)2^{-N}\). Positive contraction margins by themselves
do not establish summability.

The separate `ensemble_laws.scale_gap_formal` runner builds parametric
rational Lean theorems for this algebra, with an invalid strengthened
product bound as a negative control. The certificate type
`scale_gap_budget_v1` itself earns no formal or physical claim. Canonical
replay rejects rehashed changes to budgets or unverified premise flags.

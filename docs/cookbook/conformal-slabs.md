# Conformal slabs

Split conformal on 19 residual magnitudes uses the finite-sample
index `ceil((n+1)(1-alpha))`. For `alpha = 0.1` that index is 18
and `q = 0.88`.

| Register | Meaning |
|---|---|
| Sound enclosure | certain, about the function |
| Conformal | marginal coverage, exchangeability |
| Model-based | coverage only if the model is right |

```python
from omnibias.core.uncertainty import (
    GuaranteeKind,
    UncertaintyInterval,
    combine_enclosure_with_conformal,
    honesty_payload,
    refuse_conformal_seal,
    split_conformal,
    worked_example,
)

ex = worked_example()
assert ex["q"] == 0.88
assert split_conformal([0.02, 0.05, 0.07, 0.09, 0.11, 0.14, 0.16, 0.19, 0.22, 0.25, 0.28, 0.33, 0.38, 0.44, 0.51, 0.60, 0.72, 0.88, 1.15], alpha=0.1) == 0.88
box = UncertaintyInterval(2.31, 2.47, GuaranteeKind.SOUND_ENCLOSURE)
stmt = combine_enclosure_with_conformal(box, ex["q"], alpha=0.1)
assert abs(stmt.combined_lo - (2.31 - 0.88)) < 1e-12
raised = False
try:
    _ = box + UncertaintyInterval(-ex["q"], ex["q"], GuaranteeKind.CONFORMAL, level=0.9)
except TypeError:
    raised = True
assert raised
seal_raised = False
try:
    refuse_conformal_seal(UncertaintyInterval(-1.0, 1.0, GuaranteeKind.CONFORMAL, level=0.9))
except ValueError:
    seal_raised = True
assert seal_raised
assert honesty_payload()["conformal_sealed"] is False
```
